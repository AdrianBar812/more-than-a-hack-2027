"""LightGBM possession model: leave-one-match-out on degraded DFL, Viterbi decoding, ATD output.

    python pipeline/train_lgb.py --ref-seconds 3 --out-dir data/pipeline/lgb_v1

Stages (all in one run):
  1. LOMO: for each of the 7 DFL matches, train on the other 6 (all degradation seeds) and predict
     the held-out match with its own seed-2 degradation -> out-of-fold probabilities.
  2. Decoder: Viterbi over -log p with a switch cost (H<->A) and a dead-ball cost (live<->D),
     grid-searched on the out-of-fold probabilities, scored with possession_metrics.evaluate
     (spellF1) against the DFL labels debounced to --ref-seconds.
  3. Final model on all 7 matches, ATD probabilities, decoded with the best costs -> CSV.

Nothing from the competition match's labels is used. The public window is read once at the very end
as a pipeline check (it is never used to choose a parameter).
"""

import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import labels as lab  # noqa: E402
import possession_metrics as pm  # noqa: E402

FEAT = os.environ.get("POSS_FEAT_DIR") or os.path.join(ROOT, "data", "pipeline", "features")
GAP = 200
SEEDS = (0, 1, 2)
PARAMS = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=31,
              min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1,
              lambda_l2=5.0, verbose=-1, seed=0, num_threads=16)
ROUNDS = 300
STEP = 5          # train on every 5th frame (adjacent frames are near-duplicates)
COLS = None       # feature subset (indices) set from --drop; None = all
WEIGHTS = None    # dict "<match>_s<seed>_h<half>" -> per-row weights (adversarial density ratio), or None


TWO_STAGE = False  # idea 1: dead-vs-live model + home-vs-away model trained on live frames only


class TwoStage:
    """P(D) from a binary model on all frames, P(A | live) from a binary model on live frames only.

    P = [(1 - pD)(1 - pA), (1 - pD) pA, pD]. Mirrors the two-branch design of Borghesi et al. (2023),
    whose possession loss is switched off on dead frames.
    """

    def __init__(self, m_dead, m_ha):
        self.m_dead, self.m_ha = m_dead, m_ha

    def predict(self, X):
        pd_ = self.m_dead.predict(X)
        pa = self.m_ha.predict(X)
        return np.column_stack([(1 - pd_) * (1 - pa), (1 - pd_) * pa, pd_])

    def feature_importance(self, kind="gain"):
        a, b = self.m_dead.feature_importance(kind), self.m_ha.feature_importance(kind)
        return a / max(a.sum(), 1e-9) + b / max(b.sum(), 1e-9)

    def save_model(self, path):
        self.m_dead.save_model(path.replace(".txt", "_dead.txt"))
        self.m_ha.save_model(path.replace(".txt", "_ha.txt"))


def sel(X):
    return X if COLS is None else X[:, COLS]


def matches():
    return sorted({os.path.basename(p)[:14] for p in glob.glob(os.path.join(FEAT, "DFL*_s0.npz"))})


def train_set(ms, ref, offset=0):
    X, y, w = [], [], []
    for m in ms:
        ys = {}
        for s in SEEDS:
            z = np.load(os.path.join(FEAT, "%s_s%d.npz" % (m, s)))
            for h in (1, 2):
                if h not in ys:
                    ys[h] = lab.debounce(z["y%d" % h], ref)
                X.append(sel(z["X%d" % h][offset::STEP]))
                y.append(ys[h][offset::STEP])
                if WEIGHTS is not None:
                    w.append(WEIGHTS["%s_s%d_h%d" % (m, s, h)] if offset == 0 else np.ones(len(y[-1]), np.float32))
    return np.vstack(X), np.concatenate(y), (np.concatenate(w) if w else None)


class Ensemble:
    """K LightGBM models, member j trained on frames offset j (mod STEP) with its own bagging seed."""

    def __init__(self, models):
        self.models = models

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)

    def feature_importance(self, kind="gain"):
        return np.mean([m.feature_importance(kind) for m in self.models], axis=0)

    def save_model(self, path):
        for j, m in enumerate(self.models):
            m.save_model(path.replace(".txt", "_%d.txt" % j))


def fit(ms, ref, names, ensemble=1):
    import lightgbm as lgb
    models = []
    if TWO_STAGE:
        X, y, w = train_set(ms, ref)
        prm = dict(PARAMS, objective="binary")
        prm.pop("num_class")
        md = lgb.train(prm, lgb.Dataset(X, (y == 2).astype(int), weight=w, feature_name=names), ROUNDS)
        live = y < 2
        mh = lgb.train(prm, lgb.Dataset(X[live], (y[live] == 1).astype(int), weight=None if w is None else w[live], feature_name=names), ROUNDS)
        return TwoStage(md, mh)
    for j in range(ensemble):
        X, y, w = train_set(ms, ref, offset=j % STEP)
        prm = dict(PARAMS, seed=j, bagging_seed=j, feature_fraction_seed=j)
        models.append(lgb.train(prm, lgb.Dataset(X, y, weight=w, feature_name=names), ROUNDS))
    return models[0] if ensemble == 1 else Ensemble(models)


def viterbi(logp, c_sw, c_d):
    nll = -np.log(np.clip(logp, 1e-6, 1.0))
    cost = np.array([[0, c_sw, c_d], [c_sw, 0, c_d], [c_d, c_d, 0]], dtype=np.float64)
    n = len(nll)
    score = nll[0].copy()
    back = np.zeros((n, 3), dtype=np.int8)
    ar = np.arange(3)
    for t in range(1, n):
        cand = score[:, None] + cost
        b = cand.argmin(axis=0)
        back[t] = b
        score = cand[b, ar] + nll[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = score.argmin()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


MERGE_S = 3.0      # merge decoded H/A spells shorter than this into a neighbour (round 8: +0.006 out of fold)


def decode(p, c_sw, c_d):
    """Viterbi over -log p, then merge H/A spells shorter than MERGE_S seconds (labels.debounce)."""
    path = viterbi(p, c_sw, c_d)
    return lab.debounce(path, MERGE_S).astype(np.int8) if MERGE_S > 0 else path


def combine(halves):
    """Concatenate halves with an unscored gap -> (array, blocks)."""
    parts, blocks, pos = [], [], 0
    for h, a in enumerate(halves, 1):
        parts.append(a)
        blocks.append((h, pos, pos + len(a) - 1))
        pos += len(a) + GAP
        parts.append(np.full((GAP,) + a.shape[1:], np.nan if a.ndim > 1 else pm.X, dtype=a.dtype if a.ndim == 1 else float))
    return np.concatenate(parts), blocks


# ---- decoder search (runs in worker processes) ---------------------------------------------
_OOF = {}


def _task(args):
    match, c_sw, c_d, mode = args
    probs, gt, blocks = _OOF[match]
    pred = np.full(len(gt), pm.X, dtype=np.int8)
    for _h, a, e in blocks:
        p = probs[a:e + 1]
        pred[a:e + 1] = p.argmax(1) if mode == "raw" else decode(p, c_sw, c_d)
    r = pm.evaluate(gt, pred)
    return r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-seconds", type=float, default=3.0)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "data", "pipeline", "lgb_v1"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--ensemble", type=int, default=1, help="number of models averaged")
    ap.add_argument("--groups", default="", help="extra feature groups to use from the cache: near,short,restart (comma separated)")
    ap.add_argument("--exclude", default="", help="comma separated feature-name prefixes to drop (e.g. sw_acc,sw_spike)")
    ap.add_argument("--merge-seconds", type=float, default=3.0,
                    help="merge decoded H/A spells shorter than this (0 = off; the v2 candidate without it used 0)")
    ap.add_argument("--costs", type=float, nargs=2, metavar=("SWITCH", "DEAD"),
                    help="use these decoder costs instead of the out-of-fold best (the v2 candidate used 16 20)")
    ap.add_argument("--two-stage", action="store_true", help="idea 1: separate dead-vs-live and home-vs-away models")
    ap.add_argument("--drop", type=int, default=0, help="drop the first K features of data/pipeline/prune.json")
    ap.add_argument("--adv-weights", action="store_true", help="weight DFL rows by the adversarial density ratio")
    ap.add_argument("--public-check", action="store_true",
                    help="score the decoded ATD output on the public window (pipeline check only)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    global COLS, WEIGHTS, TWO_STAGE, MERGE_S
    TWO_STAGE = args.two_stage
    MERGE_S = args.merge_seconds
    all_names = json.load(open(os.path.join(FEAT, "feature_names.json")))
    names = all_names
    extra = {"near": "nb_", "short": "sw_", "restart": "rs_"}
    chosen = tuple(extra[g] for g in args.groups.split(",") if g)
    excl = tuple(x for x in args.exclude.split(",") if x)
    keep = [i for i, n in enumerate(all_names)
            if (not n.startswith(tuple(extra.values())) or n.startswith(chosen)) and not (excl and n.startswith(excl))]
    if len(keep) != len(all_names):
        COLS = keep
        names = [all_names[i] for i in keep]
    if args.drop:
        gone = set(json.load(open(os.path.join(ROOT, "data", "pipeline", "prune.json")))["order"][:args.drop])
        COLS = [i for i, n in enumerate(all_names) if n not in gone]
        names = [all_names[i] for i in COLS]
    if args.adv_weights:
        WEIGHTS = dict(np.load(os.path.join(ROOT, "data", "pipeline", "adv_weights.npz")))
    print("features: %d%s" % (len(names), "  adversarial weights" if WEIGHTS else ""), flush=True)
    ms = matches()
    R = args.ref_seconds
    t0 = time.time()

    # ---- 1. leave-one-match-out ----
    for m in ms:
        model = fit([o for o in ms if o != m], R, names, args.ensemble)
        z = np.load(os.path.join(FEAT, "%s_s2.npz" % m))
        probs, gts = [], []
        for h in (1, 2):
            probs.append(model.predict(sel(z["X%d" % h])))
            gts.append(lab.debounce(z["y%d" % h], R).astype(np.int8))
        P, blocks = combine(probs)
        G, _ = combine(gts)
        G = np.where(np.isnan(P[:, 0]), pm.X, G).astype(np.int8)
        _OOF[m] = (P, G, blocks)
        print("LOMO %s done (%.0fs)" % (m[-4:], time.time() - t0), flush=True)
    np.savez(os.path.join(args.out_dir, "oof.npz"), **{m + "_P": _OOF[m][0] for m in ms}, **{m + "_G": _OOF[m][1] for m in ms})

    # ---- 2. decoder search ----
    pool = mp.get_context("fork").Pool(args.workers)
    raw = np.array(pool.map(_task, [(m, 0, 0, "raw") for m in ms]), dtype=float)
    print("\nraw argmax      OOF mean spellF1 %.4f  macroF1 %.4f  spells pred/true %.2f" % (
        raw[:, 0].mean(), raw[:, 1].mean(), raw[:, 2].sum() / raw[:, 3].sum()), flush=True)
    grid = [(a, b) for a in (8, 16, 32, 64, 128) for b in (3, 6, 12, 20, 32, 48, 80)]
    results = []
    for c_sw, c_d in grid:
        r = np.array(pool.map(_task, [(m, c_sw, c_d, "viterbi") for m in ms]), dtype=float)
        results.append((r[:, 0].mean(), r[:, 1].mean(), r[:, 2].sum() / r[:, 3].sum(), c_sw, c_d))
    results.sort(reverse=True)
    print("top decoder settings (OOF mean over 7 held-out matches):")
    for s, mf, ratio, a, b in results[:6]:
        print("  switch cost %-3s dead cost %-3s  spellF1 %.4f  macroF1 %.4f  spells pred/true %.2f" % (a, b, s, mf, ratio))
    best = results[0]
    c_sw, c_d = best[3], best[4]
    if args.costs:
        c_sw, c_d = args.costs
        print("decoder costs fixed by --costs: switch %g, dead %g" % (c_sw, c_d))

    # ---- 3. final model, ATD ----
    model = fit(ms, R, names, args.ensemble)
    model.save_model(os.path.join(args.out_dir, "model.txt"))
    imp = model.feature_importance("gain")
    order = np.argsort(-imp)[:15]
    print("\ntop features (gain share):", ", ".join("%s %.1f%%" % (names[i], 100 * imp[i] / imp.sum()) for i in order[:8]))
    a = np.load(os.path.join(FEAT, "atd.npz"))
    codes = np.full(199319, pm.X, dtype=np.int8)
    raw_codes = codes.copy()
    probs_atd = {}
    for h in (1, 2):
        p = model.predict(sel(a["X%d" % h]))
        s = int(a["start%d" % h])
        probs_atd["P%d" % h], probs_atd["start%d" % h] = p, s
        codes[s:s + len(p)] = decode(p, c_sw, c_d)
        raw_codes[s:s + len(p)] = p.argmax(1)
    np.savez(os.path.join(args.out_dir, "atd_probs.npz"), **probs_atd)
    letters = np.array(list("HADX"))[codes]
    csv = os.path.join(args.out_dir, "lgb_decoded.csv")
    with open(csv, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("%d,%s\n" % (i, l) for i, l in enumerate(letters))
    json.dump({"ref_seconds": R, "switch_cost": c_sw, "dead_cost": c_d, "oof_spell_f1": best[0], "oof_macro_f1": best[1],
               "raw_oof_spell_f1": float(raw[:, 0].mean()), "grid": results}, open(os.path.join(args.out_dir, "decoder.json"), "w"), indent=1)

    if not args.public_check:
        print("wrote", csv, "(%.0fs total; public window not read)" % (time.time() - t0))
        return
    # ---- pipeline check on the public window (read once, nothing is chosen from it) ----
    gt = pm.label_codes(pm.read_label_csv(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")))
    for name, c in (("raw argmax", raw_codes), ("decoded", codes)):
        r = pm.evaluate(gt, c)
        print("PUBLIC WINDOW (check only) %-10s spellF1 %.4f  macroF1 %.4f  spells %d (true %d)  F1 H %.3f A %.3f D %.3f  D recall %.3f" % (
            name, r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"],
            r["per_class"]["H"]["f1"], r["per_class"]["A"]["f1"], r["per_class"]["D"]["f1"], r["per_class"]["D"]["recall"]))
    print("wrote", csv, "(%.0fs total)" % (time.time() - t0))


if __name__ == "__main__":
    main()
