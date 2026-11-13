"""Tune the HMM (starter/tested_models/HMM.py) on degraded DFL matches, scored with spellF1.

The old tuner (tune_hmm_from_dfl.py) only turned event-duration means into three transition
times. This one runs the real HMM on DFL matches degraded to look like the ATD
(pipeline/degrade.py) and searches every HMM parameter against the project's own metric
(evaluation/possession_metrics.evaluate), so the answer is "what scores best on other games".

Protocol (nothing from the competition match's labels is used):
  * tune on matches 0-3, degradation seeds 0 and 1          (8 tasks per trial)
  * held out: matches 4-6, a fresh degradation seed 2       (never seen while searching)
  * reference labels: DFL ball status/possession, debounced to --ref-seconds (default 3 s)
  * search: trial 0 = current HMM.py defaults, then random search, then local refinement

    python pipeline/tune_hmm.py --random 60 --refine 60 --out data/pipeline/hmm_tuned.json
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
sys.path.insert(0, os.path.join(ROOT, "starter", "tested_models"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, HERE)
import HMM as hmm  # noqa: E402
import labels as lab  # noqa: E402
import possession_metrics as pm  # noqa: E402

FPS = 25
PITCH = (105.3, 68.0)
GAP = 200  # unscored frames between the two halves of a match
CANON = os.path.join(ROOT, "data", "pipeline", "dfl_canon")

# name -> (low, high, kind). kind: "log" log-uniform, "lin" uniform, "int", "bool"
SPACE = {
    # v2: widened after round 1 ended on the bounds of LOOSE_BALL_DISTANCE, D_LOGLIK_STRONG,
    # D_LOGLIK_STILL, AMBIGUOUS_LLR and MIN_DEAD_SECONDS (v1 ranges in hmm_tuned.json history)
    "SECONDS_BETWEEN_SWITCHES": (5.0, 150.0, "log"),
    "SECONDS_BETWEEN_STOPS": (10.0, 300.0, "log"),
    "SECONDS_OF_DEAD_BALL": (3.0, 120.0, "log"),
    "CLOSER_SLOPE": (0.02, 2.0, "log"),
    "LOOSE_BALL_DISTANCE": (0.5, 20.0, "log"),
    "STILL_SPEED": (0.3, 4.0, "log"),
    "STILL_FRAMES": (10, 200, "int"),
    "D_LOGLIK_STRONG": (0.0, 6.0, "lin"),
    "D_LOGLIK_STILL": (0.0, 4.0, "lin"),
    "D_LOGLIK_PLAY": (-3.0, 0.0, "lin"),
    "D_LOGLIK_MISSING": (-1.0, 2.0, "lin"),
    "FLIGHT_SPEED": (6.0, 30.0, "lin"),
    "FLIGHT_DISTANCE": (1.5, 15.0, "lin"),
    "AMBIGUOUS_LLR": (0.05, 3.0, "log"),
    "MIN_DEAD_SECONDS": (0.0, 20.0, "lin"),
    "USE_FLIGHT": (0, 1, "bool"),
    "USE_AMBIGUOUS": (0, 1, "bool"),
}


def defaults():
    d = {k: getattr(hmm, k) for k in SPACE if hasattr(hmm, k)}
    d["USE_FLIGHT"], d["USE_AMBIGUOUS"] = 1, 1
    return d


def sample(rng):
    out = {}
    for k, (lo, hi, kind) in SPACE.items():
        if kind == "log":
            out[k] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        elif kind == "lin":
            out[k] = float(rng.uniform(lo, hi))
        elif kind == "int":
            out[k] = int(rng.integers(lo, hi + 1))
        else:
            out[k] = int(rng.integers(0, 2))
    return out


def perturb(base, rng, scale):
    out = dict(base)
    for k, (lo, hi, kind) in SPACE.items():
        if rng.random() > 0.35:  # change a third of the parameters per step
            continue
        if kind == "log":
            out[k] = float(np.clip(base[k] * np.exp(rng.normal(0, scale)), lo, hi))
        elif kind == "lin":
            out[k] = float(np.clip(base[k] + rng.normal(0, scale * (hi - lo) / 2), lo, hi))
        elif kind == "int":
            out[k] = int(np.clip(round(base[k] * np.exp(rng.normal(0, scale))), lo, hi))
        else:
            out[k] = int(1 - base[k]) if rng.random() < 0.2 else int(base[k])
    return out


# ---- per-process data cache ----------------------------------------------------------------
_CACHE = {}


def _half_features(ball, players, team):
    bx, by = ball[:, 0], ball[:, 1]
    dx = (players[:, :, 0] - bx[:, None]) * PITCH[0]
    dy = (players[:, :, 1] - by[:, None]) * PITCH[1]
    d = np.hypot(dx, dy)
    d_home = np.where(np.all(np.isnan(d[:, team == 0]), axis=1), np.nan, np.nanmin(np.where(np.isnan(d[:, team == 0]), np.inf, d[:, team == 0]), axis=1))
    d_away = np.where(np.all(np.isnan(d[:, team == 1]), axis=1), np.nan, np.nanmin(np.where(np.isnan(d[:, team == 1]), np.inf, d[:, team == 1]), axis=1))
    d_home[np.isinf(d_home)] = np.nan
    d_away[np.isinf(d_away)] = np.nan
    d_home[np.isnan(bx)] = np.nan
    d_away[np.isnan(bx)] = np.nan
    speed = hmm.ball_speed(bx, by, PITCH, lag=hmm.SPEED_LAG_FRAMES)
    return d_home, d_away, speed, bx, by


def load_task(match, seed, ref_seconds):
    key = (match, seed, ref_seconds)
    if key in _CACHE:
        return _CACHE[key]
    z = np.load(os.path.join(CANON, "%s_s%d.npz" % (match, seed)))
    feats, gts, blocks, pos = [], [], [], 0
    for h in (1, 2):
        f = _half_features(z["h%d_ball" % h], z["h%d_players" % h], z["h%d_team" % h])
        raw = lab.raw_labels(z["h%d_status" % h], z["h%d_poss" % h])
        gt = lab.debounce(raw, ref_seconds).astype(np.int8)
        T = len(gt)
        feats.append(f)
        gts.append(gt)
        blocks.append((h, pos, pos + T - 1))
        pos += T + GAP
        feats.append(tuple(np.full(GAP, np.nan) for _ in range(5)))
        gts.append(np.full(GAP, pm.X, dtype=np.int8))
    cat = tuple(np.concatenate([f[i] for f in feats]) for i in range(5))
    out = (cat, np.concatenate(gts), blocks)
    _CACHE[key] = out
    return out


def decode(feats, blocks, params):
    for k, v in params.items():
        if hasattr(hmm, k):
            setattr(hmm, k, v)
    d_home, d_away, speed, bx, by = feats
    ll = hmm.log_evidence(d_home, d_away, speed, bx, by)
    flight = hmm.in_flight(d_home, d_away, speed) if params.get("USE_FLIGHT", 1) else None
    amb = hmm.ambiguous_ball(ll, d_home, d_away) if params.get("USE_AMBIGUOUS", 1) else None
    labels = hmm.decode_graded(ll, blocks, hmm.transition_matrix(params["SECONDS_BETWEEN_SWITCHES"]),
                               flight, amb, int(round(params["MIN_DEAD_SECONDS"] * FPS)))
    return pm.label_codes(labels)


def smooth50(feats, blocks, window=50):
    """The published baseline on the same features: nearest player's team, D if no ball, majority vote."""
    d_home, d_away, _s, bx, _by = feats
    codes = np.full(len(d_home), -1, dtype=np.int8)
    seen = np.isfinite(d_home) & np.isfinite(d_away)
    codes[seen] = (d_away[seen] < d_home[seen]).astype(np.int8)
    out = np.full(len(codes), pm.X, dtype=np.int8)
    for _h, first, last in blocks:
        c = codes[first:last + 1]
        hc = np.concatenate(([0], np.cumsum(c == 0)))
        ac = np.concatenate(([0], np.cumsum(c == 1)))
        i = np.arange(len(c))
        lo, hi = np.maximum(0, i - window), np.minimum(len(c), i + window + 1)
        home, away = hc[hi] - hc[lo], ac[hi] - ac[lo]
        res = np.full(len(c), pm.D, dtype=np.int8)
        s = (home + away) > 0
        res[s] = np.where(home[s] >= away[s], pm.H, pm.A)
        out[first:last + 1] = res
    return out


def run_task(args):
    match, seed, ref_seconds, params, mode = args
    feats, gt, blocks = load_task(match, seed, ref_seconds)
    pred = smooth50(feats, blocks) if mode == "smooth50" else decode(feats, blocks, params)
    r = pm.evaluate(gt, pred)
    return r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]


def score(pool, tasks, params, ref_seconds, mode="hmm"):
    res = pool.map(run_task, [(m, s, ref_seconds, params, mode) for m, s in tasks])
    a = np.array(res, dtype=float)
    return float(a[:, 0].mean()), float(a[:, 1].mean()), float(a[:, 2].sum() / a[:, 3].sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--random", type=int, default=60)
    ap.add_argument("--refine", type=int, default=60)
    ap.add_argument("--ref-seconds", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "pipeline", "hmm_tuned.json"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", help="JSON from an earlier run to warm-start the search from")
    args = ap.parse_args()

    matches = sorted({os.path.basename(p)[:14] for p in glob.glob(os.path.join(CANON, "*_s0.npz"))})
    train = [(m, s) for m in matches[:4] for s in (0, 1)]
    held = [(m, 2) for m in matches[4:]]
    rng = np.random.default_rng(args.seed)
    pool = mp.get_context("fork").Pool(args.workers)
    R = args.ref_seconds

    base = defaults()
    best, best_p = score(pool, train, base, R), base
    log = [{"trial": 0, "kind": "defaults", "train_spell": best[0], "train_macro": best[1], "params": base}]
    print("trial   0 defaults        train spellF1 %.4f macroF1 %.4f  spells pred/gt %.2f" % best, flush=True)
    if args.init:  # warm start from an earlier round; kept only if it beats the defaults on train
        with open(args.init) as handle:
            start = json.load(handle)["best_params"]
        s = score(pool, train, start, R)
        log.append({"trial": 0, "kind": "init", "train_spell": s[0], "train_macro": s[1], "params": start})
        print("trial   0 init            train spellF1 %.4f macroF1 %.4f  spells pred/gt %.2f" % s, flush=True)
        if s[0] > best[0]:
            best, best_p = s, start
    t0 = time.time()
    total = args.random + args.refine
    for i in range(1, total + 1):
        if i <= args.random:
            cand, kind = sample(rng), "random"
        else:
            frac = (i - args.random) / max(args.refine, 1)
            cand, kind = perturb(best_p, rng, 0.5 * (1 - frac) + 0.1), "refine"
        s = score(pool, train, cand, R)
        log.append({"trial": i, "kind": kind, "train_spell": s[0], "train_macro": s[1], "params": cand})
        if s[0] > best[0]:
            best, best_p = s, cand
            print("trial %3d %-8s NEW BEST train spellF1 %.4f macroF1 %.4f  spells pred/gt %.2f  (%.0fs)"
                  % (i, kind, s[0], s[1], s[2], time.time() - t0), flush=True)

    print("\n== held-out matches %s, fresh degradation seed ==" % [m[-4:] for m in matches[4:]])
    rows = [("HMM defaults", score(pool, held, base, R)),
            ("HMM tuned", score(pool, held, best_p, R)),
            ("smooth50 baseline", score(pool, held, base, R, mode="smooth50"))]
    for name, s in rows:
        print("  %-18s spellF1 %.4f  macroF1 %.4f  predicted/true possessions %.2f" % ((name,) + s))
    print("== robustness of the tuned params to the reference definition (held-out) ==")
    for r2 in (2.0, 3.0, 5.0):
        d, t, b = (score(pool, held, p, r2, mode=m)[0] for p, m in ((base, "hmm"), (best_p, "hmm"), (base, "smooth50")))
        print("  reference N=%.0fs  defaults %.4f  tuned %.4f  smooth50 %.4f" % (r2, d, t, b))

    with open(args.out, "w") as handle:
        json.dump({"ref_seconds": R, "best_params": best_p, "train_spell_f1": best[0],
                   "heldout": {n: {"spell_f1": s[0], "macro_f1": s[1]} for n, s in rows},
                   "trials": log}, handle, indent=1)
    print("\nbest params ->", args.out)
    for k, v in best_p.items():
        print("  %-26s %s  (default %s)" % (k, round(v, 3) if isinstance(v, float) else v,
                                            round(base[k], 3) if isinstance(base[k], float) else base[k]))


if __name__ == "__main__":
    main()
