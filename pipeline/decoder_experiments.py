"""Decoder-level experiments on the saved out-of-fold probabilities (no retraining).

Four ideas, each scored with possession_metrics.evaluate against the debounced DFL labels, and each reported
(a) as the best cell of its grid on all 7 matches (optimistic) and (b) with the cell chosen on the OTHER 6 matches
for every held-out match (honest), both as paired differences against the baseline Viterbi 16 / 20:

  1. debounce   merge decoded H/A spells shorter than N seconds into a neighbour
  2. vis-cost   H<->A switch cost that depends on whether the ball is visible
  3. duration   minimum-duration / soft short-run penalty decoder (graph-expanded Viterbi at 5 frames per step)
  4. count      dead cost chosen per match so the number of possessions matches a prior (mean count of the other matches)

    python pipeline/decoder_experiments.py --oof data/pipeline/lgb_v2
The public window is not read.
"""

import argparse
import itertools
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import labels as lab  # noqa: E402
import possession_metrics as pm  # noqa: E402
import train_lgb as T  # noqa: E402

DATA = {}   # match -> dict(P, G, blocks, miss)
STEP = 5


def load(oof_dir):
    o = dict(np.load(os.path.join(oof_dir, "oof.npz")))
    for m in T.matches():
        P, G = o[m + "_P"], o[m + "_G"]
        z = np.load(os.path.join(T.FEAT, "%s_s2.npz" % m))
        M, _ = T.combine([z["m1"].astype(float), z["m2"].astype(float)])
        ok = ~np.isnan(P[:, 0])
        d = np.diff(ok.astype(int))
        st, en = list(np.where(d == 1)[0] + 1), list(np.where(d == -1)[0])
        if ok[0]:
            st = [0] + st
        if ok[-1]:
            en += [len(P) - 1]
        DATA[m] = dict(P=P, G=G, blocks=list(zip(st, en)), miss=np.nan_to_num(M) > 0.5)


def nll_of(P):
    return -np.log(np.clip(P, 1e-6, 1.0))


def vit_tv(P, miss, sw_vis, sw_miss, dead):
    """3-state Viterbi whose H<->A cost depends on whether the ball is missing at the arrival frame."""
    nll = nll_of(P)
    cs = [np.array([[0, s, dead], [s, 0, dead], [dead, dead, 0]], float) for s in (sw_vis, sw_miss)]
    n = len(nll)
    score = nll[0].copy()
    back = np.zeros((n, 3), np.int8)
    ar = np.arange(3)
    for t in range(1, n):
        cand = score[:, None] + cs[int(miss[t])]
        b = cand.argmin(0)
        back[t] = b
        score = cand[b, ar] + nll[t]
    path = np.empty(n, np.int8)
    path[-1] = score.argmin()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def vit_dur(P, sw, dead, min_ha_s, lam_ha, min_d_s, lam_d=1e6):
    """Graph-expanded Viterbi at STEP frames per step: a class needs `min` seconds, leaving earlier costs
    lam * (missing fraction of the minimum) (lam = 1e6 is a hard minimum)."""
    n = len(P)
    nc = -(-n // STEP)
    nll = np.zeros((nc * STEP, 3))
    nll[:n] = nll_of(P)
    nll = nll.reshape(nc, STEP, 3).sum(1)
    Lc = [max(1, int(round(min_ha_s * 25 / STEP))), max(1, int(round(min_ha_s * 25 / STEP))), max(1, int(round(min_d_s * 25 / STEP)))]
    lam = [lam_ha, lam_ha, lam_d]
    nodes, idx = [], {}
    for s in range(3):
        for k in range(Lc[s]):
            idx[(s, k)] = len(nodes)
            nodes.append((s, k))
    N = len(nodes)
    cls = np.array([s for s, _ in nodes])
    C = np.full((N, N), np.inf)
    base = np.array([[0, sw, dead], [sw, 0, dead], [dead, dead, 0]], float)
    for (s, k), i in idx.items():
        if k < Lc[s] - 1:
            C[i, idx[(s, k + 1)]] = 0.0
        else:
            C[i, i] = 0.0
        pen = lam[s] * (Lc[s] - (k + 1)) / Lc[s] if k < Lc[s] - 1 else 0.0
        for s2 in range(3):
            if s2 != s:
                C[i, idx[(s2, 0)]] = base[s, s2] + pen
    score = nll[0][cls].copy()
    back = np.zeros((nc, N), np.int16)
    ar = np.arange(N)
    for t in range(1, nc):
        cand = score[:, None] + C
        b = cand.argmin(0)
        back[t] = b
        score = cand[b, ar] + nll[t][cls]
    path = np.empty(nc, np.int16)
    path[-1] = score.argmin()
    for t in range(nc - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return np.repeat(cls[path], STEP)[:n].astype(np.int8)


def vit_combo(P, miss, sw_vis, sw_miss, dead, min_ha_s, lam_ha, min_d_s, deb_s):
    """Ideas 1+2+3 together: duration graph, visibility-dependent H<->A cost, then merge spells < deb_s."""
    n = len(P)
    nc = -(-n // STEP)
    nll = np.zeros((nc * STEP, 3))
    nll[:n] = nll_of(P)
    nll = nll.reshape(nc, STEP, 3).sum(1)
    mc = np.zeros(nc * STEP)
    mc[:n] = miss
    mc = (mc.reshape(nc, STEP).mean(1) > 0.5).astype(int)
    Lc = [max(1, int(round(min_ha_s * 25 / STEP)))] * 2 + [max(1, int(round(min_d_s * 25 / STEP)))]
    lam = [lam_ha, lam_ha, 1e6]
    nodes, idx = [], {}
    for s_ in range(3):
        for k in range(Lc[s_]):
            idx[(s_, k)] = len(nodes)
            nodes.append((s_, k))
    N = len(nodes)
    cls = np.array([a for a, _ in nodes])
    Cs = []
    for sw in (sw_vis, sw_miss):
        base = np.array([[0, sw, dead], [sw, 0, dead], [dead, dead, 0]], float)
        C = np.full((N, N), np.inf)
        for (a, k), i in idx.items():
            if k < Lc[a] - 1:
                C[i, idx[(a, k + 1)]] = 0.0
            else:
                C[i, i] = 0.0
            pen = lam[a] * (Lc[a] - (k + 1)) / Lc[a] if k < Lc[a] - 1 else 0.0
            for a2 in range(3):
                if a2 != a:
                    C[i, idx[(a2, 0)]] = base[a, a2] + pen
        Cs.append(C)
    score = nll[0][cls].copy()
    back = np.zeros((nc, N), np.int16)
    ar = np.arange(N)
    for t in range(1, nc):
        cand = score[:, None] + Cs[mc[t]]
        b = cand.argmin(0)
        back[t] = b
        score = cand[b, ar] + nll[t][cls]
    path = np.empty(nc, np.int16)
    path[-1] = score.argmin()
    for t in range(nc - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    out = np.repeat(cls[path], STEP)[:n].astype(np.int8)
    return lab.debounce(out, deb_s).astype(np.int8) if deb_s > 0 else out


def score_match(m, decode):
    d = DATA[m]
    pred = np.full(len(d["G"]), pm.X, np.int8)
    for a, e in d["blocks"]:
        pred[a:e + 1] = decode(d["P"][a:e + 1], d["miss"][a:e + 1])
    r = pm.evaluate(d["G"], pred)
    return r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]


def decoder(kind, params):
    if kind == "base":
        return lambda P, miss: T.viterbi(P, 16, 20)
    if kind == "debounce":
        n_s = params
        return lambda P, miss: lab.debounce(T.viterbi(P, 16, 20), n_s).astype(np.int8)
    if kind == "vis":
        sv, sm = params
        return lambda P, miss: vit_tv(P, miss, sv, sm, 20)
    if kind == "dur":
        return lambda P, miss: vit_dur(P, 16, 20, *params)
    if kind == "combo":
        mh, lam, md, sv, sm, deb = params
        return lambda P, miss: vit_combo(P, miss, sv, sm, 20, mh, lam, md, deb)
    if kind == "dead":
        return lambda P, miss: T.viterbi(P, 16, params)
    raise ValueError(kind)


def task(a):
    kind, params, m = a
    return score_match(m, decoder(kind, params))


def run_grid(pool, kind, cells, ms):
    res = pool.map(task, [(kind, c, m) for c in cells for m in ms])
    return np.array(res, float).reshape(len(cells), len(ms), 4)


def report(name, cells, R, base, ms):
    S = R[:, :, 0]
    b = base[:, 0]
    best = int(np.argmax(S.mean(1)))
    # honest: choose the cell on the other matches, score it on the held-out one
    honest = np.array([S[int(np.argmax(np.delete(S, j, axis=1).mean(1))), j] for j in range(len(ms))])
    print("\n== %s" % name)
    for c, row, full in sorted(zip(cells, S.mean(1), R), key=lambda x: -x[1])[:5]:
        print("   %-26s spellF1 %.4f  macroF1 %.4f  pred/true %.2f" % (c, row, full[:, 1].mean(), full[:, 2].sum() / full[:, 3].sum()))
    d_best = S[best] - b
    d_hon = honest - b
    print("   best cell on all 7 : %s  %.4f  paired %+.4f  better in %d/7" % (cells[best], S[best].mean(), d_best.mean(), (d_best > 0).sum()))
    print("   honest (LOMO-chosen): %.4f  paired %+.4f  better in %d/7  per match %s" % (honest.mean(), d_hon.mean(), (d_hon > 0).sum(), np.round(d_hon, 3)))
    return d_hon.mean(), int((d_hon > 0).sum())


def combo_main(args):
    load(args.oof)
    ms = T.matches()
    pool = mp.get_context("fork").Pool(args.workers)
    base = run_grid(pool, "base", [None], ms)[0]
    dur = [(0.2, 1e6), (3, 1e6), (3, 32), (5, 1e6), (5, 32)]
    cells = [(a, l, 0.2, sv, sm, deb) for (a, l) in dur for (sv, sm) in ((16, 16), (16, 32), (16, 64)) for deb in (0, 2, 3)]
    R = run_grid(pool, "combo", cells, ms)
    print("baseline 16/20: %.4f" % base[:, 0].mean())
    report("1+2+3 combined (min H/A s, short-run penalty, min D s, switch visible, switch missing, debounce s)", cells, R, base, ms)
    best = int(np.argmax(R[:, :, 0].mean(1)))
    print("BEST_CELL", cells[best])
    np.save(os.path.join(ROOT, "data", "pipeline", "combo_cell.npy"), np.array(cells[best], float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combo", action="store_true")
    ap.add_argument("--oof", default=os.path.join(ROOT, "data", "pipeline", "lgb_v2"))
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    if args.combo:
        return combo_main(args)
    load(args.oof)
    ms = T.matches()
    pool = mp.get_context("fork").Pool(args.workers)
    base = run_grid(pool, "base", [None], ms)[0]
    print("baseline Viterbi 16/20: spellF1 %.4f macroF1 %.4f" % (base[:, 0].mean(), base[:, 1].mean()))
    out = {}
    cells = [0, 1, 2, 3, 4, 5]
    out["1 debounce"] = report("1. debounce the decoded output (N seconds)", cells, run_grid(pool, "debounce", cells, ms), base, ms)
    cells = list(itertools.product((8, 16, 32), (8, 16, 32, 64)))
    out["2 vis-cost"] = report("2. visibility-dependent switch cost (visible, missing)", cells, run_grid(pool, "vis", cells, ms), base, ms)
    cells = [(m_ha, lam, m_d) for m_ha in (0.2, 2, 3, 5) for lam in (1e6, 32, 12) for m_d in (0.2, 2, 5)]
    cells = [c for c in cells if not (c[0] == 0.2 and c[1] != 1e6)]
    out["3 duration"] = report("3. duration decoder (min H/A s, short-run penalty, min D s)", cells, run_grid(pool, "dur", cells, ms), base, ms)

    # 4. count-matching dead cost: for every held-out match pick the dead cost whose predicted possession count is closest
    # to the mean TRUE count of the other 6 matches (a prior from other games, no label of the match itself)
    cells = [8, 12, 16, 20, 32, 48, 64, 96, 128]
    R = run_grid(pool, "dead", cells, ms)
    true_n = R[0, :, 3]
    S = R[:, :, 0]
    picked, scores = [], []
    for j in range(len(ms)):
        prior = np.delete(true_n, j).mean()
        c = int(np.argmin(np.abs(R[:, j, 2] - prior)))
        picked.append(cells[c])
        scores.append(S[c, j])
    d = np.array(scores) - base[:, 0]
    print("\n== 4. dead cost chosen so the possession count matches the other matches' mean")
    print("   picked dead costs per match %s ; prior counts %s" % (picked, np.round([np.delete(true_n, j).mean() for j in range(len(ms))], 0)))
    print("   spellF1 %.4f  paired %+.4f  better in %d/7  per match %s" % (np.mean(scores), d.mean(), (d > 0).sum(), np.round(d, 3)))
    out["4 count"] = (d.mean(), int((d > 0).sum()))
    print("\nsummary (honest paired gain, matches better): ")
    for k, (g, n) in out.items():
        print("   %-12s %+.4f  %d/7   %s" % (k, g, n, "PASSES (>= 0.01 and >= 5/7)" if g >= 0.01 and n >= 5 else "does not pass"))


if __name__ == "__main__":
    main()
