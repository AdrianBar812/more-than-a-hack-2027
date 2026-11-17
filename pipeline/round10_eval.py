"""Round 10: feature groups (short windows, restart geometry, nearest players) x decoders, on DFL out-of-fold and the ATD public window.

    python pipeline/round10_eval.py --runs lgb_v2 lgb_near lgb_r10_short lgb_r10_restart lgb_r10_sr lgb_r10_all lgb_r10_all_n2

For every run (a directory under data/pipeline with oof.npz and atd_probs.npz) it scores three decoders, all after Viterbi 16/20:
  m3     merge H/A spells shorter than 3 s (the adopted decoder)
  shift  m3 + move every H<->A boundary by d_ha frames and every boundary touching D by d_d frames (grid; best on all 7 / chosen on the other 6)
  combo  duration graph + visibility-dependent switch cost + debounce (decoder_experiments.vit_combo, cell fixed from round 8b)
References are always the N = 3 s DFL labels of lgb_v2 (so a run trained with another N is not scored against its own labels).
The public window is read only for information; nothing is chosen from it.
"""

import argparse
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import atd  # noqa: E402
import decoder_experiments as D  # noqa: E402
import labels as lab  # noqa: E402
import possession_metrics as pm  # noqa: E402
import train_lgb as T  # noqa: E402

PD = os.path.join(ROOT, "data", "pipeline")
P_ALL = {}      # tag -> match -> P
GT = {}         # match -> (G, blocks, miss)
SHIFTS = [(a, b) for a in (-25, -12, 0, 12, 25) for b in (-25, 0, 25)]
COMBO = (3, 32, 0.2, 16, 32, 3)


def shift_boundaries(path, d_ha, d_d):
    if d_ha == 0 and d_d == 0:
        return path
    cut = np.flatnonzero(np.diff(path) != 0) + 1
    if len(cut) == 0:
        return path
    labels = np.concatenate(([path[0]], path[cut])).astype(np.int8)
    new = []
    for i, pos in enumerate(cut):
        d = d_ha if (labels[i] < 2 and labels[i + 1] < 2) else d_d
        new.append(int(pos) + d)
    n = len(path)
    out, prev = [], 0
    for pos in new:
        pos = max(pos, prev + 1)
        out.append(pos)
        prev = pos
    nxt = n
    for i in range(len(out) - 1, -1, -1):
        out[i] = min(out[i], nxt - 1)
        nxt = out[i]
    edges = [0] + out + [n]
    res = np.empty(n, np.int8)
    for i, lb in enumerate(labels):
        res[edges[i]:edges[i + 1]] = lb
    return res


def decode(kind, par, P, miss):
    if kind == "m3":
        return T.decode(P, 16, 20)
    if kind == "shift":
        return shift_boundaries(T.decode(P, 16, 20), *par)
    if kind == "combo":
        mh, lam, md, sv, sm, deb = COMBO
        return D.vit_combo(P, miss, sv, sm, 20, mh, lam, md, deb)
    raise ValueError(kind)


def score(a):
    tag, kind, par, m = a
    G, blocks, miss = GT[m]
    P = P_ALL[tag][m]
    pred = np.full(len(G), pm.X, np.int8)
    for s, e in blocks:
        pred[s:e + 1] = decode(kind, par, P[s:e + 1], miss[s:e + 1])
    r = pm.evaluate(G, pred)
    return r["spell_f1_mean"], r["macro_f1"], r["per_class"]["D"]["recall"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    D.load(os.path.join(PD, "lgb_v2"))
    ms = T.matches()
    for m in ms:
        GT[m] = (D.DATA[m]["G"], D.DATA[m]["blocks"], D.DATA[m]["miss"])
    for tag in args.runs:
        o = np.load(os.path.join(PD, tag, "oof.npz"))
        P_ALL[tag] = {m: o[m + "_P"] for m in ms}
    pool = mp.get_context("fork").Pool(args.workers)
    base = np.array(pool.map(score, [("lgb_v2", "m3", None, m) for m in ms]))[:, 0]
    print("baseline = lgb_v2 + merge 3 s: out-of-fold spellF1 %.4f\n" % base.mean())

    gt_pub = pm.label_codes(pm.read_label_csv(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")))
    ball, _, _ = atd.load()

    def pub(tag, kind, par):
        a = np.load(os.path.join(PD, tag, "atd_probs.npz"))
        c = np.full(199319, pm.X, np.int8)
        for h in (1, 2):
            p = a["P%d" % h]
            s = int(a["start%d" % h])
            c[s:s + len(p)] = decode(kind, par, p, np.isnan(ball[s:s + len(p), 0]))
        r = pm.evaluate(gt_pub, c)
        return r["spell_f1_mean"], r["macro_f1"], r["per_class"]["D"]["recall"], r["n_pred_spells"]

    print("%-16s %-18s | %7s %7s %10s %6s | %7s %7s %6s %5s" % ("run", "decoder", "OOF", "paired", "better", "D rec", "ATD", "macro", "D rec", "poss"))
    for tag in args.runs:
        cells = [("m3", None), ("combo", None)] + [("shift", s) for s in SHIFTS]
        R = np.array(pool.map(score, [(tag, k, p, m) for k, p in cells for m in ms])).reshape(len(cells), len(ms), 3)
        S = R[:, :, 0]
        rows = []
        # m3
        rows.append(("merge 3 s", S[0], R[0, :, 2].mean(), pub(tag, "m3", None)))
        # shift: best on all 7 and honest
        si = list(range(2, len(cells)))
        best = max(si, key=lambda i: S[i].mean())
        hon = np.array([S[max(si, key=lambda i: np.delete(S[i], j).mean()), j] for j in range(len(ms))])
        rows.append(("shift %s" % (cells[best][1],), S[best], R[best, :, 2].mean(), pub(tag, "shift", cells[best][1])))
        rows.append(("shift (honest)", hon, float("nan"), (float("nan"),) * 4))
        rows.append(("combo 1+2+3", S[1], R[1, :, 2].mean(), pub(tag, "combo", None)))
        for name, sc, drec, pb in rows:
            d = sc - base
            print("%-16s %-18s | %7.4f %+7.4f %7d/7 %6.3f | %7.4f %7.4f %6.3f %5s" % (
                tag, name, sc.mean(), d.mean(), (d > 0).sum(), drec, pb[0], pb[1], pb[2], "-" if np.isnan(pb[0]) else "%d" % pb[3]), flush=True)


if __name__ == "__main__":
    main()
