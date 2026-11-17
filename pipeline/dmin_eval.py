"""Round 11: minimum dead-ball span (2 s / 3 s) and a cheaper dead-ball entry when the 5 players nearest the ball are slow.

    python pipeline/dmin_eval.py

Decoder (5 frames per step, like decoder_experiments.vit_dur): Viterbi over -log p with switch cost 16, dead cost 20
(or `cheap` while the smoothed speed of the 5 players nearest to the ball is below `thr` x the half's median player
speed), a hard minimum length of `min_d` seconds for D, then the adopted merge of H/A spells shorter than 3 s.
References are the N = 3 s DFL labels, scored like the OOF runs. The ATD public window is read for information only.
"""
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
import features as F  # noqa: E402
import labels as lab  # noqa: E402
import player_features as pf  # noqa: E402
import possession_metrics as pm  # noqa: E402
import train_lgb as T  # noqa: E402

PD = os.path.join(ROOT, "data", "pipeline")
STEP = 5
SIG = {}        # match -> combined slowness signal (rel. speed of the 5 players nearest the ball)
GT = {}


def near5_speed(ball, players):
    """Mean speed (m/s) of the 5 players nearest the (gap-filled) ball, smoothed 2 s, divided by the half's median speed."""
    T_ = len(ball)
    bx, by = ball[:, 0].astype(float), ball[:, 1].astype(float)
    ok = ~np.isnan(bx)
    if ok.sum() == 0:
        return np.full(T_, np.nan, np.float32)
    idx = np.where(ok, np.arange(T_), -1)
    last = np.maximum.accumulate(idx)
    nxt = np.where(ok, np.arange(T_), T_)
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    pick = np.where((last >= 0) & ((nxt >= T_) | (np.arange(T_) - last <= nxt - np.arange(T_))), last, nxt)
    pick = np.clip(pick, 0, T_ - 1)
    bx, by = bx[pick], by[pick]
    sp = pf.player_speed(players[:, :, :2])
    d = np.hypot((players[:, :, 0] - bx[:, None]) * pf.LENGTH, (players[:, :, 1] - by[:, None]) * pf.WIDTH)
    d = np.where(np.isnan(d) | np.isnan(sp), np.inf, d)
    k = np.argpartition(d, 5, axis=1)[:, :5]
    s5 = np.take_along_axis(sp, k, axis=1)
    dd = np.take_along_axis(d, k, axis=1)
    s5 = np.where(np.isinf(dd), np.nan, s5)
    with np.errstate(invalid="ignore"):
        m = np.nanmean(s5, axis=1)
    m = F.wmean(m, 51)
    return (m / np.nanmedian(sp)).astype(np.float32)


def vit(P, slow, sw, dead, cheap, min_d_s):
    """Graph Viterbi at STEP frames per step; D needs min_d_s seconds; D entry costs `cheap` where slow else `dead`."""
    n = len(P)
    nc = -(-n // STEP)
    nll = np.zeros((nc * STEP, 3))
    nll[:n] = D.nll_of(P)
    nll = nll.reshape(nc, STEP, 3).sum(1)
    sl = np.zeros(nc * STEP)
    sl[:n] = slow
    sl = (sl.reshape(nc, STEP).mean(1) > 0.5).astype(int)
    Ld = max(1, int(round(min_d_s * 25 / STEP)))
    nodes = [(0, 0), (1, 0)] + [(2, k) for k in range(Ld)]
    idx = {nd: i for i, nd in enumerate(nodes)}
    N = len(nodes)
    cls = np.array([a for a, _ in nodes])
    Cs = []
    for dc in (dead, cheap):
        base = np.array([[0, sw, dc], [sw, 0, dc], [dc, dc, 0]], float)
        C = np.full((N, N), np.inf)
        for (s, k), i in idx.items():
            if s == 2 and k < Ld - 1:
                C[i, idx[(2, k + 1)]] = 0.0          # must stay in D until the minimum is reached
            else:
                C[i, i] = 0.0
                for s2 in range(3):
                    if s2 != s:
                        C[i, idx[(s2, 0)]] = base[s, s2]
        Cs.append(C)
    score = nll[0][cls].copy()
    back = np.zeros((nc, N), np.int16)
    ar = np.arange(N)
    for t in range(1, nc):
        cand = score[:, None] + Cs[sl[t]]
        b = cand.argmin(0)
        back[t] = b
        score = cand[b, ar] + nll[t][cls]
    path = np.empty(nc, np.int16)
    path[-1] = score.argmin()
    for t in range(nc - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return np.repeat(cls[path], STEP)[:n].astype(np.int8)


def decode(P, sig, par):
    thr, cheap, min_d = par
    slow = (sig < thr) if thr else np.zeros(len(P), bool)
    path = vit(P, np.nan_to_num(slow.astype(float)), 16, 20, cheap if thr else 20, min_d)
    return lab.debounce(path, 3.0).astype(np.int8)


def score(a):
    par, m = a
    G, blocks, _ = GT[m]
    P = D.DATA[m]["P"]
    pred = np.full(len(G), pm.X, np.int8)
    for s, e in blocks:
        pred[s:e + 1] = decode(P[s:e + 1], SIG[m][s:e + 1], par)
    r = pm.evaluate(G, pred)
    return r["spell_f1_mean"], r["macro_f1"], r["per_class"]["D"]["recall"], r["per_class"]["D"]["precision"], r["n_pred_spells"]


def main():
    D.load(os.path.join(PD, "lgb_v2"))
    ms = T.matches()
    for m in ms:
        GT[m] = (D.DATA[m]["G"], D.DATA[m]["blocks"], D.DATA[m]["miss"])
        z = np.load(os.path.join(PD, "dfl_canon", "%s_s2.npz" % m))
        S, _ = T.combine([near5_speed(z["h%d_ball" % h], z["h%d_players" % h]) for h in (1, 2)])
        SIG[m] = S
    # distribution check (DFL, scored frames): dead vs live
    allsig = np.concatenate([SIG[m][~np.isnan(D.DATA[m]["P"][:, 0])] for m in ms])
    allg = np.concatenate([D.DATA[m]["G"][~np.isnan(D.DATA[m]["P"][:, 0])] for m in ms])
    for nm, k in (("live", allg < 2), ("dead", allg == 2)):
        q = np.nanpercentile(allsig[k], [10, 25, 50, 75, 90])
        print("near-5 relative speed on DFL %s: p10 %.2f p25 %.2f p50 %.2f p75 %.2f p90 %.2f" % ((nm,) + tuple(q)))
    pool = mp.get_context("fork").Pool(16)
    cells = [(0, 20, 0.2)]
    cells += [(0, 20, 2.0), (0, 20, 3.0)]
    slow_cells = [(t, c) for t in (0.6, 0.8, 1.0, 1.2) for c in (6, 12)]
    for md in (0.2, 2.0, 3.0):
        cells += [(t, c, md) for t, c in slow_cells]
    cells = list(dict.fromkeys(cells))
    R = np.array(pool.map(score, [(c, m) for c in cells for m in ms])).reshape(len(cells), len(ms), 5)
    S = R[:, :, 0]
    base = S[0]
    # ATD public window
    gt_pub = pm.label_codes(pm.read_label_csv(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")))
    ball, players, _ = atd.load()
    a = np.load(os.path.join(PD, "lgb_v2", "atd_probs.npz"))
    atd_sig = {}
    for h in (1, 2):
        s0 = int(a["start%d" % h])
        n = len(a["P%d" % h])
        atd_sig[h] = near5_speed(ball[s0:s0 + n], players[s0:s0 + n])

    def pub(par):
        c = np.full(199319, pm.X, np.int8)
        for h in (1, 2):
            p = a["P%d" % h]
            s0 = int(a["start%d" % h])
            c[s0:s0 + len(p)] = decode(p, atd_sig[h], par)
        r = pm.evaluate(gt_pub, c)
        return r["spell_f1_mean"], r["macro_f1"], r["per_class"]["D"]["recall"], r["per_class"]["D"]["precision"], r["n_pred_spells"]

    print("ATD relative near-5 speed: median %.2f, p25 %.2f" % (np.nanmedian(np.concatenate(list(atd_sig.values()))), np.nanpercentile(np.concatenate(list(atd_sig.values())), 25)))
    print("\ncontrol = same 5-frame Viterbi 16/20 + merge 3 s, no minimum, no cheap cost: OOF %.4f" % base.mean())
    print("%-34s | %7s %7s %6s | D rec D prec | %7s %7s %6s %6s" % ("variant (thr, cheap cost, min D s)", "OOF", "paired", "better", "ATD", "macro", "D rec", "D prec"))
    for i, c in enumerate(cells):
        dlt = S[i] - base
        p = pub(c)
        thr, ch, md = c
        name = ("slow<%.1f cost %d" % (thr, ch) if thr else "no cheap cost") + (", min D %.0f s" % md if md >= 1 else "")
        print("%-34s | %7.4f %+7.4f %4d/7 | %5.3f %6.3f | %7.4f %7.4f %6.3f %6.3f" % (
            name, S[i].mean(), dlt.mean(), (dlt > 0).sum(), R[i, :, 2].mean(), R[i, :, 3].mean(), p[0], p[1], p[2], p[3]), flush=True)
    # honest selection of the slow cell for each min D
    print("\nhonest (cell chosen on the other 6 matches):")
    for md in (0.2, 2.0, 3.0):
        ci = [i for i, c in enumerate(cells) if c[2] == md and c[0]]
        hon = np.array([S[max(ci, key=lambda i: np.delete(S[i], j).mean()), j] for j in range(len(ms))])
        ref = S[cells.index((0, 20, md))] if (0, 20, md) in cells else base
        print("min D %s: slow cell honest %.4f (paired vs control %+.4f, %d/7); vs same min D without slow %+.4f" % (
            "none" if md < 1 else "%.0f s" % md, hon.mean(), (hon - base).mean(), ((hon - base) > 0).sum(), (hon - ref).mean()))
    print("per-match control:", np.round(base, 3))


if __name__ == "__main__":
    main()
