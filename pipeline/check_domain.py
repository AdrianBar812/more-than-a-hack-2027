"""Adversarial validation: can a classifier tell degraded DFL from the ATD test match?

AUC near 0.5 = the training data looks like the test data on these features. Features are
orientation-free (no absolute x/y of centroids) so attack-direction conventions are not a tell.
Train on ATD half 1 + DFL matches 0-3, test on ATD half 2 + DFL matches 4-6.

    python pipeline/check_domain.py
"""

import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import atd  # noqa: E402
import degrade as dg  # noqa: E402
import player_features as pf  # noqa: E402

STEP = 25  # one sample per second


def orient_free(players, team, ball):
    f = pf.team_features(players, team, 4.0)
    for k in ("cx_h", "cx_a", "cy_h", "cy_a", "cx_gap"):
        f.pop(k)
    visible = ~np.isnan(ball[:, 0])
    bf = pf.ball_features(ball, players)
    f["ball_speed"] = bf["ball_speed"]
    f["ball_near"] = bf["ball_near"]
    f["ball_absx"] = np.abs(ball[:, 0] - 0.5)
    f["ball_absy"] = np.abs(ball[:, 1] - 0.5)
    f["vis_frac_10s"] = pf.visibility_fraction(visible)
    f["ball_visible"] = visible.astype(np.float32)
    return f


def auc(s, l):
    o = np.argsort(s)
    r = np.empty(len(s))
    r[o] = np.arange(len(s))
    n1 = l.sum()
    return (r[l == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * (len(l) - n1))


def run_lengths(m):
    cut = np.flatnonzero(np.diff(m.astype(int)) != 0) + 1
    seg = np.split(m, cut)
    return np.array([len(s) for s in seg if s[0]]) / 25.0


KEEP_RAW = ("ball_visible", "vis_frac_10s", "n_tracked")


def match_normalise(X, names):
    """Per-match robust z-score (median / IQR) of every continuous feature, from that match's own inputs."""
    out = X.copy()
    for j, n in enumerate(names):
        if n in KEEP_RAW:
            continue
        col = X[:, j]
        q25, med, q75 = np.nanpercentile(col, [25, 50, 75])
        out[:, j] = (col - med) / max(q75 - q25, 1e-6)
    return out


if __name__ == "__main__":
    import lightgbm as lgb
    NORM = "--normalise" in sys.argv
    ball, players, team = atd.load()
    f = orient_free(players, team, ball)
    names = sorted(f)
    X_atd = np.column_stack([f[n] for n in names])
    if NORM:
        X_atd = match_normalise(X_atd, names)
    g1 = np.arange(11185, 90652, STEP)
    g2 = np.arange(110808, 190198, STEP)

    dfl = {}
    files = sorted(glob.glob(os.path.join(ROOT, "data", "pipeline", "dfl_canon", "*_s0.npz")))
    for path in files:
        z = np.load(path)
        for h in (1, 2):
            ff = orient_free(z["h%d_players" % h], z["h%d_team" % h], z["h%d_ball" % h])
            dfl[(os.path.basename(path)[:14], h)] = np.column_stack([ff[n] for n in names])
        if NORM:
            both = match_normalise(np.vstack([dfl[(os.path.basename(path)[:14], h)] for h in (1, 2)]), names)
            n1 = len(dfl[(os.path.basename(path)[:14], 1)])
            dfl[(os.path.basename(path)[:14], 1)], dfl[(os.path.basename(path)[:14], 2)] = both[:n1], both[n1:]
        for h in (1, 2):
            k = (os.path.basename(path)[:14], h)
            dfl[k] = dfl[k][::STEP]
    mids = sorted({k[0] for k in dfl})
    tr_m, te_m = mids[:4], mids[4:]
    Xtr = np.vstack([X_atd[g1]] + [dfl[(m, h)] for m in tr_m for h in (1, 2)])
    ytr = np.r_[np.zeros(len(g1)), np.ones(len(Xtr) - len(g1))]
    Xte = np.vstack([X_atd[g2]] + [dfl[(m, h)] for m in te_m for h in (1, 2)])
    yte = np.r_[np.zeros(len(g2)), np.ones(len(Xte) - len(g2))]
    prm = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=50,
               feature_fraction=0.8, verbose=-1, seed=0)
    w = np.where(ytr == 0, (ytr == 1).sum() / (ytr == 0).sum(), 1.0)  # balance classes
    m = lgb.train(prm, lgb.Dataset(Xtr, ytr, weight=w, feature_name=names), 150)
    print("adversarial AUC (0.5 = indistinguishable): %.3f" % auc(m.predict(Xte), yte))
    for gain, n in sorted(zip(m.feature_importance("gain"), names), reverse=True)[:6]:
        print("   %-14s gain %.0f" % (n, gain))

    print("\nmarginals            ATD(graded)   DFL degraded (all halves)")
    allD = np.vstack(list(dfl.values()))
    for n in ("ball_visible", "ball_speed", "ball_near", "speed_all", "n_tracked", "ball_absx"):
        i = names.index(n)
        a = X_atd[np.r_[g1, g2]][:, i]
        d = allD[:, i]
        print("  %-14s  p10/50/90 %s   %s" % (n, np.round(np.nanpercentile(a, [10, 50, 90]), 2),
                                             np.round(np.nanpercentile(d, [10, 50, 90]), 2)))
    miss_atd = np.concatenate([np.isnan(ball[a:e, 0]) for a, e in dg.GRADED])
    print("  ball missing share  ATD %.2f   DFL degraded %.2f" % (
        miss_atd.mean(), np.mean([np.load(p)["h%d_ball_missing" % h].mean() for p in files for h in (1, 2)])))
    rl_a = np.concatenate([run_lengths(np.isnan(ball[a:e, 0])) for a, e in dg.GRADED])
    rl_d = np.concatenate([run_lengths(np.load(p)["h%d_ball_missing" % h]) for p in files for h in (1, 2)])
    print("  missing-run length  ATD median %.1fs mean %.1fs   DFL median %.1fs mean %.1fs" % (
        np.median(rl_a), rl_a.mean(), np.median(rl_d), rl_d.mean()))
