"""Label-free domain alignment on the LightGBM features: adversarial pruning and importance weights.

An adversarial classifier is trained to tell degraded-DFL frames from ATD frames. Features that carry
that distinction are the ones most likely to be league / tracker / team tells rather than possession
cues. Two uses, neither of which touches a competition label:

  prune    drop the most "tell-like" features round by round while the adversarial AUC falls
           -> data/pipeline/prune.json  (ordered drop list + AUC curve)
  weights  density-ratio weights w = p / (1 - p) for the DFL training frames (cross-fitted by match)
           -> data/pipeline/adv_weights.npz

    python pipeline/adversarial.py --prune
    python pipeline/adversarial.py --weights
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FEAT = os.path.join(ROOT, "data", "pipeline", "features")
STEP = 25
PRM = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=50,
           feature_fraction=0.8, verbose=-1, seed=0, num_threads=16)


def auc(s, l):
    o = np.argsort(s)
    r = np.empty(len(s))
    r[o] = np.arange(len(s))
    n1 = l.sum()
    return (r[l == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * (len(l) - n1))


def load(seed=2):
    names = json.load(open(os.path.join(FEAT, "feature_names.json")))
    a = np.load(os.path.join(FEAT, "atd.npz"))
    atd = {h: a["X%d" % h][::STEP] for h in (1, 2)}
    dfl = {}
    for p in sorted(glob.glob(os.path.join(FEAT, "DFL*_s%d.npz" % seed))):
        z = np.load(p)
        m = os.path.basename(p)[:14]
        for h in (1, 2):
            dfl[(m, h)] = z["X%d" % h][::STEP]
    return names, atd, dfl


def split_auc(names, cols, atd, dfl, rounds=150, want_gain=False):
    """Mean AUC over two folds (train on half 1 + matches 0-3 / test the rest, and the reverse)."""
    import lightgbm as lgb
    mids = sorted({k[0] for k in dfl})
    folds = ((mids[:4], 1), (mids[4:], 2))
    aucs, gain = [], np.zeros(len(cols))
    for tr_m, h_tr in folds:
        te_m = [m for m in mids if m not in tr_m]
        h_te = 3 - h_tr
        Xtr = np.vstack([atd[h_tr][:, cols]] + [dfl[(m, h)][:, cols] for m in tr_m for h in (1, 2)])
        ytr = np.r_[np.zeros(len(atd[h_tr])), np.ones(len(Xtr) - len(atd[h_tr]))]
        Xte = np.vstack([atd[h_te][:, cols]] + [dfl[(m, h)][:, cols] for m in te_m for h in (1, 2)])
        yte = np.r_[np.zeros(len(atd[h_te])), np.ones(len(Xte) - len(atd[h_te]))]
        w = np.where(ytr == 0, (ytr == 1).sum() / (ytr == 0).sum(), 1.0)
        mod = lgb.train(PRM, lgb.Dataset(Xtr, ytr, weight=w, feature_name=[names[c] for c in cols]), rounds)
        aucs.append(auc(mod.predict(Xte), yte))
        gain += mod.feature_importance("gain")
    return float(np.mean(aucs)), gain


def prune(per_round=6, floor=40):
    names, atd, dfl = load()
    cols = list(range(len(names)))
    dropped, curve = [], []
    while len(cols) > floor:
        a, gain = split_auc(names, cols, atd, dfl)
        curve.append((len(cols), a))
        print("%3d features  adversarial AUC %.3f   dropping: %s" % (
            len(cols), a, ", ".join(names[cols[i]] for i in np.argsort(-gain)[:3])), flush=True)
        drop = [cols[i] for i in np.argsort(-gain)[:per_round]]
        dropped += [names[c] for c in drop]
        cols = [c for c in cols if c not in drop]
    json.dump({"order": dropped, "curve": curve}, open(os.path.join(ROOT, "data", "pipeline", "prune.json"), "w"), indent=1)


def weights(clip=(0.2, 5.0)):
    """Cross-fitted density-ratio weights for every DFL (match, half, seed) frame, normalised to mean 1."""
    import lightgbm as lgb
    names, atd, dfl = load()
    cols = list(range(len(names)))
    mids = sorted({k[0] for k in dfl})
    out = {}
    a_all = np.vstack([atd[1], atd[2]])
    for m in mids:  # classifier never sees the match it scores
        tr = [k for k in dfl if k[0] != m]
        Xtr = np.vstack([a_all] + [dfl[k] for k in tr])
        ytr = np.r_[np.zeros(len(a_all)), np.ones(len(Xtr) - len(a_all))]
        w = np.where(ytr == 0, (ytr == 1).sum() / (ytr == 0).sum(), 1.0)
        mod = lgb.train(PRM, lgb.Dataset(Xtr, ytr, weight=w, feature_name=names), 150)
        for s in (0, 1, 2):
            z = np.load(os.path.join(FEAT, "%s_s%d.npz" % (m, s)))
            for h in (1, 2):
                p = np.clip(mod.predict(z["X%d" % h][::5]), 1e-3, 1 - 1e-3)
                r = np.clip(p / (1 - p), *clip)
                out["%s_s%d_h%d" % (m, s, h)] = (r / r.mean()).astype(np.float32)
        print("weights for", m, flush=True)
    np.savez(os.path.join(ROOT, "data", "pipeline", "adv_weights.npz"), **out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prune", action="store_true")
    ap.add_argument("--weights", action="store_true")
    ap.add_argument("--auc", action="store_true")
    args = ap.parse_args()
    if args.auc:
        names, atd, dfl = load()
        a, gain = split_auc(names, list(range(len(names))), atd, dfl)
        print("all %d features: adversarial AUC %.3f" % (len(names), a))
        for g, n in sorted(zip(gain, names), reverse=True)[:12]:
            print("   %-28s gain share %.1f%%" % (n, 100 * g / gain.sum()))
    if args.prune:
        prune()
    if args.weights:
        weights()
