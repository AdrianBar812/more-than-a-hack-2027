"""Sweep the debounce N (minimum possession length in the DFL training labels).

N changes the labels, so scoring a model against labels built with the same N is circular (a larger
N is simply easier). For every training N this script runs leave-one-match-out, then scores the
out-of-fold probabilities against EVERY reference N (decoder costs tuned per cell on a small grid).
A good N is one that holds up across references, not one that wins its own.

    python pipeline/sweep_n.py --ns 2 3 4 5 --out data/pipeline/sweep_n.json

The public ground truth is not read here.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "evaluation"))
import labels as lab  # noqa: E402
import possession_metrics as pm  # noqa: E402
import train_lgb as T  # noqa: E402

GRID = [(c_sw, c_d) for c_sw in (16, 48) for c_d in (8, 20, 40)]


def lomo_probs(ms, n_train, names):
    out = {}
    for m in ms:
        model = T.fit([o for o in ms if o != m], n_train, names)
        z = np.load(os.path.join(T.FEAT, "%s_s2.npz" % m))
        P, _ = T.combine([model.predict(z["X%d" % h]) for h in (1, 2)])
        out[m] = P
    return out


def reference(m, n_ref, probs):
    z = np.load(os.path.join(T.FEAT, "%s_s2.npz" % m))
    G, blocks = T.combine([lab.debounce(z["y%d" % h], n_ref).astype(np.int8) for h in (1, 2)])
    return np.where(np.isnan(probs[:, 0]), pm.X, G).astype(np.int8), blocks


def best_cell(ms, probs, n_ref, workers):
    T._OOF.clear()
    for m in ms:
        G, blocks = reference(m, n_ref, probs[m])
        T._OOF[m] = (probs[m], G, blocks)
    pool = mp.get_context("fork").Pool(workers)
    best = None
    for c_sw, c_d in GRID:
        r = np.array(pool.map(T._task, [(m, c_sw, c_d, "viterbi") for m in ms]), dtype=float)
        cell = (float(r[:, 0].mean()), float(r[:, 1].mean()), float(r[:, 2].sum() / r[:, 3].sum()), c_sw, c_d)
        if best is None or cell[0] > best[0]:
            best = cell
    pool.close()
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=float, nargs="+", default=[2, 3, 4, 5])
    ap.add_argument("--out", default=os.path.join(T.ROOT, "data", "pipeline", "sweep_n.json"))
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    import json as _json
    names = _json.load(open(os.path.join(T.FEAT, "feature_names.json")))
    ms = T.matches()
    t0 = time.time()
    table = {}
    for n_train in args.ns:
        probs = lomo_probs(ms, n_train, names)
        table[n_train] = {}
        for n_ref in args.ns:
            table[n_train][n_ref] = best_cell(ms, probs, n_ref, args.workers)
        print("train N=%g done (%.0fs)" % (n_train, time.time() - t0), flush=True)

    print("\nspellF1 (rows = N used for training labels, columns = reference N used for scoring)")
    print("train \\ ref  " + "".join("%8g" % r for r in args.ns) + "     mean     min")
    for n_train in args.ns:
        row = [table[n_train][r][0] for r in args.ns]
        print("N=%-9g " % n_train + "".join("%8.4f" % v for v in row) + "  %7.4f %7.4f" % (np.mean(row), np.min(row)))
    print("\nmacroF1")
    for n_train in args.ns:
        row = [table[n_train][r][1] for r in args.ns]
        print("N=%-9g " % n_train + "".join("%8.4f" % v for v in row))
    print("\npredicted/true possessions")
    for n_train in args.ns:
        row = [table[n_train][r][2] for r in args.ns]
        print("N=%-9g " % n_train + "".join("%8.2f" % v for v in row))
    json.dump({str(k): {str(r): v for r, v in d.items()} for k, d in table.items()}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
