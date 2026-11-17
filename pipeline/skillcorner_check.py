"""One-match transfer check on SkillCorner open data (broadcast tracking, A-League): NO retraining.

Converts one match to our canonical arrays (metres -> 0-1, anonymous per-team slots, 10 -> 25 fps), runs the
DFL-trained LightGBM + Viterbi decoder, and compares with the SkillCorner `possession` field.

    python pipeline/skillcorner_check.py --match 1874553 --tracking <jsonl> --meta <match.json> --model data/pipeline/lgb_v2/model.txt

Caveats: SkillCorner's `possession.group` is empty both for a dead ball and whenever the ball is not detected
(96.5% empty when the ball is missing), so dead ball is only checkable on ball-visible frames; home/away is
checkable wherever the group is set. Only detected players / ball are used (extrapolated positions are dropped).
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "evaluation"))
import degrade as dg  # noqa: E402
import features as F  # noqa: E402
import train_lgb as T  # noqa: E402

FPS_IN, FPS_OUT = 10.0, 25.0


def upsample(a):
    """Linear 10 -> 25 fps along axis 0; NaN wherever either neighbouring sample is missing."""
    n = len(a)
    m = int((n - 1) * FPS_OUT / FPS_IN) + 1
    u = np.arange(m) * FPS_IN / FPS_OUT
    i0 = np.minimum(np.floor(u + 1e-9).astype(int), n - 1)
    i1 = np.minimum(i0 + 1, n - 1)
    w = (u - i0).reshape((-1,) + (1,) * (a.ndim - 1))
    out = (1 - w) * a[i0] + w * a[i1]
    return np.where(w < 1e-9, a[i0], out)


def load(tracking, meta):
    m = json.load(open(meta))
    home_id = m["home_team"]["id"]
    ids = {}
    for p in m["players"]:
        team = 0 if p["team_id"] == home_id else 1
        ids[p["id"]] = team
        ids[p["trackable_object"]] = team
    rows = [json.loads(l) for l in open(tracking)]
    n = len(rows)
    period = np.array([r["period"] or 0 for r in rows])
    ball = np.full((n, 2), np.nan)
    lab = np.full(n, 2, np.int8)
    pos = {}
    for t, r in enumerate(rows):
        b = r["ball_data"]
        if b["x"] is not None and b["is_detected"]:
            ball[t] = (b["x"], b["y"])
        g = r["possession"]["group"]
        lab[t] = {"home team": 0, "away team": 1}.get(g, 2)
        for p in r["player_data"]:
            if p["is_detected"] and p["x"] is not None and p["player_id"] in ids:
                pos.setdefault(p["player_id"], np.full((n, 2), np.nan))[t] = (p["x"], p["y"])
    L, W = float(m["pitch_length"]), float(m["pitch_width"])
    halves = []
    for h, side in ((1, m["home_team_side"][0]), (2, m["home_team_side"][1])):
        idx = np.flatnonzero(period == h)
        pid = sorted(pos)
        P = np.stack([upsample(pos[i][idx]) for i in pid])          # [P, T, 2] metres
        team = np.array([ids[i] for i in pid])
        arr = np.concatenate([P, np.full(P.shape[:2] + (1,), np.nan)], axis=2)
        slots, st = dg.to_slots(arr, team)
        slots[:, :, 0] = slots[:, :, 0] / L + 0.5
        slots[:, :, 1] = slots[:, :, 1] / W + 0.5
        b = upsample(ball[idx])
        b = np.column_stack([b[:, 0] / L + 0.5, b[:, 1] / W + 0.5])
        u = np.round(np.arange(len(b)) * FPS_IN / FPS_OUT).astype(int).clip(0, len(idx) - 1)
        halves.append(dict(ball=b, players=slots, team=st, lab=lab[idx][u], expect_right=(side == "left_to_right")))
    return halves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracking", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--model", default=os.path.join(os.path.dirname(HERE), "data", "pipeline", "lgb_v2", "model.txt"))
    ap.add_argument("--costs", type=float, nargs=2, default=(16, 20))
    args = ap.parse_args()
    import lightgbm as lgb
    model = lgb.Booster(model_file=args.model)
    rows = []
    for k, hf in enumerate(load(args.tracking, args.meta), 1):
        names, X, right = F.build_features(hf["ball"], hf["players"], hf["team"])
        P = model.predict(X)
        dec = T.viterbi(P, *args.costs)
        vis = ~np.isnan(hf["ball"][:, 0])
        print("half %d: %d frames at 25 fps, ball visible %.1f%%, players tracked/frame %.1f, direction inferred %s (declared %s)" % (
            k, len(vis), 100 * vis.mean(), np.mean(np.sum(~np.isnan(hf["players"][:, :, 0]), axis=1)),
            "right" if right else "left", "right" if hf["expect_right"] else "left"))
        rows.append((hf["lab"], P, dec, vis))
    lab = np.concatenate([r[0] for r in rows])
    P = np.vstack([r[1] for r in rows])
    dec = np.concatenate([r[2] for r in rows])
    vis = np.concatenate([r[3] for r in rows])
    labelled = lab < 2
    ha = (P[:, 1] > P[:, 0]).astype(int)
    print("\nlabelled home/away frames: %d (%.1f%% of frames); of those ball visible %.1f%%" % (
        labelled.sum(), 100 * labelled.mean(), 100 * vis[labelled].mean()))
    for name, sel in (("all labelled", labelled), ("ball visible", labelled & vis), ("ball missing", labelled & ~vis)):
        if sel.sum():
            print("  H/A accuracy (P(H) vs P(A)) %-13s %.3f  (n=%d)" % (name, np.mean(ha[sel] == lab[sel]), sel.sum()))
    live = labelled & (dec < 2)
    print("  H/A accuracy after Viterbi, frames not decoded as D: %.3f (n=%d); labelled frames decoded as D: %.3f" % (
        np.mean(dec[live] == lab[live]), live.sum(), np.mean(dec[labelled] == 2)))
    print("\ndead ball vs empty possession (only meaningful where the ball is visible):")
    for name, sel in (("ball visible", vis), ("ball missing", ~vis)):
        none = lab[sel] == 2
        d = dec[sel] == 2
        print("  %-13s empty %.3f | decoded D %.3f | both %.3f | decoded D given empty %.3f | empty given decoded D %.3f" % (
            name, none.mean(), d.mean(), np.mean(none & d), np.mean(d[none]) if none.any() else float("nan"), np.mean(none[d]) if d.any() else float("nan")))
    print("\nconfusion matrices (rows = SkillCorner possession: home / away / empty, columns = decoded H / A / D):")
    for name, sel in (("all frames", np.ones(len(lab), bool)), ("ball visible", vis), ("ball missing", ~vis)):
        cm = np.array([[np.sum(sel & (lab == a) & (dec == b)) for b in range(3)] for a in range(3)])
        print("  %s (n=%d)" % (name, sel.sum()))
        for a, row in zip(("home ", "away ", "empty"), cm):
            print("    %s %8d %8d %8d" % (a, *row))
    print("overall decoded share H/A/D: %s ; labelled share H/A/empty: %s" % (
        np.round(np.bincount(dec, minlength=3) / len(dec), 3), np.round(np.bincount(lab, minlength=3) / len(lab), 3)))


if __name__ == "__main__":
    main()
