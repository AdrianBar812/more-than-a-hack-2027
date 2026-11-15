"""LightGBM feature builder: one function for DFL (degraded) and the ATD, so train == test.

    names, X = build_features(ball, players, team)

Inputs are the canonical arrays (coordinates normalised 0-1, NaN = not tracked):
    ball     [T, >=2]   x, y          (NaN rows where the ball is not observed)
    players  [T, S, >=2] x, y          anonymous slots
    team     [S]        0 = home, 1 = away

Everything is offline (centred windows, "next seen" values), like the Viterbi decoding that follows.

Attack direction: the ATD carries no goalkeeper or direction information, so it is inferred per
half from the data itself (see `home_attacks_right`) and x is mirrored so that HOME ALWAYS ATTACKS
TOWARDS +x. Features are therefore comparable across halves and matches.

Feature groups (prefix):
    p_*   player-only: team centroids / spread / lines / speed / stillness / zone counts at 1, 4, 12 s
    b_*   ball, only where it is observed (NaN elsewhere): distance to each team, speed, region
    bw_*  ball windows: nan-aware means of the b_* signals plus "fraction of visible frames in which
          the home team was nearest"
    bl_*, bn_*  last-seen / next-seen ball values and how many seconds away they are (they bridge the
          10-17 s ball gaps of the ATD)
    v_*   how much of the surrounding time the ball is visible (the ATD loses it in stoppages)
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import player_features as pf  # noqa: E402

FPS = 25
L, W = pf.LENGTH, pf.WIDTH
WINDOWS = (25, 101, 301)          # 1 s, 4 s, 12 s (frames, odd)
VIS_WINDOWS = (101, 301, 751)     # 4 s, 12 s, 30 s
STILL_SPEED = 0.6                 # m/s, a player counts as standing still below this
AGE_CAP = 120.0                   # seconds
LONG_WINDOWS = (751, 1501)        # 30 s, 60 s: a dead-ball run lasts 10-20 s, the play around it longer
LONG_KEYS = ("still_frac", "rstill_frac", "speed_all", "rspeed_all", "sx_h", "sx_a", "touch_frac")


def wmean(x, win):
    """NaN-aware centred moving mean along axis 0 (win frames); NaN where the window has no data."""
    x = np.asarray(x, dtype=np.float64)
    ok = ~np.isnan(x)
    v = np.where(ok, x, 0.0)
    pad = np.zeros((1,) + x.shape[1:])
    cs = np.cumsum(np.concatenate([pad, v]), axis=0)
    cc = np.cumsum(np.concatenate([pad, ok.astype(np.float64)]), axis=0)
    T, h = len(x), win // 2
    lo = np.clip(np.arange(T) - h, 0, T)
    hi = np.clip(np.arange(T) + h + 1, 0, T)
    s, c = cs[hi] - cs[lo], cc[hi] - cc[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / c, np.nan).astype(np.float32)


def home_attacks_right(players, team):
    """True if the home team attacks towards +x in this half.

    A team's goalkeeper is the player that stays closest to its own goal line, so over a half the
    extreme player on the goalkeeper's side sits much nearer the border than the extreme on the
    other side. Compare the median left/right extremes of each team; both teams vote.
    """
    x = players[:, :, 0]
    votes = 0.0
    for tm, sign in ((0, 1.0), (1, -1.0)):
        xs = x[:, team == tm]
        enough = np.sum(~np.isnan(xs), axis=1) >= 6
        if not enough.any():
            continue
        xs = xs[enough]
        left = np.nanmedian(np.nanmin(xs, axis=1))
        right = 1.0 - np.nanmedian(np.nanmax(xs, axis=1))
        # own goal on the left (left < right) -> attacks right
        votes += sign * (right - left)
    return votes > 0


def _line(x, n_from, n_to, largest):
    """Mean of the n_from..n_to-th most extreme values per row (NaN if not enough players)."""
    fill = -np.inf if largest else np.inf
    s = np.sort(np.where(np.isnan(x), fill, x), axis=1)
    if largest:
        s = s[:, ::-1]
    seg = s[:, n_from:n_to]
    seg = np.where(np.isinf(seg), np.nan, seg)
    with np.errstate(invalid="ignore"):
        return np.where(np.isnan(seg).any(axis=1), np.nan, seg.mean(axis=1))


def _nearest(d, mask):
    sub = np.where(np.isnan(d[:, mask]), np.inf, d[:, mask])
    if sub.shape[1] == 0:
        return np.full(len(d), np.nan)
    m = sub.min(axis=1)
    return np.where(np.isinf(m), np.nan, m)


def _count_within(d, mask, r):
    sub = d[:, mask]
    with np.errstate(invalid="ignore"):
        return np.sum(sub < r, axis=1).astype(np.float64)


def _last_next(vis):
    """Index of the last / next frame with the ball (clipped), and the distance in seconds."""
    T = len(vis)
    idx = np.arange(T)
    last = np.maximum.accumulate(np.where(vis, idx, -1))
    nxt = np.minimum.accumulate(np.where(vis, idx, T)[::-1])[::-1]
    age_last = np.where(last >= 0, (idx - last) / FPS, np.nan)
    age_next = np.where(nxt < T, (nxt - idx) / FPS, np.nan)
    return last, nxt, np.minimum(age_last, AGE_CAP), np.minimum(age_next, AGE_CAP)


EXTRA = False   # round-3 features (CONTEXT.md): no gain on the public window, off by default
_EXTRA_MARKS = ("rspeed", "rstill", "touch_", "gk_speed")


def _is_extra(k):
    return k.startswith(("sl_", "sn_")) or any(m in k for m in _EXTRA_MARKS) or (k.startswith("p_") and k.endswith(("_w751", "_w1501")))


def build_features(ball, players, team, attack_right=None):
    T = len(ball)
    px = players[:, :, 0].astype(np.float64)
    py = players[:, :, 1].astype(np.float64)
    bx = ball[:, 0].astype(np.float64)
    by = ball[:, 1].astype(np.float64)
    if attack_right is None:
        attack_right = home_attacks_right(players, team)
    if not attack_right:  # mirror so home attacks +x
        px, bx = 1.0 - px, 1.0 - bx
    home, away = team == 0, team == 1
    f = {}

    # ---------------- player-only ----------------
    speed = pf.player_speed(np.stack([players[:, :, 0], players[:, :, 1]], axis=2))
    base = {}
    with np.errstate(invalid="ignore"):
        for name, m in (("h", home), ("a", away)):
            base["n_" + name] = np.sum(~np.isnan(px[:, m]), axis=1).astype(np.float64)
            base["cx_" + name] = np.nanmean(px[:, m], axis=1)
            base["cy_" + name] = np.nanmean(py[:, m], axis=1)
            base["sx_" + name] = np.nanstd(px[:, m], axis=1)
            base["sy_" + name] = np.nanstd(py[:, m], axis=1)
            base["speed_" + name] = np.nanmean(speed[:, m], axis=1)
        base["cx_all"] = np.nanmean(px, axis=1)
        base["cy_all"] = np.nanmean(py, axis=1)
        base["speed_all"] = np.nanmean(speed, axis=1)
        valid = ~np.isnan(speed)
        base["still_frac"] = np.where(valid.sum(1) > 0,
                                      np.sum((speed < STILL_SPEED) & valid, axis=1) / np.maximum(valid.sum(1), 1),
                                      np.nan)
    # speed relative to the half's own median (cancels league / tracker differences in pace; the DFL
    # players are ~20% faster than the ATD's, an adversarial tell that needs no labels to remove)
    med = np.nanmedian(speed)
    rel = speed / med
    still = (speed < STILL_SPEED) & ~np.isnan(speed)
    with np.errstate(invalid="ignore"):
        base["rspeed_all"] = np.nanmean(rel, axis=1)
        base["rspeed_h"] = np.nanmean(rel[:, home], axis=1)
        base["rspeed_a"] = np.nanmean(rel[:, away], axis=1)
        valid_r = ~np.isnan(rel)
        base["rstill_frac"] = np.where(valid_r.sum(1) > 0,
                                       np.sum((rel < 0.35) & valid_r, axis=1) / np.maximum(valid_r.sum(1), 1), np.nan)
        # set-piece geometry: players standing on the touchline, keepers (extreme x per team) standing still
        tracked = ~np.isnan(px)
        touch = (np.abs(py - 0.5) > 0.44) & tracked
        base["touch_frac"] = np.where(tracked.sum(1) > 0, touch.sum(1) / np.maximum(tracked.sum(1), 1), np.nan)
        base["touch_still"] = np.sum(touch & still, axis=1).astype(np.float64)
        for name, mask, deep_is_min in (("h", home, True), ("a", away, False)):
            xs = np.where(np.isnan(px[:, mask]), np.inf if deep_is_min else -np.inf, px[:, mask])
            gk = np.argmin(xs, axis=1) if deep_is_min else np.argmax(xs, axis=1)
            gsp = speed[:, mask][np.arange(T), gk]
            base["gk_speed_" + name] = np.where(np.isinf(xs[np.arange(T), gk]), np.nan, gsp)
    # lines: home attacks +x, so its front is the largest x; away's front is the smallest x
    base["front_h"] = _line(px[:, home], 0, 3, largest=True)
    base["back_h"] = _line(px[:, home], 1, 4, largest=False)   # 2nd-4th deepest: skips the keeper
    base["front_a"] = _line(px[:, away], 0, 3, largest=False)
    base["back_a"] = _line(px[:, away], 1, 4, largest=True)
    ph, pa = px[:, home], px[:, away]
    base["h_att3"] = np.sum(ph > 2 / 3, axis=1).astype(np.float64)
    base["h_def3"] = np.sum(ph < 1 / 3, axis=1).astype(np.float64)
    base["a_att3"] = np.sum(pa < 1 / 3, axis=1).astype(np.float64)
    base["a_def3"] = np.sum(pa > 2 / 3, axis=1).astype(np.float64)
    base["gap_x"] = base["cx_h"] - base["cx_a"]
    base["len_h"] = base["front_h"] - base["back_h"]
    base["len_a"] = base["back_a"] - base["front_a"]
    for k, v in base.items():
        for w in WINDOWS:
            f["p_%s_w%d" % (k, w)] = wmean(v, w)
    for k in LONG_KEYS:
        for w in LONG_WINDOWS:
            f["p_%s_w%d" % (k, w)] = wmean(base[k], w)
    cx = wmean(base["cx_all"], 25)
    for k in ("cx_all", "cx_h", "cx_a"):
        s = wmean(base[k], 25)
        f["p_%s_vel2s" % k] = np.concatenate([s[50:], np.full(50, np.nan)]) - np.concatenate([np.full(50, np.nan), s[:-50]])
    del cx

    # ---------------- ball (observed frames only) ----------------
    vis = ~np.isnan(bx)
    dx = (px - bx[:, None]) * L
    dy = (py - by[:, None]) * W
    d = np.hypot(dx, dy)                      # NaN where ball or player missing
    dh, da = _nearest(d, home), _nearest(d, away)
    dh[~vis], da[~vis] = np.nan, np.nan
    dmin = np.fmin(dh, da)
    diff = da - dh                            # > 0: the home team is closer to the ball
    lag = 5
    v = np.full(T, np.nan)
    v[lag:] = np.hypot((bx[lag:] - bx[:-lag]) * L, (by[lag:] - by[:-lag]) * W) * FPS / lag
    v = np.where(v > 45.0, np.nan, v)
    border = np.minimum(np.minimum(bx, 1 - bx), np.minimum(by, 1 - by))
    near_home = np.where(np.isnan(diff), np.nan, (diff > 0).astype(np.float64))
    near_home_close = np.where(np.isnan(dmin) | (dmin > 5.0), np.nan, near_home)

    raw = {"bx": bx, "by_abs": np.abs(by - 0.5), "dh": dh, "da": da, "dmin": dmin, "diff": diff,
           "speed": v, "border": border,
           "n5_h": np.where(vis, _count_within(d, home, 5.0), np.nan),
           "n5_a": np.where(vis, _count_within(d, away, 5.0), np.nan),
           "n10_h": np.where(vis, _count_within(d, home, 10.0), np.nan),
           "n10_a": np.where(vis, _count_within(d, away, 10.0), np.nan)}
    for k, val in raw.items():
        f["b_" + k] = val.astype(np.float32)
    for k in ("diff", "dmin", "speed", "bx", "border", "n5_h", "n5_a"):
        for w in WINDOWS:
            f["bw_%s_w%d" % (k, w)] = wmean(raw[k], w)
    for w in (101, 301, 751):
        f["bw_near_home_w%d" % w] = wmean(near_home, w)
        f["bw_near_home_close_w%d" % w] = wmean(near_home_close, w)
    for w in WINDOWS:
        f["bw_speed_max_w%d" % w] = _wmax(v, w)

    # ---------------- last seen / next seen ----------------
    last, nxt, age_last, age_next = _last_next(vis)
    keep = {"diff": wmean(diff, 25), "dmin": wmean(dmin, 25), "bx": wmean(bx, 25),
            "speed": wmean(v, 25), "by_abs": wmean(np.abs(by - 0.5), 25)}
    f["bl_age"] = age_last.astype(np.float32)
    f["bn_age"] = age_next.astype(np.float32)
    for k, arr in keep.items():
        f["bl_" + k] = np.where(last >= 0, arr[np.clip(last, 0, T - 1)], np.nan).astype(np.float32)
        f["bn_" + k] = np.where(nxt < T, arr[np.clip(nxt, 0, T - 1)], np.nan).astype(np.float32)

    # ---------------- players around the last / next seen ball position ----------------
    # In a stoppage (throw-in, free kick, goal kick) players gather and stand still around the spot
    # where the ball was last seen; this is observable even when the ball itself is missing.
    for tag, ix, ok in (("l", last, last >= 0), ("n", nxt, nxt < T)):
        ixc = np.clip(ix, 0, T - 1)
        ddx = (px - bx[ixc][:, None]) * L
        ddy = (py - by[ixc][:, None]) * W
        dd = np.where(ok[:, None], np.hypot(ddx, ddy), np.nan)
        with np.errstate(invalid="ignore"):
            c5 = np.where(ok, np.sum(dd < 5.0, axis=1), np.nan).astype(np.float64)
            c10 = np.where(ok, np.sum(dd < 10.0, axis=1), np.nan).astype(np.float64)
            s10 = np.where(ok, np.sum((dd < 10.0) & still, axis=1), np.nan).astype(np.float64)
        for k, arr in (("c5", c5), ("c10", c10), ("s10", s10)):
            f["s%s_%s" % (tag, k)] = arr.astype(np.float32)
            f["s%s_%s_w101" % (tag, k)] = wmean(arr, 101)

    # ---------------- visibility ----------------
    for w in VIS_WINDOWS:
        f["v_frac_w%d" % w] = wmean(vis.astype(np.float64), w)
    f["v_gap_len"] = np.where(vis, 0.0, np.minimum(age_last + age_next, 2 * AGE_CAP)).astype(np.float32)

    if not EXTRA:
        f = {k: v for k, v in f.items() if not _is_extra(k)}
    names = sorted(f)
    X = np.column_stack([np.asarray(f[n], dtype=np.float32) for n in names])
    return names, X, bool(attack_right)


def _wmax(x, win):
    """NaN-aware centred moving max (via a strided view; win is small)."""
    h = win // 2
    xp = np.pad(x.astype(np.float64), (h, h), constant_values=np.nan)
    view = np.lib.stride_tricks.sliding_window_view(xp, win)
    with np.errstate(all="ignore"):
        out = np.nanmax(np.where(np.isnan(view), -np.inf, view), axis=1)
    return np.where(np.isinf(out), np.nan, out).astype(np.float32)


# ---------------------------------------------------------------------------------------------
# Cache: python pipeline/features.py --build
# ---------------------------------------------------------------------------------------------
FEAT_DIR = os.path.join(os.path.dirname(HERE), "data", "pipeline", "features")


def build_all(seeds=3):
    """Features for every degraded DFL match/seed and for the ATD graded halves.

    DFL: data/pipeline/features/<match>_s<seed>.npz  with X1, X2 (float32) and raw labels y1, y2
         (0 H, 1 A, 2 D; NOT debounced, so the debounce N stays a training-time choice) and the
         ball-missing masks m1, m2.
    ATD: data/pipeline/features/atd.npz with X1, X2 for the two graded halves and their frame ranges.
    """
    import glob
    import json
    import labels as lab
    import atd
    os.makedirs(FEAT_DIR, exist_ok=True)
    names = None
    for path in sorted(glob.glob(os.path.join(os.path.dirname(HERE), "data", "pipeline", "dfl_canon", "*.npz"))):
        z = np.load(path)
        out = {}
        for h in (1, 2):
            names, X, ar = build_features(z["h%d_ball" % h], z["h%d_players" % h], z["h%d_team" % h])
            out["X%d" % h] = X
            out["y%d" % h] = lab.raw_labels(z["h%d_status" % h], z["h%d_poss" % h])
            out["m%d" % h] = z["h%d_ball_missing" % h]
            out["right%d" % h] = ar
        np.savez(os.path.join(FEAT_DIR, os.path.basename(path)), **out)
        print("wrote", os.path.basename(path), out["X1"].shape, flush=True)
    ball, players, team = atd.load()
    out = {}
    for h, (a, e) in enumerate(((11185, 90652), (110808, 190198)), 1):
        names, X, ar = build_features(ball[a:e], players[a:e], team)
        out["X%d" % h], out["start%d" % h], out["right%d" % h] = X, a, ar
    np.savez(os.path.join(FEAT_DIR, "atd.npz"), **out)
    with open(os.path.join(FEAT_DIR, "feature_names.json"), "w") as handle:
        json.dump(names, handle)
    print("wrote atd.npz, feature_names.json (%d features)" % len(names))


if __name__ == "__main__":
    if "--build" in sys.argv:
        build_all()
    else:
        print(__doc__)
