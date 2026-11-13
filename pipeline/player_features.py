"""Player-only per-frame features, identical for DFL-derived and ATD canonical arrays.

Input is the canonical representation: players [T, S, 3] (x, y normalised 0-1, speed unused
because ATD player speed exists on only ~92% of slots and DFL's is a different estimator),
team [S] (0 home / 1 away). Speeds are recomputed from positions so both sources share one
definition. Windows are centred (offline decoding), in seconds.

Nothing here touches the ball or any label, so it can be fitted on the unlabelled test match.
"""

import numpy as np

FPS = 25
LENGTH, WIDTH = 105.3, 68.0


def _smooth(x, win):
    """Centred moving average along axis 0, NaN-aware, win in frames."""
    if win <= 1:
        return x
    ok = ~np.isnan(x)
    v = np.where(ok, x, 0.0)
    k = np.ones(win)
    num = np.apply_along_axis(lambda a: np.convolve(a, k, "same"), 0, v)
    den = np.apply_along_axis(lambda a: np.convolve(a, k, "same"), 0, ok.astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def player_speed(players):
    """Finite-difference speed in m/s per slot, [T, S]; NaN where either frame is missing."""
    dx = np.diff(players[:, :, 0], axis=0) * LENGTH
    dy = np.diff(players[:, :, 1], axis=0) * WIDTH
    v = np.hypot(dx, dy) * FPS
    v = np.where(v > 12.0, np.nan, v)  # identity switches / teleports
    return np.vstack([v[:1], v])


def team_features(players, team, win_seconds=2.0):
    """Dict of [T] arrays, windowed over +/- win_seconds/2."""
    win = int(win_seconds * FPS) | 1
    x = players[:, :, 0]
    y = players[:, :, 1]
    speed = player_speed(players)
    out = {}
    with np.errstate(invalid="ignore"):
        out["n_tracked"] = np.sum(~np.isnan(x), axis=1).astype(np.float32)
        out["speed_all"] = _smooth(np.nanmean(speed, axis=1), win)
        for name, mask in (("h", team == 0), ("a", team == 1)):
            out["cx_" + name] = _smooth(np.nanmean(x[:, mask], axis=1), win)
            out["cy_" + name] = _smooth(np.nanmean(y[:, mask], axis=1), win)
            out["sx_" + name] = _smooth(np.nanstd(x[:, mask], axis=1), win)
            out["sy_" + name] = _smooth(np.nanstd(y[:, mask], axis=1), win)
            out["speed_" + name] = _smooth(np.nanmean(speed[:, mask], axis=1), win)
    out["cx_gap"] = out["cx_h"] - out["cx_a"]
    out["abs_cx_mid"] = np.abs((out["cx_h"] + out["cx_a"]) / 2 - 0.5)
    return out


def stack(feats):
    names = sorted(feats)
    return names, np.column_stack([feats[n] for n in names]).astype(np.float32)


def ball_features(ball, players, win_seconds=2.0):
    """Ball features that exist wherever the ball is observed. ball [T, 3+] (x, y normalised).

    Speed is recomputed from positions: the ATD's own ball speed channel is present on <8% of
    visible frames. Returns dict of [T] arrays; NaN where the ball is not observed.
    """
    win = int(win_seconds * FPS) | 1
    bx, by = ball[:, 0] * LENGTH, ball[:, 1] * WIDTH
    v = np.hypot(np.diff(bx), np.diff(by)) * FPS
    v = np.where(v > 45.0, np.nan, v)  # tracker jumps
    v = np.concatenate([v[:1], v])
    d = np.hypot(players[:, :, 0] * LENGTH - bx[:, None], players[:, :, 1] * WIDTH - by[:, None])
    with np.errstate(invalid="ignore"):
        near = np.where(np.all(np.isnan(d), axis=1), np.nan, np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1))
    near = np.where(np.isinf(near), np.nan, near)
    return {"ball_speed": _smooth(v, win), "ball_near": _smooth(near, win),
            "ball_x": ball[:, 0], "ball_y": ball[:, 1]}


def visibility_fraction(visible, win_seconds=10.0):
    return _smooth(visible.astype(np.float32), int(win_seconds * FPS) | 1)
