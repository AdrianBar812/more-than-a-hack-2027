"""Make DFL tracking look like Metrica ATD, so a model trained on DFL transfers to the test match.

Steps (all seeded, all algorithmic -- no labels of the competition match are used):
  1. canonical(): DFL metres (origin = pitch centre) -> 0-1 coordinates, anonymous per-team slots.
  2. degrade_players(): position noise + block dropouts so ~20.5 players are tracked on average.
  3. fit_missing_model(): on the UNLABELLED ATD, learn P(ball missing | player-only features).
     The ATD loses the ball in long blocks (median ~11 s) and mostly while play is slow or the ball
     is near a goal line, so an i.i.d. dropout would erase a real cue.
  4. sample_ball_mask(): block-structured sampling of the ball-missing mask from that probability.

    python pipeline/degrade.py --seeds 3        # writes data/pipeline/dfl_canon/<match>_s<seed>.npz
"""

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import atd  # noqa: E402
import player_features as pf  # noqa: E402

FPS = 25
RAW_DIR = os.path.join(ROOT, "data", "pipeline", "dfl_raw")
OUT_DIR = os.path.join(ROOT, "data", "pipeline", "dfl_canon")
MODEL_PATH = os.path.join(ROOT, "data", "pipeline", "missing_model.txt")
MISS_WIN = 4.0           # seconds, window of the player features the missing model sees
BLOCK_RATE = 0.14        # 1/s: sum of loss and regain rates; set so mean missing-run ~17 s as in the ATD
GRADED = ((11185, 90652), (110808, 190198))


# ---- 1. canonical conversion ---------------------------------------------------------------
def to_slots(players, team):
    """[P, T, 3] metres -> [T, S, 3] with persons packed into anonymous slots, per team.

    A slot is reused by a later person once the earlier one has left the pitch (interval
    colouring), so slots carry no identity, like the ATD tracks.
    """
    P, T, _ = players.shape
    valid = ~np.isnan(players[:, :, 0])
    parts, slot_team = [], []
    for tm in (0, 1):
        idx = sorted((i for i in range(P) if team[i] == tm and valid[i].any()),
                     key=lambda i: int(np.argmax(valid[i])))
        ends, assign = [], {}
        for i in idx:
            s = int(np.argmax(valid[i]))
            e = T - int(np.argmax(valid[i][::-1]))
            for k, en in enumerate(ends):
                if en <= s:
                    ends[k], assign[i] = e, k
                    break
            else:
                ends.append(e)
                assign[i] = len(ends) - 1
        arr = np.full((T, len(ends), 3), np.nan, dtype=np.float32)
        for i, k in assign.items():
            arr[valid[i], k] = players[i][valid[i]]
        parts.append(arr)
        slot_team += [tm] * len(ends)
    return np.concatenate(parts, axis=1), np.array(slot_team, dtype=np.int8)


def canonical(raw_path):
    """-> list of half dicts: ball_true [T,2], players [T,S,3], team [S], status, poss (DFL truth)."""
    z = np.load(raw_path)
    px, py = [float(v) for v in z["pitch"]]
    halves = []
    for h in (1, 2):
        players, team = to_slots(z["h%d_players" % h], z["h%d_player_team" % h])
        players[:, :, 0] = players[:, :, 0] / px + 0.5
        players[:, :, 1] = players[:, :, 1] / py + 0.5
        players[:, :, 2] = np.nan  # speeds are recomputed from positions on both sides
        ball = z["h%d_ball" % h][:, :2].copy()
        ball[:, 0] = ball[:, 0] / px + 0.5
        ball[:, 1] = ball[:, 1] / py + 0.5
        halves.append(dict(half=h, ball_true=ball, players=players, team=team,
                           status=z["h%d_ball_status" % h], poss=z["h%d_ball_poss" % h]))
    return halves


# ---- 2. player degradation -----------------------------------------------------------------
def alt_runs(T, mean_on, mean_off, rng):
    """Boolean [T]: alternating on/off blocks with geometric lengths (seconds)."""
    mask = np.ones(T, dtype=bool)
    i, on = 0, bool(rng.integers(2))
    while i < T:
        n = int(rng.geometric(1.0 / ((mean_on if on else mean_off) * FPS)))
        if not on:
            mask[i:i + n] = False
        i += n
        on = not on
    return mask


def smooth_noise(shape, sigma, rng, corr_seconds=1.0):
    """Gaussian noise with std `sigma`, correlated over ~corr_seconds along axis 0.

    White per-frame noise would add ~sigma*sqrt(2)*25 m/s of fake finite-difference speed and
    wreck every speed feature; real tracker error is slowly varying.
    """
    k = int(corr_seconds * FPS)
    white = rng.normal(0, 1, (shape[0] + k - 1,) + tuple(shape[1:])).astype(np.float32)
    cs = np.cumsum(white, axis=0, dtype=np.float64)
    cs = np.concatenate([np.zeros((1,) + cs.shape[1:]), cs], axis=0)
    out = (cs[k:] - cs[:-k]) / np.sqrt(k)  # moving sum / sqrt(k): unit variance
    return (out[: shape[0]] * sigma).astype(np.float32)


def degrade_players(players, rng, noise_m=0.05, mean_up=300.0, mean_out=2.0):
    """Position noise and per-slot dropout blocks (~1.4 of 22 players untracked on average)."""
    out = players.copy()
    T, S, _ = out.shape
    out[:, :, 0] += smooth_noise((T, S), noise_m / pf.LENGTH, rng, 3.0)
    out[:, :, 1] += smooth_noise((T, S), noise_m / pf.WIDTH, rng, 3.0)
    for s in range(S):
        out[~alt_runs(T, mean_up, mean_out, rng), s] = np.nan
    return view_loss(out, rng)


def view_loss(players, rng, mean_up=40.0, mean_out=8.0):
    """Episodes where the tracker loses several players at once (camera view / far side of play).

    The ATD sometimes tracks only ~14 of 22 players; per-slot dropout cannot produce that. During
    an episode the k players farthest (in x) from the median tracked player are dropped each frame.
    """
    out = players.copy()
    episode = ~alt_runs(len(out), mean_up, mean_out, rng)
    cut = np.flatnonzero(np.diff(episode.astype(int)) != 0) + 1
    for seg in np.split(np.arange(len(out)), cut):
        if not episode[seg[0]]:
            continue
        k = int(rng.integers(1, 10))
        x = out[seg, :, 0]
        with np.errstate(invalid="ignore"):
            d = np.abs(x - np.nanmedian(x, axis=1, keepdims=True))
        d = np.where(np.isnan(d), -1.0, d)
        far = np.argsort(-d, axis=1)[:, :k]
        rows = np.repeat(np.arange(len(seg)), k)
        sub = out[seg]
        sub[rows, far.ravel()] = np.nan
        out[seg] = sub
    return out


# ---- 3. missingness model (fitted on the unlabelled ATD) ------------------------------------
def _feature_matrix(players, team):
    names, X = pf.stack(pf.team_features(players, team, win_seconds=MISS_WIN))
    return names, X


def fit_missing_model(path=MODEL_PATH):
    import lightgbm as lgb
    ball, players, team = atd.load()
    names, X = _feature_matrix(players, team)
    y = np.isnan(ball[:, 0]).astype(int)
    graded = np.zeros(len(y), bool)
    for a, e in GRADED:
        graded[a:e] = True
    idx = np.flatnonzero(graded)[::5]
    params = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=500,
                  bagging_fraction=0.5, bagging_freq=1, feature_fraction=0.8, verbose=-1, seed=0)
    model = lgb.train(params, lgb.Dataset(X[idx], y[idx], feature_name=names), 150)
    model.save_model(path)
    return model


def load_missing_model():
    import lightgbm as lgb
    if not os.path.exists(MODEL_PATH):
        return fit_missing_model()
    return lgb.Booster(model_file=MODEL_PATH)


# ---- 4. block-structured mask --------------------------------------------------------------
def sample_ball_mask(p, rng, rate=BLOCK_RATE):
    """True = ball missing. Two-state chain with stationary P(missing) = p[t] and cycle rate `rate`.

    Loss rate = rate * p, regain rate = rate * (1 - p), per second. Run lengths come out
    geometric (the ATD's are heavier tailed), which is the known approximation here.
    """
    T = len(p)
    q = np.clip(p, 0.02, 0.98)
    lose = (rate * q / FPS).astype(np.float64)
    regain = (rate * (1 - q) / FPS).astype(np.float64)
    u = rng.random(T)
    missing = np.zeros(T, dtype=bool)
    state = bool(rng.random() < q[0])
    for t in range(T):
        if state:
            if u[t] < regain[t]:
                state = False
        elif u[t] < lose[t]:
            state = True
        missing[t] = state
    return missing


def degrade_half(half, model, rng, ball_noise_m=0.05):
    players = degrade_players(half["players"], rng)
    _, X = _feature_matrix(players, half["team"])
    p = model.predict(X)
    p = np.where(np.isnan(p), 0.4, p)
    missing = sample_ball_mask(p, rng)
    ball = half["ball_true"].copy()
    ball[:, 0] += smooth_noise((len(ball),), ball_noise_m / pf.LENGTH, rng, 0.2)
    ball[:, 1] += smooth_noise((len(ball),), ball_noise_m / pf.WIDTH, rng, 0.2)
    ball[missing] = np.nan
    return dict(half, players=players, ball=ball, ball_missing=missing, p_missing=p.astype(np.float32))


def build(seeds, out_dir=OUT_DIR):
    model = load_missing_model()
    os.makedirs(out_dir, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(RAW_DIR, "*.npz"))):
        mid = os.path.basename(path)[:-4]
        halves = canonical(path)
        for seed in range(seeds):
            rng = np.random.default_rng([seed, abs(hash(mid)) % (2 ** 31)] if False else [seed, int(mid[-4:], 36)])
            arrays = {}
            for half in halves:
                d = degrade_half(half, model, rng)
                h = d["half"]
                for k in ("ball", "ball_true", "players", "team", "status", "poss", "ball_missing", "p_missing"):
                    arrays["h%d_%s" % (h, k)] = d[k]
            np.savez_compressed(os.path.join(out_dir, "%s_s%d.npz" % (mid, seed)), **arrays)
            miss = np.mean(np.concatenate([arrays["h1_ball_missing"], arrays["h2_ball_missing"]]))
            print("%s seed %d: ball missing %.1f%%" % (mid, seed, 100 * miss), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--refit", action="store_true", help="refit the missingness model on the ATD")
    args = ap.parse_args()
    if args.refit or not os.path.exists(MODEL_PATH):
        fit_missing_model()
    build(args.seeds)
