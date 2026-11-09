"""Heuristic evidence + a 3-state HMM (H / A / D), decoded with Viterbi.

A sketch, not a tuned solution. It replaces baseline.py's "majority vote over a window" with a
model that knows possessions are intervals:

  1. Per-frame EVIDENCE from the ATD (no learning): which team is closer to the ball, how fast the
     ball moves, whether it is outside the pitch.
  2. A 3-state HMM whose transition matrix makes switching expensive. Viterbi then returns the
     single most likely sequence, so flicker disappears and a possession is one run.
  3. Where the ball is missing the evidence is flat, so Viterbi simply carries the previous state
     through the gap. That is the "repeat my previous answer" behaviour EVALUATION.md says is free
     inside a possession.

    python starter/HMM.py                    # writes evaluation/results/hmm_sketch.csv
    python evaluation/run_evaluation.py      # sanity check only -- see below

Every constant below is a PRIOR chosen from football reasoning, not fitted. Do not tune them against
the public 5 minutes (RULES.md, EVALUATION.md): 17 possessions pick the wrong value. If you want to
learn them, do it on labels from OTHER games.

Known weaknesses, deliberately left as openings:
  * dead-ball detection is crude (ball off the pitch, or ball nearly still for a second or more);
  * the Smart Tagging (set pieces = restarts) is not used yet;
  * a player holding the ball still can look like a dead ball;
  * H<->A switches are penalised the same everywhere; a learned model would do better.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from baseline import (FIELD_SEP, GROUP_SEP, MISSING, RESULTS_DIR,  # noqa: E402
                      team_of_each_track, write_submission)
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402

FPS = 25
HOME, AWAY = 0, 1
H, A, D = 0, 1, 2
LABELS = np.array(["H", "A", "D"])

# ---- priors (not tuned) ---------------------------------------------------------------------
# Evidence for H/A: a logistic on (distance of nearest away - nearest home), in metres.
CLOSER_SLOPE = 0.6          # per metre: 2 m closer -> ~77% for that team
LOOSE_BALL_DISTANCE = 6.0   # evidence fades when nobody is near the ball (loose / in the air)

# Evidence for D.
STILL_SPEED = 0.7           # m/s: below this the ball is "not moving"
STILL_FRAMES = FPS          # ... for at least this many consecutive frames
OFF_PITCH_MARGIN = 0.01     # normalised: ball this far outside 0-1 counts as out of play
D_LOGLIK_STRONG = 3.0       # log-odds boost for D when the ball is off the pitch
D_LOGLIK_STILL = 1.5        # ... and when it has been still for a second
D_LOGLIK_PLAY = -0.5        # mild penalty for D when the ball is moving on the pitch

# "Mid-air" ball -> X instead of D. The ATD has no ball height, so flight is inferred from
# kinematics: a fast ball that nobody is near is almost certainly a pass, cross or clearance.
SPEED_LAG_FRAMES = 5        # ball speed = displacement over this many frames (0.2 s)
FLIGHT_SPEED = 10.0         # m/s: faster than a dribble or a held ball
FLIGHT_DISTANCE = 3.0       # m: nearest player is farther than this

# Transitions, as expected seconds between changes -> per-frame switch probability.
SECONDS_BETWEEN_SWITCHES = 12.0   # H<->A: possessions last ~14 s in the public sample, but
                                  # check the full-match number is NOT published; this is a prior
SECONDS_BETWEEN_STOPS = 40.0      # live -> dead
SECONDS_OF_DEAD_BALL = 10.0       # dead -> live


def read_features(raw_path, team_of_track, n_frames, pitch):
    """One pass over the ATD. Returns per-frame arrays (NaN where unknown):

    d_home, d_away  distance (m) from the ball to the nearest tracked player of each team
    speed           ball speed (m/s), computed from positions -- the ATD gives the ball no speed
    bx, by          ball position, normalised 0-1
    """
    length, width = pitch
    d_home = np.full(n_frames, np.nan)
    d_away = np.full(n_frames, np.nan)
    speed = np.full(n_frames, np.nan)
    bx = np.full(n_frames, np.nan)
    by = np.full(n_frames, np.nan)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3 or parts[2][0] == MISSING:
                continue
            frame = int(parts[0]) - 1
            ball = parts[2].split(",")
            x, y = float(ball[0]), float(ball[1])   # ball[2] is NaN in this export
            bx[frame], by[frame] = x, y

            best = [np.inf, np.inf]
            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if group[0] == MISSING:
                    continue
                p = group.split(",")
                dx = (float(p[0]) - x) * length
                dy = (float(p[1]) - y) * width
                d = dx * dx + dy * dy
                team = team_of_track[slot]
                if d < best[team]:
                    best[team] = d
            d_home[frame] = np.sqrt(best[HOME]) if np.isfinite(best[HOME]) else np.nan
            d_away[frame] = np.sqrt(best[AWAY]) if np.isfinite(best[AWAY]) else np.nan
    return d_home, d_away, ball_speed(bx, by, pitch), bx, by


def ball_speed(bx, by, pitch, lag=SPEED_LAG_FRAMES):
    """Ball speed in m/s from the displacement over `lag` frames. NaN unless both ends are seen."""
    length, width = pitch
    speed = np.full(len(bx), np.nan)
    dx = (bx[lag:] - bx[:-lag]) * length
    dy = (by[lag:] - by[:-lag]) * width
    speed[lag:] = np.hypot(dx, dy) * FPS / lag
    return speed


def in_flight(d_home, d_away, speed):
    """True where the ball is moving fast and nobody is near it: treated as 'in the air'."""
    nearest = np.fmin(d_home, d_away)
    with np.errstate(invalid="ignore"):
        return (speed > FLIGHT_SPEED) & (nearest > FLIGHT_DISTANCE)


def still_run(speed, min_frames):
    """True where the ball has been below STILL_SPEED for at least `min_frames` in a row.

    NaN (ball not seen) breaks a run: we do not assume a missing ball is a still one.
    """
    slow = np.nan_to_num(speed, nan=np.inf) < STILL_SPEED
    out = np.zeros(len(slow), dtype=bool)
    start = None
    for i, flag in enumerate(np.append(slow, False)):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= min_frames:
                out[start:i] = True
            start = None
    return out


def log_evidence(d_home, d_away, speed, bx, by):
    """(n_frames, 3) log-likelihood-style scores for H, A, D. Flat (0) where nothing is known."""
    n = len(d_home)
    ll = np.zeros((n, 3))

    seen = np.isfinite(bx)
    both = seen & np.isfinite(d_home) & np.isfinite(d_away)

    # H vs A: who is closer, faded when the ball is far from everyone.
    diff = d_away[both] - d_home[both]                      # > 0: home is closer
    nearest = np.minimum(d_home[both], d_away[both])
    k = CLOSER_SLOPE * np.exp(-nearest / LOOSE_BALL_DISTANCE)
    ll[both, H] = -np.logaddexp(0.0, -k * diff)             # log sigmoid(k * diff)
    ll[both, A] = -np.logaddexp(0.0, k * diff)

    # D: off the pitch, or still for a while; mild penalty when it is clearly moving in play.
    off = seen & ((bx < -OFF_PITCH_MARGIN) | (bx > 1 + OFF_PITCH_MARGIN)
                  | (by < -OFF_PITCH_MARGIN) | (by > 1 + OFF_PITCH_MARGIN))
    still = still_run(speed, STILL_FRAMES) & ~off
    moving = seen & ~off & ~still
    ll[off, D] += D_LOGLIK_STRONG
    ll[still, D] += D_LOGLIK_STILL
    ll[moving, D] += D_LOGLIK_PLAY
    return ll


def transition_matrix(seconds_between_switches=SECONDS_BETWEEN_SWITCHES):
    """Log transition matrix. Rows = from, columns = to, order H, A, D."""
    p_swap = 1.0 / (seconds_between_switches * FPS)
    p_stop = 1.0 / (SECONDS_BETWEEN_STOPS * FPS)
    p_restart = 1.0 / (SECONDS_OF_DEAD_BALL * FPS)
    T = np.array([
        [1 - p_swap - p_stop, p_swap, p_stop],
        [p_swap, 1 - p_swap - p_stop, p_stop],
        [p_restart / 2, p_restart / 2, 1 - p_restart],
    ])
    return np.log(T)


def viterbi(ll, log_T):
    """Most likely state sequence. ll is (n, 3); log_T is (3, 3)."""
    n = len(ll)
    score = np.log(np.full(3, 1 / 3)) + ll[0]
    back = np.zeros((n, 3), dtype=np.int8)
    for t in range(1, n):
        cand = score[:, None] + log_T          # cand[i, j]: be in i at t-1, move to j
        back[t] = np.argmax(cand, axis=0)
        score = cand[back[t], np.arange(3)] + ll[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = np.argmax(score)
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode_graded(ll, blocks, log_T, flight=None):
    """Run Viterbi separately inside each graded half. Everything else is X (never scored).

    If `flight` is given, frames decoded as D that look like a ball in the air become X: the model
    does not know, so it abstains. On spellF1 an X repeats the previous label, on macroF1 it is a
    miss -- no worse than the wrong D it replaces.
    """
    labels = np.full(len(ll), "X", dtype="<U1")
    for _half, first, last in blocks:
        labels[first:last + 1] = LABELS[viterbi(ll[first:last + 1], log_T)]
    if flight is not None:
        labels[(labels == "D") & flight] = "X"
    return labels


def build(game=GAME, data_dir=None):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    print("reading {} ...".format(os.path.basename(raw_path)))
    feats = read_features(raw_path, team_of_each_track(meta_path), n_frames, GAMES[game]["pitch"])
    d_home, d_away, speed, _bx, _by = feats
    return log_evidence(*feats), in_flight(d_home, d_away, speed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--switch-seconds", type=float, nargs="+",
                        default=[SECONDS_BETWEEN_SWITCHES],
                        help="expected seconds between H<->A switches; one submission per value")
    parser.add_argument("--out", default=RESULTS_DIR, help="folder to write into")
    parser.add_argument("--keep-flight-as-d", action="store_true",
                        help="do not turn D frames with an airborne-looking ball into X")
    args = parser.parse_args()

    ll, flight = build()
    blocks = half_video_blocks(GAME)
    for seconds in args.switch_seconds:
        labels = decode_graded(ll, blocks, transition_matrix(seconds),
                                None if args.keep_flight_as_d else flight)
        name = "hmm_switch{:g}".format(seconds)
        path = write_submission(os.path.join(args.out, name + ".csv"), labels)

        graded = labels != "X"
        changes = int(np.sum((labels[1:] != labels[:-1]) & graded[1:] & graded[:-1]))
        in_graded = np.zeros(len(labels), dtype=bool)
        for _half, first, last in blocks:
            in_graded[first:last + 1] = True
        print("{:<16} H {:>6}  A {:>6}  D {:>6}  X {:>6}  | label changes {}".format(
            name, *(int(np.sum(labels == c)) for c in "HAD"),
            int(np.sum((labels == "X") & in_graded)), changes))
    print("\nNow score them:  python evaluation/run_evaluation.py")
