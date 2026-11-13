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
import json
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
# Scale: the H and A scores are log-sigmoids, so an even split scores ln(0.5) = -0.69 each. D must
# score BELOW that when the ball is moving in play, or D wins every time the teams are ambiguous.
D_LOGLIK_PLAY = -1.0        # penalty for D when the ball is tracked and moving on the pitch
D_LOGLIK_MISSING = 0.3      # small lean towards D when the ball is not tracked at all (weak
                            # evidence: live play loses the ball too -- crowded boxes, aerial duels)
MIN_DEAD_SECONDS = 2.0      # a D run shorter than this is not a stoppage: replaced by the label
                            # before it

# "Mid-air" ball -> X instead of D. The ATD has no ball height, so flight is inferred from
# kinematics: a fast ball that nobody is near is almost certainly a pass, cross or clearance.
SPEED_LAG_FRAMES = 5        # ball speed = displacement over this many frames (0.2 s)
FLIGHT_SPEED = 10.0         # m/s: faster than a dribble or a held ball
FLIGHT_DISTANCE = 3.0       # m: nearest player is farther than this
# Ambiguous ball -> X instead of D. When the team evidence is a near coin flip (nobody clearly
# closer), the model has no business choosing a state. |ll_H - ll_A| is the log-odds gap between
# the two teams: 0.3 is roughly a 43/57 split.
AMBIGUOUS_LLR = 0.3
# Transitions, as expected seconds between changes -> per-frame switch probability.
SECONDS_BETWEEN_SWITCHES = 16.9   # H<->A: possessions last ~14 s in the public sample, but
                                  # check the full-match number is NOT published; this is a prior
SECONDS_BETWEEN_STOPS = 36.2      # live -> dead
SECONDS_OF_DEAD_BALL = 20.1       # dead -> live


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

def ambiguous_ball(ll, d_home, d_away):
    """True where the ball is tracked but the two teams are about equally likely to have it.

    A missing ball is not 'ambiguous': it has no evidence at all and is handled by the HMM carrying
    its state through the gap.
    """
    seen = np.isfinite(d_home) & np.isfinite(d_away)
    return seen & (np.abs(ll[:, H] - ll[:, A]) < AMBIGUOUS_LLR)

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
    ll[~seen, D] += D_LOGLIK_MISSING
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


def drop_short_d(labels, first, last, min_frames):
    """Inside [first, last], replace each D run shorter than `min_frames` with the label before it
    (or the one after it, if the D run opens the half). Real stoppages last seconds; a one-second
    'dead ball' in the middle of a possession is a decoding artefact."""
    if min_frames <= 0:
        return
    start = None
    for frame in range(first, last + 2):
        is_d = frame <= last and labels[frame] == "D"
        if is_d and start is None:
            start = frame
        elif not is_d and start is not None:
            if frame - start < min_frames:
                if start > first:
                    labels[start:frame] = labels[start - 1]
                elif frame <= last:
                    labels[start:frame] = labels[frame]
            start = None


def decode_graded(ll, blocks, log_T, flight=None, ambiguous=None, min_dead_frames=0):
    """Run Viterbi separately inside each graded half. Everything else is X (never scored).

    If `flight` is given, frames decoded as D that look like a ball in the air become X: the model
    does not know, so it abstains. On spellF1 an X repeats the previous label, on macroF1 it is a
    miss -- no worse than the wrong D it replaces.
    """
    labels = np.full(len(ll), "X", dtype="<U1")
    for _half, first, last in blocks:
        labels[first:last + 1] = LABELS[viterbi(ll[first:last + 1], log_T)]
        drop_short_d(labels, first, last, min_dead_frames)
    if flight is not None:
        labels[(labels == "D") & flight] = "X"
    if ambiguous is not None:
        labels[(labels == "D") & ambiguous] = "X"
    return labels


def build(game=GAME, data_dir=None):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    print("reading {} ...".format(os.path.basename(raw_path)))
    feats = read_features(raw_path, team_of_each_track(meta_path), n_frames, GAMES[game]["pitch"])
    d_home, d_away, speed, _bx, _by = feats
    ll = log_evidence(*feats)
    return ll, in_flight(d_home, d_away, speed), ambiguous_ball(ll, d_home, d_away)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--switch-seconds", type=float, nargs="+",
                        default=[SECONDS_BETWEEN_SWITCHES],
                        help="expected seconds between H<->A switches; one submission per value")
    parser.add_argument("--out", default=RESULTS_DIR, help="folder to write into")
    parser.add_argument("--keep-flight-as-d", action="store_true",
                        help="do not turn D frames with an airborne-looking ball into X")
    parser.add_argument("--min-dead-seconds", type=float, default=MIN_DEAD_SECONDS,
                        help="D runs shorter than this are replaced by the previous label (0 = off)")
    parser.add_argument("--keep-ambiguous-as-d", action="store_true",
                        help="do not turn D frames with a coin-flip team evidence into X")
    parser.add_argument("--params",
                        help="JSON written by pipeline/tune_hmm.py (uses its 'best_params'). Sets every "
                             "tuned constant, the switch time, --min-dead-seconds and the two --keep-* "
                             "flags; those CLI options are then ignored.")
    parser.add_argument("--name", help="output file name (without .csv); default hmm_switch<seconds>")
    args = parser.parse_args()

    use_flight, use_ambiguous = not args.keep_flight_as_d, not args.keep_ambiguous_as_d
    min_dead_seconds = args.min_dead_seconds
    if args.params:
        with open(args.params) as handle:
            tuned = json.load(handle)
        tuned = tuned.get("best_params", tuned)
        # Same assignment pipeline/tune_hmm.py makes before scoring, so the result is identical.
        for key, value in tuned.items():
            if key in globals() and key.isupper():
                globals()[key] = value
        args.switch_seconds = [tuned["SECONDS_BETWEEN_SWITCHES"]]
        min_dead_seconds = tuned["MIN_DEAD_SECONDS"]
        use_flight, use_ambiguous = bool(tuned["USE_FLIGHT"]), bool(tuned["USE_AMBIGUOUS"])
        print("parameters from", args.params)

    ll, flight, ambiguous = build()
    blocks = half_video_blocks(GAME)
    for seconds in args.switch_seconds:
        labels = decode_graded(ll, blocks, transition_matrix(seconds),
                                flight if use_flight else None,
                                ambiguous if use_ambiguous else None,
                                int(round(min_dead_seconds * FPS)))
        name = args.name or "hmm_switch{:g}".format(seconds)
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
