"""Explain why HMM.py decided D on a given stretch of frames.

For every frame the HMM calls D, this says which of three things put it there:

  off-pitch   the ball is outside the pitch                        (direct evidence for D)
  still       the ball has barely moved for a second or more        (direct evidence for D)
  gap         the ball is not tracked, so the evidence is flat and Viterbi carried the state through
  inertia     the ball is tracked and moving on the pitch, no evidence for D, yet the model stayed in D
              because leaving it costs more than the evidence is worth

It also marks frames that the mid-air rule would turn into X (flight), and prints the raw evidence
next to the decision so you can see how close the call was.

    python starter/tested_models/explain_hmm.py                          # every D run in the graded halves
    python starter/tested_models/explain_hmm.py --start 12800 --end 13000          # D runs overlapping a range
    python starter/tested_models/explain_hmm.py --start 12800 --end 13000 --frames # per-frame table
    python starter/tested_models/explain_hmm.py --switch-seconds 20      # use the same value you decoded with

Run it with the same --switch-seconds (and the same constants in HMM.py) as the submission you are
trying to understand, or the decisions will not match.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))        # .../starter/tested_models
STARTER = os.path.dirname(HERE)                          # baseline.py lives here
ROOT = os.path.dirname(STARTER)                          # games.py lives here
sys.path.insert(0, HERE)      # HMM.py sits next to this file
sys.path.insert(0, STARTER)
sys.path.insert(0, ROOT)
import HMM  # noqa: E402
from baseline import team_of_each_track  # noqa: E402
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402

FPS = HMM.FPS
CAUSES = ("off-pitch", "still", "gap", "inertia")


def mmss(frame):
    """Video time of a frame as m:ss.s (frame / 25 -- not the match clock)."""
    seconds = frame / FPS
    return "{}:{:04.1f}".format(int(seconds // 60), seconds % 60)


def classify(ll, seen):
    """Cause of each frame's D, as an array of strings (meaningful only where the label is D)."""
    cause = np.full(len(ll), "inertia", dtype=object)
    cause[~seen] = "gap"
    cause[ll[:, HMM.D] >= HMM.D_LOGLIK_STILL - 1e-9] = "still"
    cause[ll[:, HMM.D] >= HMM.D_LOGLIK_STRONG - 1e-9] = "off-pitch"
    return cause


def d_runs(labels, first, last):
    """[(start, end)] of consecutive D frames inside [first, last], end inclusive."""
    runs, start = [], None
    for frame in range(first, last + 1):
        if labels[frame] == "D" and start is None:
            start = frame
        elif labels[frame] != "D" and start is not None:
            runs.append((start, frame - 1))
            start = None
    if start is not None:
        runs.append((start, last))
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", type=int, help="first video frame (default: start of match)")
    parser.add_argument("--end", type=int, help="last video frame (default: end of match)")
    parser.add_argument("--switch-seconds", type=float, default=HMM.SECONDS_BETWEEN_SWITCHES,
                        help="the value the submission was decoded with")
    parser.add_argument("--min-dead-seconds", type=float, default=HMM.MIN_DEAD_SECONDS,
                        help="the value the submission was decoded with")
    parser.add_argument("--frames", action="store_true", help="print a per-frame table")
    parser.add_argument("--step", type=int, default=5, help="frame step for --frames (default 5)")
    args = parser.parse_args()

    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, None)
    n_frames = source_frames("tactical", GAME)
    print("reading {} ...".format(os.path.basename(raw_path)))
    d_home, d_away, speed, bx, by = HMM.read_features(
        raw_path, team_of_each_track(meta_path), n_frames, GAMES[GAME]["pitch"])
    ll = HMM.log_evidence(d_home, d_away, speed, bx, by)
    flight = HMM.in_flight(d_home, d_away, speed)
    seen = np.isfinite(bx)

    labels = HMM.decode_graded(ll, half_video_blocks(GAME),
                               HMM.transition_matrix(args.switch_seconds),
                               min_dead_frames=int(round(args.min_dead_seconds * FPS)))  # before the X rules
    cause = classify(ll, seen)

    lo = args.start if args.start is not None else 0
    hi = args.end if args.end is not None else n_frames - 1

    if args.frames:
        print("\n  frame   time     label  ball   speed  d_home d_away |   ll_H    ll_A    ll_D  cause")
        for f in range(lo, hi + 1, max(1, args.step)):
            if labels[f] == "X" and not (lo <= f <= hi):
                continue
            note = ""
            if labels[f] == "D":
                note = cause[f] + ("  +flight->X" if flight[f] else "")
            print("{:7d}  {:>7}    {}     {}  {:6.1f} {:6.1f} {:6.1f} | {:7.2f} {:7.2f} {:7.2f}  {}".format(
                f, mmss(f), labels[f], "yes" if seen[f] else "NO ", speed[f] if seen[f] else np.nan,
                d_home[f], d_away[f], ll[f, 0], ll[f, 1], ll[f, 2], note))
        return

    print("\nD runs that overlap frames {}..{} (switch-seconds {:g}):\n".format(lo, hi, args.switch_seconds))
    print("  {:>7}-{:<7} {:>7}-{:<7} {:>6}  {}  {}".format(
        "start", "end", "from", "to", "secs", "  ".join("{:>9}".format(c) for c in CAUSES), "flight->X"))
    totals = {c: 0 for c in CAUSES}
    for first, last, in HMM_blocks():
        for start, end in d_runs(labels, first, last):
            if end < lo or start > hi:
                continue
            frames = np.arange(start, end + 1)
            counts = {c: int(np.sum(cause[frames] == c)) for c in CAUSES}
            for c in CAUSES:
                totals[c] += counts[c]
            print("  {:>7}-{:<7} {:>7}-{:<7} {:6.1f}  {}  {:>9}".format(
                start, end, mmss(start), mmss(end + 1), (end - start + 1) / FPS,
                "  ".join("{:>9}".format(counts[c]) for c in CAUSES), int(np.sum(flight[frames]))))

    total = sum(totals.values())
    if total:
        print("\n  overall: " + ", ".join("{} {:.0%}".format(c, totals[c] / total) for c in CAUSES))
        print("  'gap' and 'inertia' mean the model had no evidence for D in that frame: the label came "
              "from the frames around it.")
    else:
        print("  (no D frames in this range)")


def HMM_blocks():
    """[(first, last)] video-frame ranges of the graded halves."""
    return [(first, last) for _half, first, last in half_video_blocks(GAME)]


if __name__ == "__main__":
    main()
