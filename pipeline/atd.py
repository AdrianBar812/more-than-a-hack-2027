"""Load the Metrica tactical ATD into the canonical arrays (and cache them).

Canonical frame arrays (all float32, coordinates normalised 0-1, speed m/s, NaN = missing):
    ball     [T, 3]      x, y, speed
    players  [T, S, 3]   x, y, speed per track slot
    team     [S]         0 = home, 1 = away
Frame index i == submission row i (ATD frame_id - 1).

    python pipeline/atd.py        # writes data/pipeline/atd_tactical.npz
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from baseline import team_of_each_track  # noqa: E402
from games import GAME, GAMES, require, source_frames  # noqa: E402

CACHE = os.path.join(ROOT, "data", "pipeline", "atd_tactical.npz")


def load(game=GAME, use_cache=True):
    if use_cache and os.path.exists(CACHE):
        z = np.load(CACHE)
        return z["ball"], z["players"], z["team"]
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, None)
    team = team_of_each_track(meta_path)
    S = len(team)
    T = source_frames("tactical", game)
    ball = np.full((T, 3), np.nan, dtype=np.float32)
    players = np.full((T, S, 3), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            i = int(parts[0]) - 1
            vals = np.array((parts[1].replace(";", ",") + "," + parts[2]).split(","), dtype=np.float32)
            players[i] = vals[:-3].reshape(S, 3)
            ball[i] = vals[-3:]
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, ball=ball, players=players, team=team)
    return ball, players, team


if __name__ == "__main__":
    b, p, t = load(use_cache=False)
    print("ball", b.shape, "visible %.1f%%" % (100 * np.mean(~np.isnan(b[:, 0]))))
    print("players", p.shape, "home slots", int((t == 0).sum()), "away slots", int((t == 1).sum()))
