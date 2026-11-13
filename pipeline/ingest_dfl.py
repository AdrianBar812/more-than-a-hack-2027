"""Stream-parse one DFL (IDSSE) match into a compact raw npz.

Output (data/pipeline/dfl_raw/<match_id>.npz), one entry per half h in {1, 2}:
    h{h}_frames        int64   [T]        DFL frame numbers (N), contiguous
    h{h}_ball          float32 [T, 3]     x, y (metres, pitch centre = 0,0), speed
    h{h}_ball_status   int8    [T]        1 = alive, 0 = dead, -1 = missing
    h{h}_ball_poss     int8    [T]        1 = home, 2 = away, 0 = missing
    h{h}_players       float32 [P, T, 3]  x, y (m), speed; NaN where the person is not tracked
    h{h}_player_team   int8    [P]        0 = home, 1 = away
    h{h}_player_id     <U20    [P]
plus `home_team_id`, `away_team_id`, `pitch` (length, width).

The DFL file is line-structured (one <Frame .../> per line), so a plain line loop is ~10x faster
than a full XML parse and keeps memory flat. Referee frames are skipped.

    python pipeline/ingest_dfl.py --data-dir /home/bark/Downloads/28196177/data
"""

import argparse
import glob
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "data", "pipeline", "dfl_raw")

ATTR = re.compile(r'(\w+)="([^"]*)"')


def match_info(info_path):
    root = ET.parse(info_path).getroot()
    general = root.find(".//General")
    env = root.find(".//Environment")
    pitch = (float(env.get("PitchX", 105)), float(env.get("PitchY", 68))) if env is not None else (105.0, 68.0)
    return general.get("HomeTeamId"), general.get("GuestTeamId"), general.get("MatchId"), pitch


def parse_positions(pos_path, home_id, away_id):
    """-> {half: dict(person -> (team, list of (N, x, y, s)), ball rows)}"""
    halves = {}
    cur = None
    with open(pos_path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("<FrameSet"):
                a = dict(ATTR.findall(line))
                team = a["TeamId"]
                half = 1 if a["GameSection"] == "firstHalf" else 2
                bucket = halves.setdefault(half, {"ball": [], "players": {}})
                if team == "BALL":
                    cur = bucket["ball"]
                elif team in (home_id, away_id):
                    cur = bucket["players"].setdefault(
                        a["PersonId"], (0 if team == home_id else 1, []))[1]
                else:
                    cur = None  # referee
            elif cur is not None and line.startswith("<Frame "):
                a = dict(ATTR.findall(line))
                if "BallStatus" in a:
                    cur.append((int(a["N"]), float(a["X"]), float(a["Y"]), float(a["S"]),
                                int(a["BallStatus"]), int(a["BallPossession"])))
                else:
                    cur.append((int(a["N"]), float(a["X"]), float(a["Y"]), float(a["S"])))
    return halves


def to_arrays(halves):
    out = {}
    for half, bucket in halves.items():
        ball = np.array(bucket["ball"], dtype=np.float64)
        first, last = int(ball[:, 0].min()), int(ball[:, 0].max())
        frames = np.arange(first, last + 1)
        T = len(frames)
        b = np.full((T, 3), np.nan, dtype=np.float32)
        status = np.full(T, -1, dtype=np.int8)
        poss = np.zeros(T, dtype=np.int8)
        idx = ball[:, 0].astype(np.int64) - first
        b[idx] = ball[:, 1:4]
        status[idx] = ball[:, 4]
        poss[idx] = ball[:, 5]

        ids, teams, arrs = [], [], []
        for pid, (team, rows) in bucket["players"].items():
            r = np.array(rows, dtype=np.float64)
            if len(r) == 0:
                continue
            arr = np.full((T, 3), np.nan, dtype=np.float32)
            i = r[:, 0].astype(np.int64) - first
            ok = (i >= 0) & (i < T)
            arr[i[ok]] = r[ok, 1:4]
            ids.append(pid)
            teams.append(team)
            arrs.append(arr)
        out["h%d_frames" % half] = frames
        out["h%d_ball" % half] = b
        out["h%d_ball_status" % half] = status
        out["h%d_ball_poss" % half] = poss
        out["h%d_players" % half] = np.stack(arrs)
        out["h%d_player_team" % half] = np.array(teams, dtype=np.int8)
        out["h%d_player_id" % half] = np.array(ids)
    return out


def ingest(info_path, pos_path, out_dir=OUT_DIR):
    home_id, away_id, match_id, pitch = match_info(info_path)
    arrays = to_arrays(parse_positions(pos_path, home_id, away_id))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, match_id + ".npz")
    np.savez_compressed(path, home_team_id=home_id, away_team_id=away_id,
                        pitch=np.array(pitch), **arrays)
    return match_id, path


def summarise(path):
    z = np.load(path)
    for h in (1, 2):
        st, po = z["h%d_ball_status" % h], z["h%d_ball_poss" % h]
        alive = st == 1
        lab = np.where(~alive, 2, po - 1)  # 0 H, 1 A, 2 D
        runs = np.flatnonzero(np.diff(lab) != 0) + 1
        seg = np.split(lab, runs)
        poss = [len(s) for s in seg if s[0] in (0, 1)]
        print("  half %d: %6d frames  dead %.1f%%  ball missing %.1f%%  possessions %d  mean %.1fs  players %d" % (
            h, len(st), 100 * np.mean(~alive), 100 * np.mean(st == -1), len(poss),
            np.mean(poss) / 25 if poss else 0, z["h%d_players" % h].shape[0]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/home/bark/Downloads/28196177/data")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--match", help="only this DFL-MAT id")
    args = ap.parse_args()
    for info in sorted(glob.glob(os.path.join(args.data_dir, "*matchinformation*.xml"))):
        mid = re.search(r"(DFL-MAT-\w+)\.xml", info).group(1)
        if args.match and mid != args.match:
            continue
        pos = glob.glob(os.path.join(args.data_dir, "*positions_raw_observed*%s.xml" % mid))[0]
        print("ingesting", mid, flush=True)
        _, path = ingest(info, pos, args.out)
        summarise(path)
