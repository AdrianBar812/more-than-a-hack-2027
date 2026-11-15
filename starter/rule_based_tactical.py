"""Tactical-only, rule-based possession model for More than a Hack 2027.

Inputs
------
* tactical ATD text: player and ball coordinates, one row per tracking frame
* tactical ATD XML: team membership for the 46 tracking slots
* games.py: pitch size, frame count and graded-half frame ranges
* tactical Smart Tagging XML: weak phase-team hints and restart timing context

No broadcast video, broadcast tracking or machine-learning model is used. Tactical
homography is used only for projection confidence and out-of-play corroboration.

The model is deliberately built around possession spells rather than frame accuracy:
1. Measure each team's closest-player distance to the ball and nearby-player support.
2. Smooth team evidence over a short centred window.
3. Use a hysteresis state machine to avoid flicker and confirm genuine H/A changes.
4. Use Smart Tagging as weak evidence: detect the live restart inside set-piece windows,
   label only the pre-restart part D, and bridge post-goal dead-ball time to the next kickoff.
5. Use event-specific restart logic, weak attacking/defensive phase priors, homography confidence,
   and conservative short-gap interpolation of the ball track.
6. Detect stoppages using multi-player motion: the ball must be stationary and most visible
   players must be almost motionless for a sustained, smoothed window. A single nearby
   player moving with/around the ball does not suppress strong team-wide stop evidence.
7. Extend credible out-of-play episodes until a plausible in-play restart.

Run from the repository root:
    python starter/rule_based_tactical.py

Validate the output format:
    python evaluation/validate_submission.py evaluation/results/rule_based_tactical.csv

The public five-minute labels are for pipeline verification only. Do not use them to
choose parameters or compare variants; spellF1 is a whole-match possession metric.
"""

import argparse
import csv
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import baseline as official_baseline  # starter/baseline.py: metadata helpers + CSV writer
from games import GAME, GAMES, half_video_blocks, require, source_frames


UNKNOWN = -1
HOME = 0
AWAY = 1
DEAD = 2
CODE_TO_LABEL = {HOME: "H", AWAY: "A", DEAD: "D", UNKNOWN: "X"}


def _float_or_nan(value):
    """Parse a tracking number, treating NaN/empty values as missing."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def load_tracking(raw_path, n_frames, n_slots):
    """Read ATD rows into player_xy[n_frames,n_slots,2] and ball_xy[n_frames,2].

    ATD frame IDs start at 1; submission/video frame indices start at 0.
    Speeds in the third column are intentionally ignored. Velocities used by the
    stoppage heuristic are derived from positions, avoiding dependence on the
    speed-channel unit convention.
    """
    player_xy = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    ball_xy = np.full((n_frames, 2), np.nan, dtype=np.float32)
    rows_read = 0

    with open(raw_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            parts = line.rstrip("\r\n").split(":")
            if len(parts) != 3:
                continue
            try:
                frame_index = int(parts[0]) - 1
            except ValueError:
                continue
            if frame_index < 0 or frame_index >= n_frames:
                continue

            rows_read += 1
            players = parts[1].split(";")
            if len(players) != n_slots:
                raise ValueError(
                    "ATD line {} has {} player slots; metadata declares {}".format(
                        line_number, len(players), n_slots
                    )
                )

            for slot, group in enumerate(players):
                values = group.split(",")
                if len(values) < 2:
                    continue
                x = _float_or_nan(values[0])
                y = _float_or_nan(values[1])
                if np.isfinite(x) and np.isfinite(y):
                    player_xy[frame_index, slot, 0] = x
                    player_xy[frame_index, slot, 1] = y

            ball_values = parts[2].split(",")
            if len(ball_values) >= 2:
                bx = _float_or_nan(ball_values[0])
                by = _float_or_nan(ball_values[1])
                if np.isfinite(bx) and np.isfinite(by):
                    ball_xy[frame_index, 0] = bx
                    ball_xy[frame_index, 1] = by

    if rows_read == 0:
        raise ValueError("No usable ATD rows found in {}".format(raw_path))

    ball_seen = int(np.isfinite(ball_xy).all(axis=1).sum())
    players_seen = np.isfinite(player_xy).all(axis=2).sum(axis=1)
    print("  tracking rows read: {:,}".format(rows_read))
    print("  ball located: {:,}/{:,} frames ({:.1f}%)".format(
        ball_seen, n_frames, 100.0 * ball_seen / n_frames
    ))
    print("  median visible player slots/frame: {:.1f}".format(float(np.median(players_seen))))
    return player_xy, ball_xy



def interpolate_short_ball_gaps(ball_xy, pitch, max_gap_frames=5, max_speed_mps=24.0):
    """Interpolate only tiny, plausible ball-track gaps.

    Returns a copy of the ball array, a mask for interpolated positions and the
    original-observation mask. Interpolated samples are deliberately lower weight
    than real detections and are never used alone to declare a dead ball.
    """
    out = np.asarray(ball_xy, dtype=np.float32).copy()
    observed = np.isfinite(out).all(axis=1)
    interpolated = np.zeros(len(out), dtype=bool)
    pitch_length, pitch_width = pitch
    n = len(out)
    i = 0
    while i < n:
        if observed[i]:
            i += 1
            continue
        start = i
        while i < n and not observed[i]:
            i += 1
        end = i
        gap = end - start
        if start == 0 or end >= n or gap > max_gap_frames:
            continue
        left = out[start - 1]
        right = out[end]
        delta_m = np.array([(right[0] - left[0]) * pitch_length,
                            (right[1] - left[1]) * pitch_width], dtype=np.float64)
        duration = (gap + 1) / 25.0
        speed = float(np.linalg.norm(delta_m) / max(duration, 1e-6))
        # Do not interpolate across large tracker jumps or obviously corrupted points.
        if not np.isfinite(speed) or speed > max_speed_mps:
            continue
        if np.any(left < -0.03) or np.any(left > 1.03) or np.any(right < -0.03) or np.any(right > 1.03):
            continue
        for offset in range(1, gap + 1):
            alpha = offset / float(gap + 1)
            out[start + offset - 1] = left + alpha * (right - left)
            interpolated[start + offset - 1] = True
    return out, interpolated, observed


def load_homography(path, n_frames):
    """Load tactical per-frame homographies, indexed by video frame (0-based)."""
    matrices = np.full((n_frames, 3, 3), np.nan, dtype=np.float64)
    error = np.full(n_frames, np.nan, dtype=np.float32)
    valid = np.zeros(n_frames, dtype=bool)
    rows = 0
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = ["frame"] + ["h{}{}".format(r, c) for r in range(3) for c in range(3)] + [
            "reprojection_error", "valid"
        ]
        missing = [name for name in required if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError("Homography CSV is missing columns: {}".format(", ".join(missing)))
        for row in reader:
            try:
                frame = int(row["frame"])
            except (TypeError, ValueError):
                continue
            if not 0 <= frame < n_frames:
                continue
            rows += 1
            try:
                matrix = np.array([
                    float(row["h{}{}".format(r, c)])
                    for r in range(3) for c in range(3)
                ], dtype=np.float64).reshape(3, 3)
                reproj = float(row["reprojection_error"])
                is_valid = int(float(row["valid"])) == 1
            except (TypeError, ValueError):
                continue
            if is_valid and np.isfinite(matrix).all() and np.isfinite(reproj):
                matrices[frame] = matrix
                error[frame] = reproj
                valid[frame] = True
    if rows == 0:
        raise ValueError("No usable rows found in homography file {}".format(path))
    usable_error = error[valid]
    median_error = float(np.median(usable_error)) if usable_error.size else float("nan")
    print("  homography rows: {:,}; valid: {:,}/{:,}; median reprojection error: {:.3f}".format(
        rows, int(valid.sum()), n_frames, median_error
    ))
    return {"matrices": matrices, "error": error, "valid": valid}


def project_points(matrices, points):
    """Project normalized pitch points to pixels with one homography per frame."""
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    px = np.full(n, np.nan, dtype=np.float64)
    py = np.full(n, np.nan, dtype=np.float64)
    if matrices is None:
        return px, py
    valid = np.isfinite(points).all(axis=1) & np.isfinite(matrices).all(axis=(1, 2))
    if not valid.any():
        return px, py
    H = matrices[valid]
    xy = points[valid]
    xh = H[:, 0, 0] * xy[:, 0] + H[:, 0, 1] * xy[:, 1] + H[:, 0, 2]
    yh = H[:, 1, 0] * xy[:, 0] + H[:, 1, 1] * xy[:, 1] + H[:, 1, 2]
    den = H[:, 2, 0] * xy[:, 0] + H[:, 2, 1] * xy[:, 1] + H[:, 2, 2]
    good = np.isfinite(den) & (np.abs(den) > 1e-9)
    indices = np.flatnonzero(valid)
    px[indices[good]] = xh[good] / den[good]
    py[indices[good]] = yh[good] / den[good]
    return px, py

def centred_sum(values, radius):
    """Centred rolling sum for a 1-D array, clipped at the sequence edges."""
    values = np.asarray(values, dtype=np.float64)
    if radius <= 0:
        return values.copy()
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    index = np.arange(len(values))
    low = np.maximum(0, index - radius)
    high = np.minimum(len(values), index + radius + 1)
    return cumulative[high] - cumulative[low]


def centred_valid_mean(values, valid, radius):
    """Rolling mean that does not treat missing-ball frames as zero evidence."""
    numerator = centred_sum(np.where(valid, values, 0.0), radius)
    denominator = centred_sum(valid.astype(np.float64), radius)
    result = np.zeros(len(values), dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result, denominator



def centred_weighted_mean(values, weights, radius):
    """Centred rolling mean using per-frame confidence weights."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    numerator = centred_sum(values * weights, radius)
    denominator = centred_sum(weights, radius)
    result = np.zeros(len(values), dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result, denominator

def centred_velocity(points, fps, lag, max_speed):
    """Estimate velocity with a centred finite difference; points end in x,y."""
    points = np.asarray(points, dtype=np.float32)
    n = points.shape[0]
    velocity = np.full(points.shape, np.nan, dtype=np.float32)
    speed = np.full(points.shape[:-1], np.nan, dtype=np.float32)
    if n <= 2 * lag:
        return velocity, speed

    left = points[:-2 * lag]
    right = points[2 * lag:]
    valid = np.isfinite(left).all(axis=-1) & np.isfinite(right).all(axis=-1)
    delta = (right - left) * (fps / float(2 * lag))
    magnitude = np.linalg.norm(delta, axis=-1)
    valid &= np.isfinite(magnitude) & (magnitude <= max_speed)

    middle_velocity = velocity[lag:n - lag]
    middle_speed = speed[lag:n - lag]
    middle_velocity[valid] = delta[valid]
    middle_speed[valid] = magnitude[valid]
    return velocity, speed


def _run_filter(mask, minimum_length, bridge_gap=0):
    """Keep true runs of at least minimum_length; optionally close tiny false gaps."""
    out = np.asarray(mask, dtype=bool).copy()
    n = len(out)

    if bridge_gap > 0 and n:
        i = 0
        while i < n:
            if out[i]:
                i += 1
                continue
            start = i
            while i < n and not out[i]:
                i += 1
            end = i
            if start > 0 and end < n and (end - start) <= bridge_gap:
                out[start:end] = True

    i = 0
    while i < n:
        if not out[i]:
            i += 1
            continue
        start = i
        while i < n and out[i]:
            i += 1
        if i - start < minimum_length:
            out[start:i] = False
    return out


def _remove_short_islands(codes, min_length):
    """Merge a short H/A island only when both surrounding spells agree."""
    out = np.asarray(codes, dtype=np.int8).copy()
    if min_length <= 0 or len(out) < 3:
        return out

    starts = [0]
    ends = []
    values = [int(out[0])]
    for index in range(1, len(out)):
        if out[index] != out[index - 1]:
            ends.append(index)
            starts.append(index)
            values.append(int(out[index]))
    ends.append(len(out))

    replacements = []
    for run_index in range(1, len(values) - 1):
        value = values[run_index]
        run_length = ends[run_index] - starts[run_index]
        if (
            value in (HOME, AWAY)
            and run_length < min_length
            and values[run_index - 1] == values[run_index + 1]
            and values[run_index - 1] in (HOME, AWAY)
        ):
            replacements.append((starts[run_index], ends[run_index], values[run_index - 1]))

    for start, end, value in replacements:
        out[start:end] = value
    return out


def compute_features(player_xy_norm, ball_xy_norm, team_of_track, pitch, fps, velocity_lag,
                     ball_observed=None, ball_interpolated=None, homography=None):
    """Compute team proximity, motion, interpolation confidence and homography cues."""
    pitch_length, pitch_width = pitch
    player_xy_norm = player_xy_norm.astype(np.float32, copy=True)
    ball_xy_norm = ball_xy_norm.astype(np.float32, copy=True)
    player_xy = player_xy_norm.copy()
    ball_xy = ball_xy_norm.copy()

    player_xy[:, :, 0] *= pitch_length
    player_xy[:, :, 1] *= pitch_width
    ball_xy[:, 0] *= pitch_length
    ball_xy[:, 1] *= pitch_width

    ball_available = np.isfinite(ball_xy).all(axis=1)
    if ball_observed is None:
        ball_observed = ball_available.copy()
    else:
        ball_observed = np.asarray(ball_observed, dtype=bool)
    if ball_interpolated is None:
        ball_interpolated = np.zeros(len(ball_xy), dtype=bool)
    else:
        ball_interpolated = np.asarray(ball_interpolated, dtype=bool)

    # Homography quality is used as a soft confidence signal, not as possession evidence.
    # The tactical matrices should be valid throughout; a poor fit is downweighted rather
    # than being allowed to turn a plausible tracking location into a categorical decision.
    n = len(ball_xy)
    h_valid = np.zeros(n, dtype=bool)
    h_error = np.full(n, np.nan, dtype=np.float32)
    h_quality = np.ones(n, dtype=np.float32)
    projected_ball_in_frame = np.zeros(n, dtype=bool)
    projected_ball_x = np.full(n, np.nan, dtype=np.float32)
    projected_ball_y = np.full(n, np.nan, dtype=np.float32)
    projected_player_outlier = np.zeros(player_xy_norm.shape[:2], dtype=bool)
    if homography is not None:
        matrices = homography["matrices"]
        h_valid = np.asarray(homography["valid"], dtype=bool)
        h_error = np.asarray(homography["error"], dtype=np.float32)
        finite_error = np.isfinite(h_error)
        # Median play-time errors are about 0.24 px; this decays gently for poor fits.
        quality_error = np.where(finite_error, np.maximum(h_error - 0.25, 0.0), 2.0)
        h_quality = np.where(h_valid, 1.0 / (1.0 + quality_error / 3.0), 0.65).astype(np.float32)
        h_quality = np.clip(h_quality, 0.40, 1.0)
        px, py = project_points(matrices, ball_xy_norm)
        projected_ball_x = px.astype(np.float32)
        projected_ball_y = py.astype(np.float32)
        projected_ball_in_frame = (
            h_valid & np.isfinite(px) & np.isfinite(py) &
            (px >= -30.0) & (px <= 1950.0) & (py >= -30.0) & (py <= 1110.0)
        )

        # Also project every player slot. We reject only extreme ATD outliers:
        # a player coordinate must be well beyond the pitch AND project well
        # outside the actual image. Ordinary players just beyond a touchline are
        # retained because they can legitimately take throw-ins.
        pxn = player_xy_norm[:, :, 0].astype(np.float64)
        pyn = player_xy_norm[:, :, 1].astype(np.float64)
        den_p = (matrices[:, None, 2, 0] * pxn +
                 matrices[:, None, 2, 1] * pyn + matrices[:, None, 2, 2])
        num_xp = (matrices[:, None, 0, 0] * pxn +
                  matrices[:, None, 0, 1] * pyn + matrices[:, None, 0, 2])
        num_yp = (matrices[:, None, 1, 0] * pxn +
                  matrices[:, None, 1, 1] * pyn + matrices[:, None, 1, 2])
        with np.errstate(divide="ignore", invalid="ignore"):
            pxp = num_xp / den_p
            pyp = num_yp / den_p
        projectable = h_valid[:, None] & np.isfinite(pxp) & np.isfinite(pyp) & (np.abs(den_p) > 1e-9)
        image_outlier = (
            (pxp < -50.0) | (pxp > 1970.0) | (pyp < -50.0) | (pyp > 1130.0)
        )
        coordinate_outlier = (
            (pxn < -0.08) | (pxn > 1.08) | (pyn < -0.08) | (pyn > 1.08)
        )
        projected_player_outlier = projectable & image_outlier & coordinate_outlier

    player_valid = np.isfinite(player_xy).all(axis=2)
    if homography is not None:
        player_valid &= ~projected_player_outlier
    home_slots = np.asarray(team_of_track) == official_baseline.HOME
    away_slots = np.asarray(team_of_track) == official_baseline.AWAY
    if home_slots.sum() == 0 or away_slots.sum() == 0:
        raise ValueError("Metadata must assign at least one slot to each team")

    dx = player_xy[:, :, 0] - ball_xy[:, None, 0]
    dy = player_xy[:, :, 1] - ball_xy[:, None, 1]
    distances = np.hypot(dx, dy)
    distances[~player_valid] = np.inf
    distances[~ball_available, :] = np.inf

    home_distances = distances[:, home_slots]
    away_distances = distances[:, away_slots]
    home_min = np.min(home_distances, axis=1)
    away_min = np.min(away_distances, axis=1)
    home_near = np.sum(home_distances < 4.0, axis=1)
    away_near = np.sum(away_distances < 4.0, axis=1)
    homography_weight = h_quality if homography is not None else np.ones(n, dtype=np.float32)
    home_score = (np.exp(-np.minimum(home_min, 40.0) / 2.2) + 0.045 * np.minimum(home_near, 4)) * homography_weight
    away_score = (np.exp(-np.minimum(away_min, 40.0) / 2.2) + 0.045 * np.minimum(away_near, 4)) * homography_weight
    home_score[~ball_available] = 0.0
    away_score[~ball_available] = 0.0

    _, ball_speed = centred_velocity(ball_xy, fps, velocity_lag, max_speed=35.0)
    _, player_speed = centred_velocity(player_xy, fps, velocity_lag, max_speed=12.0)
    finite_speed = np.isfinite(player_speed)
    visible_speed_count = finite_speed.sum(axis=1).astype(np.int16)
    with np.errstate(all="ignore"):
        median_player_speed = np.ma.median(
            np.ma.masked_invalid(player_speed), axis=1
        ).filled(np.nan).astype(np.float32)

    # Robust whole-pitch movement profile. A stoppage can have one active player
    # (for example, the player retrieving/holding the ball) while most others stop.
    # We therefore measure the fraction of visible players moving very slowly and
    # the 80th-percentile speed, rather than requiring the nearest player to stop.
    safe_speeds = np.where(finite_speed, player_speed, np.inf)
    sorted_speeds = np.sort(safe_speeds, axis=1)
    row_ids = np.arange(n)
    p80_index = np.floor(0.80 * np.maximum(visible_speed_count - 1, 0)).astype(np.int64)
    p80_player_speed = sorted_speeds[row_ids, p80_index].astype(np.float32)
    p80_player_speed[visible_speed_count < 10] = np.nan

    all_min_local = np.argmin(distances, axis=1)
    nearest_distance = np.min(distances, axis=1)
    row_indices = np.arange(len(ball_xy))
    nearest_speed = player_speed[row_indices, all_min_local]
    nearest_speed[~np.isfinite(nearest_distance)] = np.nan

    # Real detections count as 1.0, tiny interpolated gaps as 0.35, both reduced by
    # the frame's homography quality when it is available.
    ball_weight = ball_observed.astype(np.float32) + 0.35 * ball_interpolated.astype(np.float32)
    ball_weight *= homography_weight
    ball_weight[~ball_available] = 0.0

    return {
        "ball_xy_m": ball_xy,
        "ball_xy_norm": ball_xy_norm,
        "ball_valid": ball_observed,
        "ball_available": ball_available,
        "ball_interpolated": ball_interpolated,
        "ball_weight": ball_weight,
        "home_min": home_min,
        "away_min": away_min,
        "home_score": home_score,
        "away_score": away_score,
        "nearest_distance": nearest_distance,
        "ball_speed": ball_speed,
        "nearest_speed": nearest_speed,
        "median_player_speed": median_player_speed,
        "p80_player_speed": p80_player_speed,
        "player_speed": player_speed,
        "visible_speed_count": visible_speed_count,
        "homography_valid": h_valid,
        "homography_error": h_error,
        "homography_quality": h_quality,
        "projected_ball_in_frame": projected_ball_in_frame,
        "projected_ball_x": projected_ball_x,
        "projected_ball_y": projected_ball_y,
        "projected_player_outlier_count": int(projected_player_outlier.sum()),
    }

def _extend_dead_event_until_restart(event_mask, features, fps, max_extension_seconds=15.0):
    """Extend a credible stoppage trigger while players reposition, until restart.

    The trigger may be an out-of-play ball or a sustained whole-team stop. The restart
    needs an observed ball back inside the pitch and sustained movement; one noisy
    position does not end the dead-ball interval. If no restart is found within the
    search horizon, only the original trigger is retained.
    """
    result = np.asarray(event_mask, dtype=bool).copy()
    n = len(result)
    if n == 0:
        return result
    ball_obs = features["ball_valid"]
    ball_xy = features["ball_xy_norm"]
    ball_speed = features["ball_speed"]
    inside = (
        np.isfinite(ball_xy).all(axis=1) &
        (ball_xy[:, 0] >= -0.005) & (ball_xy[:, 0] <= 1.005) &
        (ball_xy[:, 1] >= -0.005) & (ball_xy[:, 1] <= 1.005)
    )
    moving_inside = ball_obs & inside & np.isfinite(ball_speed) & (ball_speed >= 0.85)
    # Require two neighbouring samples so a one-frame tracker jump is not a restart.
    restart_pair = np.zeros(n, dtype=bool)
    if n >= 2:
        restart_pair[:-1] = moving_inside[:-1] & moving_inside[1:]

    i = 0
    max_extension = max(1, int(round(max_extension_seconds * fps)))
    while i < n:
        if not result[i]:
            i += 1
            continue
        start = i
        while i < n and result[i]:
            i += 1
        run_end = i
        search_end = min(n, run_end + max_extension)
        candidates = np.flatnonzero(restart_pair[run_end:search_end])
        if candidates.size:
            restart = run_end + int(candidates[0])
            result[start:restart] = True
    return result


def make_dead_mask(features, fps, stationary_seconds, missing_seconds,
                   dead_smooth_seconds=0.40, quiet_fraction=0.78,
                   quiet_player_speed=0.45, quiet_p80_speed=0.90,
                   quiet_median_speed=0.38, return_details=False):
    """Detect dead balls from independent event, ball, and whole-team motion cues.

    Strong evidence:
      * ball credibly outside the playing area;
      * explicit pre-restart/goal Smart Tag interval;
      * a stationary observed ball while most visible players are nearly stationary;
      * a prolonged missing ball while the visible players are collectively quiet.

    Unlike V3, a confident H/A proximity score does NOT suppress multi-player
    stoppage evidence. The group-motion requirements are deliberately stringent
    because slow play or a player waiting with the ball can otherwise look like D.
    """
    ball_observed = features["ball_valid"]
    ball_available = features["ball_available"]
    ball_xy_norm = features["ball_xy_norm"]
    nearest = features["nearest_distance"]
    ball_speed = features["ball_speed"]
    median_player_speed = features["median_player_speed"]
    p80_player_speed = features["p80_player_speed"]
    player_speed = features["player_speed"]
    visible_speed_count = features["visible_speed_count"]
    finite_player_speed = np.isfinite(player_speed)
    slow_player_count = ((player_speed < quiet_player_speed) & finite_player_speed).sum(axis=1)
    slow_player_fraction = np.full(len(visible_speed_count), np.nan, dtype=np.float32)
    np.divide(slow_player_count, visible_speed_count, out=slow_player_fraction,
              where=visible_speed_count > 0)

    # 1) Ball outside the pitch. Homography is used only to corroborate marginal
    # excursions; clear excursions beyond 2% of pitch dimensions stand on their own.
    x = ball_xy_norm[:, 0]
    y = ball_xy_norm[:, 1]
    outside_pitch = (x < -0.006) | (x > 1.006) | (y < -0.006) | (y > 1.006)
    h_confirm = features["projected_ball_in_frame"] & (features["homography_quality"] >= 0.60)
    strong_outside = (x < -0.020) | (x > 1.020) | (y < -0.020) | (y > 1.020)
    outside_raw = ball_observed & ((outside_pitch & h_confirm) | (strong_outside & ~h_confirm))
    outside_raw = _run_filter(
        outside_raw, minimum_length=max(2, int(0.08 * fps)), bridge_gap=max(1, int(0.08 * fps))
    )
    outside_dead = _extend_dead_event_until_restart(outside_raw, features, fps)

    # 2) Smooth movement measurements over a centred window. This is important:
    # one player can continue moving while the rest of the match has stopped.
    radius = max(1, int(round(dead_smooth_seconds * fps)))
    valid_ball_speed = ball_observed & np.isfinite(ball_speed)
    ball_speed_smooth, ball_speed_support = centred_valid_mean(
        np.nan_to_num(ball_speed, nan=0.0), valid_ball_speed, radius
    )
    valid_group_speed = (
        (visible_speed_count >= 10) & np.isfinite(slow_player_fraction) &
        np.isfinite(median_player_speed) & np.isfinite(p80_player_speed)
    )
    slow_fraction_smooth, group_support = centred_valid_mean(
        np.nan_to_num(slow_player_fraction, nan=0.0), valid_group_speed, radius
    )
    median_speed_smooth, _ = centred_valid_mean(
        np.nan_to_num(median_player_speed, nan=0.0), valid_group_speed, radius
    )
    p80_speed_smooth, _ = centred_valid_mean(
        np.nan_to_num(p80_player_speed, nan=0.0), valid_group_speed, radius
    )
    visible_count_smooth, _ = centred_valid_mean(
        visible_speed_count.astype(np.float32), valid_group_speed, radius
    )

    quiet_group = (
        (group_support >= max(3.0, 0.35 * (2 * radius + 1))) &
        (visible_count_smooth >= 14.0) &
        (slow_fraction_smooth >= quiet_fraction) &
        (median_speed_smooth <= quiet_median_speed) &
        (p80_speed_smooth <= quiet_p80_speed)
    )
    ball_stopped = (
        ball_observed & (ball_speed_support >= 2.0) &
        (ball_speed_smooth <= 0.45)
    )

    # Strong pattern described by the user: ball nearly stopped, most players at
    # rest, but one nearby player may still be moving/standing over the ball.
    group_stop_raw = ball_stopped & quiet_group
    group_stop_trigger = _run_filter(
        group_stop_raw,
        minimum_length=max(1, int(round(stationary_seconds * fps))),
        bridge_gap=max(1, int(round(0.20 * fps))),
    )
    # Once a genuine whole-pitch stop has been established, keep D while players
    # reposition. End the spell at the first sustained in-pitch ball movement.
    group_stop_dead = _extend_dead_event_until_restart(
        group_stop_trigger, features, fps, max_extension_seconds=15.0
    )

    # A stationary, unattended ball is also suspicious, but is kept separate from
    # the all-player stop rule and requires a longer persistence interval.
    unattended_raw = (
        ball_stopped & np.isfinite(nearest) & (nearest > 3.2) &
        np.isfinite(median_speed_smooth) & (median_speed_smooth < 0.85)
    )
    unattended_dead = _run_filter(
        unattended_raw,
        minimum_length=max(1, int(round(max(stationary_seconds, 1.0) * fps))),
        bridge_gap=max(1, int(round(0.16 * fps))),
    )

    # 3) Missing-ball stoppage: missing data alone is not enough. Most visible
    # players must remain nearly still for an extended interval.
    missing_candidate = (
        ~ball_available & quiet_group &
        (visible_speed_count >= 14)
    )
    missing_dead = _run_filter(
        missing_candidate,
        minimum_length=max(1, int(round(missing_seconds * fps))),
        bridge_gap=max(1, int(round(0.20 * fps))),
    )

    stationary_dead = group_stop_dead | unattended_dead
    dead = outside_dead | stationary_dead | missing_dead
    details = {
        "outside_raw": outside_raw,
        "outside_dead": outside_dead,
        "group_stop_dead": group_stop_dead,
        "unattended_dead": unattended_dead,
        "missing_dead": missing_dead,
        "quiet_group": quiet_group,
        "ball_stopped": ball_stopped,
        "slow_fraction_smooth": slow_fraction_smooth,
        "median_speed_smooth": median_speed_smooth,
        "p80_speed_smooth": p80_speed_smooth,
        "dead": dead,
    }
    return details if return_details else dead

def classify_segment(features, fps, smooth_radius, min_score_gap, min_relative_gap,
                     transition_confirm, min_spell_frames, stationary_seconds,
                     missing_seconds, tag_bias=None, tag_dead_mask=None,
                     dead_smooth_seconds=0.40, quiet_fraction=0.78,
                     quiet_player_speed=0.45, quiet_p80_speed=0.90,
                     quiet_median_speed=0.38, return_diagnostics=False):
    """Produce H/A/D/X labels for one half, without looking across the break."""
    ball_valid = features["ball_available"]
    weights = features["ball_weight"]
    home_smoothed, observations = centred_weighted_mean(
        features["home_score"], weights, smooth_radius
    )
    away_smoothed, _ = centred_weighted_mean(
        features["away_score"], weights, smooth_radius
    )

    if tag_bias is None:
        tag_bias = np.zeros(len(ball_valid), dtype=np.float32)
    if tag_dead_mask is None:
        tag_dead_mask = np.zeros(len(ball_valid), dtype=bool)
    # Smart Tagging nudges close calls; it cannot create a team prediction by itself.
    score_gap = home_smoothed - away_smoothed + tag_bias
    relative_gap = np.abs(score_gap) / (home_smoothed + away_smoothed + np.abs(tag_bias) + 1e-6)
    best_score = np.maximum(home_smoothed, away_smoothed)
    enough_observations = observations >= max(2, int(0.12 * (2 * smooth_radius + 1)))

    candidate = np.full(len(ball_valid), UNKNOWN, dtype=np.int8)
    confident = (
        enough_observations & (best_score >= 0.20) &
        (np.abs(score_gap) >= min_score_gap) & (relative_gap >= min_relative_gap)
    )
    candidate[confident & (score_gap > 0)] = HOME
    candidate[confident & (score_gap < 0)] = AWAY

    dead_evidence = make_dead_mask(
        features, fps, stationary_seconds, missing_seconds,
        dead_smooth_seconds=dead_smooth_seconds,
        quiet_fraction=quiet_fraction,
        quiet_player_speed=quiet_player_speed,
        quiet_p80_speed=quiet_p80_speed,
        quiet_median_speed=quiet_median_speed,
        return_details=True,
    )
    automatic_dead = dead_evidence["dead"]
    # Key V4 change: do NOT suppress strong group-motion stoppage evidence just
    # because one player is close to the ball. A single player can remain active
    # while most players have stopped after a whistle or ball-out event.
    dead = automatic_dead | tag_dead_mask

    # Hysteresis: confirm an opposing team over several frames, then back-date the
    # switch to the beginning of the confirmed run. During an uncertain gap, X is
    # intentional: the evaluator forward-fills it with the preceding state.
    out = np.full(len(candidate), UNKNOWN, dtype=np.int8)
    current = UNKNOWN
    pending = UNKNOWN
    pending_start = -1
    pending_count = 0

    for index in range(len(candidate)):
        if dead[index]:
            out[index] = DEAD
            current = UNKNOWN
            pending = UNKNOWN
            pending_start = -1
            pending_count = 0
            continue

        proposed = int(candidate[index])
        if proposed == UNKNOWN:
            out[index] = UNKNOWN
            pending = UNKNOWN
            pending_start = -1
            pending_count = 0
            continue

        if proposed == current:
            out[index] = current
            pending = UNKNOWN
            pending_start = -1
            pending_count = 0
            continue

        if proposed != pending:
            pending = proposed
            pending_start = index
            pending_count = 1
        else:
            pending_count += 1

        if pending_count >= transition_confirm:
            out[pending_start:index + 1] = pending
            current = pending
            pending = UNKNOWN
            pending_start = -1
            pending_count = 0
        else:
            out[index] = UNKNOWN

    out = _remove_short_islands(out, min_spell_frames)
    if return_diagnostics:
        diagnostics = {
            "auto_dead_frames": int(automatic_dead.sum()),
            "tag_dead_frames": int(np.asarray(tag_dead_mask, dtype=bool).sum()),
            "dead_overrode_team_candidate_frames": int((automatic_dead & (candidate != UNKNOWN)).sum()),
            "outside_frames": int(dead_evidence["outside_dead"].sum()),
            "group_stop_frames": int(dead_evidence["group_stop_dead"].sum()),
            "unattended_ball_frames": int(dead_evidence["unattended_dead"].sum()),
            "missing_ball_quiet_frames": int(dead_evidence["missing_dead"].sum()),
            "quiet_group_frames": int(dead_evidence["quiet_group"].sum()),
        }
        return out, diagnostics
    return out



# Smart Tagging is deliberately used as weak evidence, never copied directly as labels.
# The values below are conservative weights added to the tactical team-evidence gap.
TAG_TEAM_WEIGHTS = {
    # Offensive phase: tagged team is weakly more likely to have possession.
    "BUILD UP": 0.10,
    "PROGRESSION": 0.10,
    "ATTACKING TRANSITION": 0.10,
    "BALL IN FINAL THIRD": 0.09,
    "BALL IN THE BOX": 0.09,
    "CREATING CHANCES": 0.09,
    "LONG BALL": 0.07,
    "PLAYERS IN THE BOX": 0.06,
    # Defensive phase: tagged team is probably defending, so the *opponent*
    # receives a small prior. These weights are intentionally weaker.
    "DEFENDING IN DEFENSIVE THIRD": -0.075,
    "DEFENDING IN MIDDLE THIRD": -0.060,
    "DEFENDING IN ATTACKING THIRD": -0.045,
    "DEFENSIVE TRANSITION": -0.040,
}


def load_smart_tagging(path, fps):
    """Load SportsCode intervals using the same standard-library schema as the repo loader."""
    events = []
    root = ET.parse(path).getroot()
    for instance in root.iter("instance"):
        start_text = instance.findtext("start")
        end_text = instance.findtext("end")
        if start_text is None or end_text is None:
            continue
        try:
            start_s = float(start_text)
            end_s = float(end_text)
        except (TypeError, ValueError):
            continue
        tags = {}
        for label in instance.findall("label"):
            group = label.findtext("group")
            value = label.findtext("text")
            if group:
                tags[group.strip()] = (value or "").strip()
        events.append({
            "code": (instance.findtext("code") or "").strip(),
            "start_frame": int(round(start_s * fps)),
            "end_frame": int(round(end_s * fps)),
            "tags": tags,
        })
    events.sort(key=lambda e: (e["start_frame"], e["end_frame"]))
    return events


def _team_tag_to_code(value, home_name, away_name):
    value = (value or "").strip().casefold()
    if not value or value in {"n/a", "na", "none"}:
        return UNKNOWN
    home = (home_name or "").strip().casefold()
    away = (away_name or "").strip().casefold()
    if value == home or value in {"barcelona", "fc barcelona"}:
        return HOME
    if value == away or value in {"athletic club", "atletic club", "athletic", "atletic"}:
        return AWAY
    return UNKNOWN


def _estimate_restart_frame(event, features, fps, velocity_lag=4):
    """Estimate the live restart with event-type-specific location/motion cues.

    Tag windows are phase intervals, not exact restart timestamps. We therefore
    require a movement onset and use event type only to shape the search.
    """
    segment_start = int(features.get("global_start", 0))
    a = max(0, int(event["start_frame"]) - segment_start)
    b = min(len(features["ball_available"]), int(event["end_frame"]) - segment_start)
    if b <= a:
        return None

    kind = (event.get("tags", {}).get("Type") or "").strip().casefold()
    speed = features["ball_speed"]
    nearest = features["nearest_distance"]
    xy = features["ball_xy_norm"]
    observed = features["ball_valid"]

    settings = {
        "throw-in": (0.55, 7.0, 1.10),
        "corner kick": (0.50, 7.5, 1.10),
        "goal kick": (0.55, 8.0, 1.15),
        "free kick": (0.60, 7.0, 1.00),
        "kick off (after goal)": (0.45, 9.0, 1.20),
        "kick off (start)": (0.45, 9.0, 1.20),
    }
    min_speed, max_nearest, prior_limit = settings.get(kind, (0.65, 6.0, 0.95))
    min_future = max(2, int(round(0.12 * fps)))
    max_future = max(min_future, int(round(0.32 * fps)))

    # Type-specific geometric cues in normalized pitch coordinates.
    x, y = xy[:, 0], xy[:, 1]
    touchline_distance_m = np.minimum(y, 1.0 - y) * 68.0
    near_touchline = touchline_distance_m <= 3.5
    corner_x = np.minimum(x, 1.0 - x) * 105.3
    corner_y = np.minimum(y, 1.0 - y) * 68.0
    near_corner = (corner_x <= 13.0) & (corner_y <= 13.0)
    goal_end_area = ((x <= 0.24) | (x >= 0.76)) & (y >= 0.15) & (y <= 0.85)
    centre_circle_area = np.hypot((x - 0.5) * 105.3, (y - 0.5) * 68.0) <= 18.0

    if kind == "throw-in":
        location_ok = near_touchline
    elif kind == "corner kick":
        location_ok = near_corner
    elif kind == "goal kick":
        location_ok = goal_end_area
    elif kind in {"kick off (start)", "kick off (after goal)"}:
        location_ok = centre_circle_area
    else:
        location_ok = np.ones(len(x), dtype=bool)

    candidates = []
    for index in range(a, max(a, b - min_future)):
        if not np.isfinite(speed[index]) or speed[index] < min_speed:
            continue
        if not np.isfinite(nearest[index]) or nearest[index] > max_nearest:
            continue
        # Prefer an actual observation at or immediately around the restart. This
        # allows one interpolated frame, but not a restart inferred only from a gap.
        local_a = max(a, index - 2)
        local_b = min(b, index + 3)
        if not observed[local_a:local_b].any():
            continue
        future = speed[index:min(b, index + max_future)]
        sustained = np.isfinite(future) & (future >= max(0.35, min_speed * 0.70))
        if int(sustained.sum()) < 2:
            continue
        prior = speed[max(a, index - int(round(0.45 * fps))):index]
        prior = prior[np.isfinite(prior)]
        if prior.size >= 3 and float(np.median(prior)) > prior_limit:
            continue
        # Require the geometry for the strict pass. If the tag's coarse interval
        # and the tracking disagree slightly, retain a stricter motion-only fallback.
        geometry = bool(location_ok[index])
        fallback_speed = speed[index] >= min_speed * 1.35
        if geometry or fallback_speed:
            candidates.append((index, geometry, float(speed[index])))

    if not candidates:
        return None
    # Earliest plausible movement onset. Prefer geometric candidates when two are close.
    candidates.sort(key=lambda item: (item[0], not item[1], -item[2]))
    index = candidates[0][0]
    return min(b, index + velocity_lag)


def _infer_throw_in_team(features, restart_local, fps):
    """Infer the likely throw-in taker from short post-restart tracking evidence."""
    n = len(features["ball_available"])
    a = max(0, restart_local)
    b = min(n, a + max(5, int(round(0.9 * fps))))
    observed = features["ball_valid"][a:b]
    if int(observed.sum()) < 2:
        return UNKNOWN, 0.0
    home = features["home_score"][a:b][observed]
    away = features["away_score"][a:b][observed]
    difference = float(np.median(home - away))
    scale = float(np.median(home + away)) + 1e-6
    relative = difference / scale
    if abs(relative) < 0.14:
        return UNKNOWN, 0.0
    return (HOME if relative > 0 else AWAY), float(np.clip(abs(relative) * 0.12, 0.025, 0.07))

def build_smart_tag_context(events, features, segment_start, segment_end,
                            fps, home_name, away_name, velocity_lag=4):
    """Return weak team bias and explicit pre-restart D masks for one half.

    SET PIECES intervals are not treated as all-dead: only the portion before a
    detected restart is labelled D. GOALS -> Kick Off (After Goal) intervals add a
    conservative post-goal dead-ball bridge. Other possession-like phase tags add
    a small team prior, never an unconditional team label.
    """
    n = segment_end - segment_start + 1
    team_bias = np.zeros(n, dtype=np.float32)  # positive favours H, negative favours A
    tagged_dead = np.zeros(n, dtype=bool)
    restart_records = []
    goal_events = []
    after_goal_kickoffs = []
    defensive_prior_mask = np.zeros(n, dtype=bool)

    # Official challenge FAQ: the second half's graded window starts at frame
    # 110808, but kickoff is at 113742; every intervening frame is D.
    known_second_half_kickoff = 113742
    if segment_start <= known_second_half_kickoff <= segment_end:
        prefix_end = known_second_half_kickoff - segment_start
        tagged_dead[:prefix_end] = True

    def local_bounds(event):
        a = max(segment_start, int(event["start_frame"]))
        b = min(segment_end + 1, int(event["end_frame"]))
        if b <= a:
            return None
        return a - segment_start, b - segment_start

    for event in events:
        code = (event.get("code") or "").strip().upper()
        tags = event.get("tags", {})
        event["_team_code"] = _team_tag_to_code(tags.get("Team"), home_name, away_name)
        event["_restart_local"] = None

        # Possession-related phase tags are weak priors. They only nudge the
        # tracking score; they never create H/A on their own and never override D.
        if code in TAG_TEAM_WEIGHTS:
            bounds = local_bounds(event)
            team = event["_team_code"]
            if bounds is not None and team in (HOME, AWAY):
                local_a, local_b = bounds
                sign = 1.0 if team == HOME else -1.0
                team_bias[local_a:local_b] += sign * TAG_TEAM_WEIGHTS[code]
                if TAG_TEAM_WEIGHTS[code] < 0:
                    defensive_prior_mask[local_a:local_b] = True

        if code == "GOALS":
            if local_bounds(event) is not None:
                goal_events.append(event)
            continue

        if code != "SET PIECES":
            continue

        bounds = local_bounds(event)
        if bounds is None:
            continue
        kind = (tags.get("Type") or "").strip().casefold()
        if kind == "kick off (after goal)":
            after_goal_kickoffs.append(event)
        local_a, local_b = bounds
        restart_local = None

        # The two opening kickoffs are published frame anchors in the official
        # loader. Use the anchor only when it is inside the corresponding tag.
        half_name = (tags.get("Half") or "").strip().casefold()
        known_kickoff = None
        if kind == "kick off (start)":
            if half_name == "1st half":
                known_kickoff = 11185
            elif half_name == "2nd half":
                known_kickoff = 113742
        if known_kickoff is not None and event["start_frame"] <= known_kickoff <= event["end_frame"]:
            restart_local = known_kickoff - segment_start
        else:
            restart_local = _estimate_restart_frame(
                event, features, fps, velocity_lag=velocity_lag
            )

        event["_restart_local"] = restart_local
        restart_global = None if restart_local is None else segment_start + restart_local
        restart_records.append((kind or "unspecified", event["start_frame"],
                                event["end_frame"], restart_global))

        if restart_local is not None:
            # A half-open [start, restart) interval is the conservative dead portion.
            tagged_dead[local_a:min(local_b, restart_local)] = True
            team = event["_team_code"]
            if team in (HOME, AWAY) and restart_local < local_b:
                sign = 1.0 if team == HOME else -1.0
                # The tagged team is a restart/phase hint, not a hard label.
                team_bias[restart_local:min(local_b, restart_local + int(round(1.2 * fps)))] += sign * 0.085
            elif kind == "throw-in" and restart_local < local_b:
                inferred_team, inferred_weight = _infer_throw_in_team(features, restart_local, fps)
                if inferred_team in (HOME, AWAY) and inferred_weight > 0:
                    sign = 1.0 if inferred_team == HOME else -1.0
                    team_bias[restart_local:min(local_b, restart_local + int(round(0.8 * fps)))] += sign * inferred_weight

    # A goal is followed by a dead-ball stretch until the next after-goal kickoff.
    # Start at the end of the GOALS annotation (not its start) to avoid labelling the
    # scoring action itself as dead. Stop at the detected restart inside the kickoff tag.
    used_kickoffs = set()
    for goal in sorted(goal_events, key=lambda e: e["start_frame"]):
        candidates = [
            (idx, kickoff) for idx, kickoff in enumerate(after_goal_kickoffs)
            if idx not in used_kickoffs and kickoff["start_frame"] >= goal["start_frame"]
        ]
        if not candidates:
            continue
        idx, kickoff = min(candidates, key=lambda item: item[1]["start_frame"])
        restart_local = kickoff.get("_restart_local")
        if restart_local is None:
            continue
        used_kickoffs.add(idx)
        dead_global_start = int(goal["end_frame"])
        dead_global_end = segment_start + int(restart_local)
        a = max(segment_start, dead_global_start)
        b = min(segment_end + 1, dead_global_end)
        if b > a:
            tagged_dead[a - segment_start:b - segment_start] = True

    # Conflicting overlapping phase tags cancel; clip so tags cannot overpower tracking.
    np.clip(team_bias, -0.16, 0.16, out=team_bias)
    restart_type_counts = {}
    for kind_name, _, _, restart_frame in restart_records:
        entry = restart_type_counts.setdefault(kind_name, {"events": 0, "detected": 0})
        entry["events"] += 1
        entry["detected"] += int(restart_frame is not None)
    diagnostics = {
        "events": len(events),
        "set_piece_events": sum(
            1 for e in events
            if (e.get("code") or "").strip().upper() == "SET PIECES"
            and local_bounds(e) is not None
        ),
        "restart_found": sum(1 for _, _, _, r in restart_records if r is not None),
        "restart_records": restart_records,
        "restart_type_counts": restart_type_counts,
        "tagged_dead_frames": int(tagged_dead.sum()),
        "team_prior_frames": int(np.count_nonzero(team_bias)),
        "defensive_prior_frames": int(np.count_nonzero(defensive_prior_mask)),
    }
    return team_bias, tagged_dead, diagnostics

def build_submission(data_dir=None, smooth_radius=25, min_score_gap=0.10,
                     min_relative_gap=0.16, transition_confirm=7,
                     min_spell_frames=18, stationary_seconds=0.85,
                     missing_seconds=1.5, use_smart_tagging=True,
                     interpolate_gap=5, use_homography=True,
                     dead_smooth_seconds=0.40, quiet_fraction=0.78,
                     quiet_player_speed=0.45, quiet_p80_speed=0.90,
                     quiet_median_speed=0.38):
    """Build a full-length label array; ungraded frames are X."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, data_dir)
    n_frames = source_frames("tactical", GAME)
    team_of_track = official_baseline.team_of_each_track(meta_path)
    home_name, away_name = official_baseline.team_names(meta_path)
    if len(team_of_track) != 46:
        print("Warning: expected 46 track slots, found {}".format(len(team_of_track)))

    print("Teams: H = {}; A = {}".format(home_name, away_name))
    print("Reading tactical ATD: {}".format(os.path.basename(raw_path)))
    player_xy, ball_xy_observed_array = load_tracking(raw_path, n_frames, len(team_of_track))
    pitch = GAMES[GAME]["pitch"]
    ball_xy, ball_interpolated, ball_observed = interpolate_short_ball_gaps(
        ball_xy_observed_array, pitch, max_gap_frames=interpolate_gap
    )
    print("  short ball gaps interpolated: {:,} frames (max gap {} frames)".format(
        int(ball_interpolated.sum()), interpolate_gap
    ))

    homography = None
    if use_homography:
        try:
            homography_path = require(["homography"], GAME, data_dir)[0]
            homography = load_homography(homography_path, n_frames)
        except (OSError, IOError, ValueError) as error:
            print("  homography unavailable ({}); continuing without projection checks".format(error))

    tag_events = []
    if use_smart_tagging:
        try:
            tag_path = require(["smart_tagging"], GAME, data_dir)[0]
            tag_events = load_smart_tagging(tag_path, GAMES[GAME]["fps"])
            print("  Smart Tagging events loaded: {:,}".format(len(tag_events)))
        except (OSError, IOError, ValueError) as error:
            print("  Smart Tagging unavailable ({}); continuing without tag evidence".format(error))

    labels = np.full(n_frames, "X", dtype="<U1")
    fps = GAMES[GAME]["fps"]

    for half_number, first_frame, last_frame in half_video_blocks(GAME):
        print("\nProcessing half {} (video frames {:,}-{:,})...".format(
            half_number, first_frame, last_frame
        ))
        # Slice first, so rolling operations and temporal state cannot cross halftime.
        p_segment = player_xy[first_frame:last_frame + 1]
        b_segment = ball_xy[first_frame:last_frame + 1]
        segment_homography = None
        if homography is not None:
            segment_homography = {
                "matrices": homography["matrices"][first_frame:last_frame + 1],
                "error": homography["error"][first_frame:last_frame + 1],
                "valid": homography["valid"][first_frame:last_frame + 1],
            }
        features = compute_features(
            p_segment, b_segment, team_of_track, pitch, fps, velocity_lag=4,
            ball_observed=ball_observed[first_frame:last_frame + 1],
            ball_interpolated=ball_interpolated[first_frame:last_frame + 1],
            homography=segment_homography,
        )
        features["global_start"] = first_frame
        tag_bias, tag_dead_mask, tag_report = build_smart_tag_context(
            tag_events, features, first_frame, last_frame, fps,
            home_name, away_name, velocity_lag=4
        ) if tag_events else (
            np.zeros(len(b_segment), dtype=np.float32),
            np.zeros(len(b_segment), dtype=bool),
            {"events": 0, "set_piece_events": 0, "restart_found": 0,
             "tagged_dead_frames": 0, "team_prior_frames": 0, "defensive_prior_frames": 0,
             "restart_records": [], "restart_type_counts": {}}
        )
        print("  Smart Tags: {}/{} set-piece restarts detected; {:,} pre-restart/goal D frames; {:,} team-prior frames".format(
            tag_report["restart_found"], tag_report["set_piece_events"],
            tag_report["tagged_dead_frames"], tag_report["team_prior_frames"]
        ))
        if tag_report.get("restart_type_counts"):
            print("  restart detection by type: {}".format(tag_report["restart_type_counts"]))
        if segment_homography is not None:
            errors = segment_homography["error"][segment_homography["valid"]]
            if errors.size:
                print("  homography median error: {:.3f}px; low-quality frames (>3px): {:,}; extreme player tracks rejected: {:,}".format(
                    float(np.median(errors)), int(np.sum(errors > 3.0)),
                    features["projected_player_outlier_count"]
                ))
        if tag_report.get("defensive_prior_frames", 0):
            print("  defensive-tag prior frames: {:,}".format(tag_report["defensive_prior_frames"]))
        segment_codes, dead_report = classify_segment(
            features=features,
            fps=fps,
            smooth_radius=smooth_radius,
            min_score_gap=min_score_gap,
            min_relative_gap=min_relative_gap,
            transition_confirm=transition_confirm,
            min_spell_frames=min_spell_frames,
            stationary_seconds=stationary_seconds,
            missing_seconds=missing_seconds,
            tag_bias=tag_bias,
            tag_dead_mask=tag_dead_mask,
            dead_smooth_seconds=dead_smooth_seconds,
            quiet_fraction=quiet_fraction,
            quiet_player_speed=quiet_player_speed,
            quiet_p80_speed=quiet_p80_speed,
            quiet_median_speed=quiet_median_speed,
            return_diagnostics=True,
        )
        print("  dead-ball evidence: {auto_dead_frames:,} automatic D; {tag_dead_frames:,} Smart-Tag D; "
              "{dead_overrode_team_candidate_frames:,} frames where motion-based D overrode H/A".format(**dead_report))
        print("    reasons: outside={outside_frames:,}; group stopped={group_stop_frames:,}; "
              "unattended ball={unattended_ball_frames:,}; missing+quiet={missing_ball_quiet_frames:,}".format(**dead_report))
        print("    quiet-group frames={:,}; group rule: >= {:.0f}% players below {:.2f} m/s, "
              "median <= {:.2f}, p80 <= {:.2f}".format(
                  dead_report.get("quiet_group_frames", 0),
                  100 * quiet_fraction, quiet_player_speed, quiet_median_speed, quiet_p80_speed
              ))
        labels[first_frame:last_frame + 1] = np.array(
            [CODE_TO_LABEL[int(code)] for code in segment_codes], dtype="<U1"
        )
        print("  labels: {}".format({
            label: int(np.sum(labels[first_frame:last_frame + 1] == label))
            for label in ("H", "A", "D", "X")
        }))

    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None,
                        help="optional directory containing tactical/; defaults to MTAH_DATA_DIR or ./data")
    parser.add_argument("--smooth-radius", type=int, default=25,
                        help="centred evidence window radius in frames (default: 25; 1 second either side at 25 fps)")
    parser.add_argument("--min-score-gap", type=float, default=0.10,
                        help="minimum absolute difference between smoothed team evidence")
    parser.add_argument("--min-relative-gap", type=float, default=0.16,
                        help="minimum relative evidence margin between the teams")
    parser.add_argument("--transition-confirm", type=int, default=7,
                        help="consecutive confident frames needed to confirm a team change")
    parser.add_argument("--min-spell-frames", type=int, default=18,
                        help="merge only shorter interior H/A islands when both neighbours agree")
    parser.add_argument("--stationary-seconds", type=float, default=0.85,
                        help="minimum stationary-ball + whole-pitch-quiet duration before assigning D")
    parser.add_argument("--missing-seconds", type=float, default=1.5,
                        help="minimum quiet missing-ball duration before assigning D")
    parser.add_argument("--dead-smooth-seconds", type=float, default=0.40,
                        help="half-width of centred smoothing for ball/team motion dead-ball evidence")
    parser.add_argument("--quiet-fraction", type=float, default=0.78,
                        help="required fraction of visible players moving below quiet-player-speed")
    parser.add_argument("--quiet-player-speed", type=float, default=0.45,
                        help="speed below which a player counts as nearly stationary (m/s)")
    parser.add_argument("--quiet-p80-speed", type=float, default=0.90,
                        help="maximum 80th-percentile player speed for a whole-pitch stoppage")
    parser.add_argument("--quiet-median-speed", type=float, default=0.38,
                        help="maximum median player speed for a whole-pitch stoppage")
    parser.add_argument("--interpolate-gap", type=int, default=5,
                        help="maximum missing-ball gap to linearly interpolate (default: 5 frames)")
    parser.add_argument("--no-homography", action="store_true",
                        help="disable homography confidence and projection checks")
    parser.add_argument("--no-smart-tagging", action="store_true",
                        help="disable Smart Tagging evidence (tracking-only ablation)")
    parser.add_argument("--out", default=os.path.join(
        ROOT, "evaluation", "results", "rule_based_tactical.csv"
    ), help="submission CSV output path")
    args = parser.parse_args()

    if args.smooth_radius < 0 or args.transition_confirm < 1 or args.min_spell_frames < 0:
        parser.error("smooth radius/min spell must be non-negative and transition confirmation >= 1")
    if args.min_score_gap < 0 or args.min_relative_gap < 0:
        parser.error("score gap thresholds must be non-negative")
    if args.stationary_seconds <= 0 or args.missing_seconds <= 0 or args.dead_smooth_seconds < 0:
        parser.error("dead-ball durations must be positive and smoothing non-negative")
    if not 0.0 < args.quiet_fraction <= 1.0:
        parser.error("quiet-fraction must be in (0, 1]")
    if min(args.quiet_player_speed, args.quiet_p80_speed, args.quiet_median_speed) <= 0:
        parser.error("quiet speed thresholds must be positive")
    if args.interpolate_gap < 0 or args.interpolate_gap > 10:
        parser.error("interpolate-gap must be between 0 and 10 frames")

    labels = build_submission(
        data_dir=args.data_dir,
        smooth_radius=args.smooth_radius,
        min_score_gap=args.min_score_gap,
        min_relative_gap=args.min_relative_gap,
        transition_confirm=args.transition_confirm,
        min_spell_frames=args.min_spell_frames,
        stationary_seconds=args.stationary_seconds,
        missing_seconds=args.missing_seconds,
        use_smart_tagging=not args.no_smart_tagging,
        interpolate_gap=args.interpolate_gap,
        use_homography=not args.no_homography,
        dead_smooth_seconds=args.dead_smooth_seconds,
        quiet_fraction=args.quiet_fraction,
        quiet_player_speed=args.quiet_player_speed,
        quiet_p80_speed=args.quiet_p80_speed,
        quiet_median_speed=args.quiet_median_speed,
    )

    output_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(output_dir, exist_ok=True)
    official_baseline.write_submission(args.out, labels)

    print("\nWrote: {}".format(os.path.abspath(args.out)))
    print("Rows: {:,}".format(len(labels)))
    print("Label counts: {}".format({
        label: int(np.sum(labels == label)) for label in ("H", "A", "D", "X")
    }))
    print("\nValidate with:")
    print("  python evaluation/validate_submission.py {}".format(args.out))


if __name__ == "__main__":
    main()
