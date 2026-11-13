#!/usr/bin/env python3
"""Calibrate HMM transition parameters from the DFL open dataset.

Reads match information and event files from the DFL dataset to compute:
  1. Mean possession duration (H <-> A turnover time) -> SECONDS_BETWEEN_SWITCHES
  2. Mean stoppage duration (dead ball length)        -> SECONDS_OF_DEAD_BALL
  3. Mean active play duration (between stoppages)    -> SECONDS_BETWEEN_STOPS

Usage:
    python starter/tested_models/tune_hmm_from_dfl.py --data-dir /home/bark/Downloads/28196177/data
"""

import argparse
import glob
import os
import xml.etree.ElementTree as ET
from datetime import datetime


def parse_iso_time(ts_str):
    """Parse ISO timestamp with timezone offset."""
    if not ts_str:
        return None
    # Python 3.7+ fromisoformat handles +HH:MM
    return datetime.fromisoformat(ts_str)


def get_match_metadata(info_path):
    """Extract home/away team IDs from matchinformation XML."""
    tree = ET.parse(info_path)
    root = tree.getroot()
    general = root.find(".//General")
    if general is None:
        raise ValueError(f"No General element in {info_path}")
    home_id = general.attrib.get("HomeTeamId")
    guest_id = general.attrib.get("GuestTeamId")
    return home_id, guest_id


# Elements indicating a set-piece / restart where DecisionTimestamp represents stoppage start
STOPPAGE_TAGS = {"ThrowIn", "FreeKick", "CornerKick", "GoalKick", "KickOff", "PenaltyShot"}


def analyze_match_events(event_path, home_id, guest_id):
    """Parse match events to extract:
      - dead ball intervals (stoppage start to restart execution)
      - active play intervals (restart execution to next stoppage)
      - single-team possession durations
    """
    tree = ET.parse(event_path)
    root = tree.getroot()

    stoppage_durations = []
    active_durations = []
    possession_durations = []

    last_restart_time = None
    current_team = None
    possession_start_time = None

    for event in root.findall("Event"):
        event_time_str = event.attrib.get("EventTime")
        if not event_time_str:
            continue
        event_time = parse_iso_time(event_time_str)

        # Check for stoppage / set piece tags inside Event
        stoppage_found = False
        for child in event:
            tag = child.tag
            if tag in STOPPAGE_TAGS:
                stoppage_found = True
                decision_str = child.attrib.get("DecisionTimestamp")
                if decision_str:
                    stoppage_start = parse_iso_time(decision_str)
                    dt_dead = (event_time - stoppage_start).total_seconds()
                    # Filter out negative or extreme outlier intervals (e.g. half-time breaks)
                    if 0.5 <= dt_dead <= 180.0:
                        stoppage_durations.append(dt_dead)

                    # Active play between previous restart and this stoppage
                    if last_restart_time is not None:
                        dt_active = (stoppage_start - last_restart_time).total_seconds()
                        if 1.0 <= dt_active <= 300.0:
                            active_durations.append(dt_active)

                # This event execution is the restart of play
                last_restart_time = event_time
                break

        # Check team possession from Play or OtherBallAction
        team = None
        for child in event.iter():
            if "Team" in child.attrib:
                team = child.attrib["Team"]
                break

        if team in (home_id, guest_id):
            if current_team != team or stoppage_found:
                if current_team is not None and possession_start_time is not None:
                    poss_len = (event_time - possession_start_time).total_seconds()
                    if 0.5 <= poss_len <= 120.0:
                        possession_durations.append(poss_len)
                current_team = team
                possession_start_time = event_time

    # Close trailing possession
    if current_team is not None and possession_start_time is not None:
        poss_len = (event_time - possession_start_time).total_seconds()
        if 0.5 <= poss_len <= 120.0:
            possession_durations.append(poss_len)

    return stoppage_durations, active_durations, possession_durations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="/home/bark/Downloads/28196177/data",
        help="Path to folder containing DFL XML files",
    )
    args = parser.parse_args()

    info_files = sorted(glob.glob(os.path.join(args.data_dir, "*matchinformation*.xml")))
    event_files = sorted(glob.glob(os.path.join(args.data_dir, "*events_raw*.xml")))

    if not info_files or not event_files:
        print(f"Error: No XML files found in {args.data_dir}")
        return

    print(f"Found {len(event_files)} matches to process in {args.data_dir}...\n")

    all_stoppages = []
    all_actives = []
    all_possessions = []

    for event_f, info_f in zip(event_files, info_files):
        match_id = os.path.basename(event_f).split("_")[-1].replace(".xml", "")
        home_id, guest_id = get_match_metadata(info_f)
        stoppages, actives, possessions = analyze_match_events(event_f, home_id, guest_id)

        print(
            f"Match {match_id}: "
            f"{len(stoppages)} stoppages, {len(actives)} active intervals, {len(possessions)} possession spells"
        )
        all_stoppages.extend(stoppages)
        all_actives.extend(actives)
        all_possessions.extend(possessions)

    import statistics

    print("\n" + "=" * 60)
    print("EMPIRICAL CALIBRATION RESULTS ACROSS ALL MATCHES")
    print("=" * 60)

    if all_possessions:
        print(f"Possession durations (N={len(all_possessions)}):")
        print(f"  Mean:   {statistics.mean(all_possessions):.2f} s")
        print(f"  Median: {statistics.median(all_possessions):.2f} s")
        print(f"  -> Recommended SECONDS_BETWEEN_SWITCHES: {statistics.mean(all_possessions):.1f} s")
    print("-" * 60)

    if all_stoppages:
        print(f"Stoppage durations (dead ball) (N={len(all_stoppages)}):")
        print(f"  Mean:   {statistics.mean(all_stoppages):.2f} s")
        print(f"  Median: {statistics.median(all_stoppages):.2f} s")
        print(f"  -> Recommended SECONDS_OF_DEAD_BALL: {statistics.mean(all_stoppages):.1f} s")
    print("-" * 60)

    if all_actives:
        print(f"Active play intervals between stoppages (N={len(all_actives)}):")
        print(f"  Mean:   {statistics.mean(all_actives):.2f} s")
        print(f"  Median: {statistics.median(all_actives):.2f} s")
        print(f"  -> Recommended SECONDS_BETWEEN_STOPS: {statistics.mean(all_actives):.1f} s")
    print("=" * 60)

    print("\nHow to apply in HMM.py:")
    print("Update lines 76-79 in starter/tested_models/HMM.py:")
    print(f"  SECONDS_BETWEEN_SWITCHES = {statistics.mean(all_possessions):.1f}")
    print(f"  SECONDS_BETWEEN_STOPS    = {statistics.mean(all_actives):.1f}")
    print(f"  SECONDS_OF_DEAD_BALL     = {statistics.mean(all_stoppages):.1f}")


if __name__ == "__main__":
    main()
