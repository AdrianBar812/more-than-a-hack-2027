"""Per-frame H/A/D labels from DFL ball status + possession, with a tunable debounce.

DFL names the possessing team on every touch, so raw spells are choppy (median ~3 s). A "controlled
possession" proxy (CHALLENGE.md) is that a team keeps the ball for a few seconds, so H/A spells
shorter than `min_spell_seconds` are merged into a neighbour. N is a hyperparameter: see
`sweep()` for the spell count / duration at several values.

Codes: 0 = H, 1 = A, 2 = D (same order as starter/tested_models/HMM.py).
"""

import numpy as np

FPS = 25
H, A, D = 0, 1, 2


def raw_labels(ball_status, ball_poss):
    """alive -> possessing team (0 home / 1 away), else D. Missing ball frames count as dead."""
    lab = np.where(ball_status == 1, ball_poss.astype(np.int16) - 1, D)
    return np.clip(lab, 0, 2).astype(np.int8)


def _runs(lab):
    cut = np.flatnonzero(np.diff(lab) != 0) + 1
    starts = np.concatenate(([0], cut))
    ends = np.concatenate((cut, [len(lab)]))
    return [[int(lab[s]), int(s), int(e)] for s, e in zip(starts, ends)]


def debounce(lab, min_spell_seconds):
    """Merge every H/A spell shorter than N s into a neighbour. Dead-ball runs are never changed.

    Repeatedly takes the shortest too-short spell and gives it to
      * the longer of its two neighbours when both are live possessions,
      * the single live neighbour when the other side is dead ball,
      * nobody when it sits between two dead-ball runs (a genuine short possession).
    Merging can create a new short spell, so it loops until stable.
    """
    n = int(round(min_spell_seconds * FPS))
    if n <= 1:
        return lab.copy()
    runs = _runs(lab)
    while True:
        short = [(r[2] - r[1], i) for i, r in enumerate(runs) if r[0] != D and r[2] - r[1] < n]
        target = None
        for length, i in sorted(short):
            left = runs[i - 1] if i > 0 else None
            right = runs[i + 1] if i + 1 < len(runs) else None
            live = [x for x in (left, right) if x is not None and x[0] != D]
            if not live:
                continue
            target = (i, max(live, key=lambda x: x[2] - x[1])[0])
            break
        if target is None:
            break
        i, team = target
        runs[i][0] = team
        merged = [runs[0]]
        for r in runs[1:]:
            if r[0] == merged[-1][0]:
                merged[-1][2] = r[2]
            else:
                merged.append(r)
        runs = merged
    out = np.empty_like(lab)
    for team, s, e in runs:
        out[s:e] = team
    return out


def spells(lab):
    """Durations (s) of H/A spells."""
    return np.array([(e - s) / FPS for t, s, e in _runs(lab) if t != D])


def sweep(lab, values=(0, 1, 2, 3, 5, 8)):
    rows = []
    for v in values:
        d = spells(debounce(lab, v))
        rows.append((v, len(d), float(d.mean()) if len(d) else 0.0, float(np.median(d)) if len(d) else 0.0))
    return rows
