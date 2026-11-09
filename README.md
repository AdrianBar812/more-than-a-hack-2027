# Team Notes — Ball Possession Models

## Ideas of models

| Approach | Needs labels from other games? | Handles missing ball | spellF1 expectation | Main risk |
|---|---|---|---|---|
| **Heuristics + HMM/Viterbi** | No | Moderate, if you interpolate and use X | Likely above the baseline | Transition penalty needs choosing without tuning on the public window |
| **LightGBM on window features + smoothing** | Yes | Good, since it can learn missingness | Potentially higher | The label source and format of other games |
| **GRU / 1D-CNN / transformer + smoothing** | Yes, more of it | Good | Potentially highest | Needs the most data and tuning, and is harder to reproduce |

---

## Important notes about files

### `tactical_adt.txt`

- Each row has multiple tracks that represent the position of a player or the ball.
- Each track has 3 values: `<position_x>:<position_y>:<speed>`.
- Tracks are not tied to one player (if the algorithm loses a player it can assign a different track to the same player once they are tracked again).
- All track values are already normalised to field position.
- Homography is already applied.

### `tactical_adt.xml`

Has more information about the tracks:

- There are 44 possible tracks.
- Each track can represent different players, **but always the same team**.
- ❓ Attributes when the game starts / half time / end (not sure).
- ❓ I think goalkeepers have a different attribute.

### `tactical_homography.csv`

- Each row represents a 3x3 matrix with homography information.
- Kinda useless because the transformations are already applied.
- Might be useful as a reference if we plan to use any additional datasets.

### `tactical_smart_tagging.xml`

- Has attributes that give more context to what is happening in the game (free kicks, penalties, etc.).
- Might be useful for special cases.

---

## HMM

Statistical and probabilistic state model.

- Already implemented.
- Scores worse than the basic model they already provided 😅
- It flags all the air balls as dead balls (probably the main issue).
- Parameters that can be tuned are listed below.

### Transitions: how sticky each state is

| Parameter | Default | Effect |
|---|---|---|
| `SECONDS_BETWEEN_SWITCHES` (`--switch-seconds`) | 12 s | Expected time between H↔A changes. Higher means longer possessions and fewer splits, but slower reaction to real turnovers. |
| `SECONDS_BETWEEN_STOPS` | 40 s | Live→dead. Higher means D is entered less readily. |
| `SECONDS_OF_DEAD_BALL` | 10 s | Dead→live. Lower ends D runs sooner. |

### Evidence for H vs A

| Parameter | Default | Effect |
|---|---|---|
| `CLOSER_SLOPE` | 0.6 per m | Confidence per metre of "who is closer". Higher means the evidence outweighs the transition penalty more often (more flicker). |
| `LOOSE_BALL_DISTANCE` | 6 m | How fast the evidence fades when nobody is near the ball. Lower means a ball with nobody nearby counts for less. |

### Evidence for D

| Parameter | Default | Effect |
|---|---|---|
| `STILL_SPEED` / `STILL_FRAMES` | 0.7 m/s / 25 | What counts as a still ball, and for how long. |
| `OFF_PITCH_MARGIN` | 0.01 | How far outside 0–1 counts as out of play. |
| `D_LOGLIK_STRONG` / `_STILL` / `_PLAY` | 3.0 / 1.5 / −0.5 | Strength of the D cues. `_PLAY` is the penalty for D when the ball is moving on the pitch. |

### Mid-air → X

| Parameter | Default | Effect |
|---|---|---|
| `SPEED_LAG_FRAMES` | 5 | Window for computing ball speed. Larger is smoother but blurs short events. |
| `FLIGHT_SPEED` / `FLIGHT_DISTANCE` | 10 m/s / 3 m | What looks airborne. Lower values turn more D frames into X. |

---

## To dos

- [ ] Look for any other available datasets of football matches that have data labelled by frame (human labelled if possible).
- [ ] Try other models.
- [ ] Try to improve the amount of dead balls the HMM generates (idk how atm).
