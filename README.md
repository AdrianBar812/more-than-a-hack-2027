# General info
## Data
* You will have to download the data from the google disk manually and add it in the `data/`
* Make you sure you separate the `tactical/` and `broadcast/`. The `games.py` expects to see at least one of them in the `data/`
* Extra training data (other games): the DFL/IDSSE Bundesliga dataset, see [Additional data](#additional-data-dfl--idsse) below. It is **not** in this repo.

## How to run
1. Run the model `python starter/tested_models/HMM.py *--switch-seconds <#sec> *--out <out_path>`
   or with the tuned parameters: `python starter/tested_models/HMM.py --params pipeline/hmm_params_v2.json --name my_run --out <out_path>`
2. Validate `python evaluation/validate_submission.py path/to/my_output.csv`
3. Evaluate `python evaluation/run_evaluation.py *--results path/to/my_folder *--out path/to/reports_dir`
   (run it without `python -I`: it imports a sibling module)

with *-- are flagged the optional arguments that you can pass

Dependencies: `requirements.txt` (scorer), `requirements-starter.txt` (starter/notebooks), `requirements-pipeline.txt` (pinned versions for `pipeline/`).

### Pipeline (DFL training data, degradation, HMM tuning)
```bash
python pipeline/ingest_dfl.py --data-dir <DFL data dir>     # 7 matches -> data/pipeline/dfl_raw/*.npz   (~70 s)
python pipeline/degrade.py --seeds 3                        # fit missing-ball model on the ATD, write data/pipeline/dfl_canon/*.npz
python pipeline/check_domain.py                             # adversarial check: degraded DFL vs the ATD
python pipeline/tune_hmm.py --random 200 --refine 300 --seed 1 --init <previous json> --out <json>
```
Everything under `data/` is gitignored. The tuned HMM parameters that we ship are in `pipeline/hmm_params_v2.json`.

### LightGBM: train, decode, and test on the public window
Install the pinned versions first: `pip install -r requirements-pipeline.txt` (numpy, pandas, scipy, lightgbm; no scikit-learn needed).

```bash
# 1. Features for every degraded DFL match/seed and for the ATD (~1.7 GB cache in data/pipeline/features/, a few minutes)
python pipeline/features.py --build

# 2. Leave-one-match-out on the 7 DFL matches, decoder search, final model on all 7, decode the ATD
python pipeline/train_lgb.py --ref-seconds 3 --out-dir data/pipeline/lgb_v2 --costs 16 20
#    --ref-seconds 3  debounce N used for the training labels
#    --costs SW D     fix the Viterbi costs (H<->A, live<->D). Without it, the best pair on the out-of-fold DFL is used
#    --merge-seconds  after Viterbi, merge H/A spells shorter than this into a neighbour (default 3; 0 = off)
#    --ensemble K     average K models (no gain measured)
#    prints: raw-argmax and decoded out-of-fold spellF1, the top decoder settings, the top features
#    writes: data/pipeline/lgb_v2/{lgb_decoded.csv, atd_probs.npz, oof.npz, model.txt, decoder.json}

# 3. Validate the submission file (format only)
python evaluation/validate_submission.py data/pipeline/lgb_v2/lgb_decoded.csv

# 4. Test on the public ground-truth window (5 minutes, a pipeline check, never for tuning)
mkdir -p data/pipeline/lgb_v2_eval && cp data/pipeline/lgb_v2/lgb_decoded.csv data/pipeline/lgb_v2_eval/lgb_v2.csv
python evaluation/run_evaluation.py --results data/pipeline/lgb_v2_eval --out data/pipeline/lgb_v2_eval/report --bootstrap 1000
```
Quick alternative for step 4: add `--public-check` to step 2 and it prints spellF1, macroF1 and per-class F1 for the raw and decoded output (the same scorer). Run the scorer without `python -I` (it imports a sibling module). All seeds are fixed. A fresh `train_lgb.py --ref-seconds 3 --costs 16 20` run from the cached features (4.5 min, 20 threads) reproduced `lgb_v2_sw16_d20_merge3.csv` byte for byte (checked 2026-11-17; the features and degradation steps were not re-run). The public window is not read unless you ask for it.

Rules for using the public window: it is 5 minutes with 17 possessions (about 6 `spellF1` points per missed possession, noise about ±0.04), so use it only to confirm the pipeline works. Do not choose parameters from it. Pick decoder costs and N from the out-of-fold DFL numbers.

</br>

# Team Notes — Ball Possession Models

## Ideas of models

| Approach | Needs labels from other games? | Handles missing ball | spellF1 expectation | Main risk |
|---|---|---|---|---|
| **Heuristics + HMM/Viterbi** | No (but tuned on DFL) | Moderate, if you interpolate and use X | Above the baseline (done, see results) | Dead ball is poorly detected; parameters are weakly identified |
| **LightGBM on window features + Viterbi** | Yes (DFL) | Good, since it can learn missingness | Done: current best (see results) | Domain gap DFL→ATD (adversarial AUC 0.98); dead-ball recall on the real match about 0.6 |
| **GRU / 1D-CNN / transformer + smoothing** | Yes, more of it | Good | Potentially highest | Needs the most data and tuning, and is harder to reproduce |

## Current results

`spellF1` / `macroF1` on the **public 5 minutes** (17 possessions, noise band ≳ ±0.04, a sanity check only, never used to pick parameters):

| Model | spellF1 | macroF1 | predicted possessions (17 true) |
|---|---|---|---|
| `baseline_smooth50` (published floor, full match 0.2433) | 0.326 | 0.621 | 26 |
| HMM, old defaults (`hmm_switch16.9`) | 0.231 | 0.641 | 47 |
| HMM, tuned on degraded DFL (`pipeline/hmm_params_v2.json`) | 0.490 | 0.630 | 14 |
| LightGBM v1, decoded (switch 48 / dead 12) | 0.607 | 0.817 | 14 |
| LightGBM v2, decoded (switch 16 / dead 48) | 0.492 | 0.842 | 9 |
| LightGBM v2, decoded (switch 16 / dead 20) | 0.613 | 0.825 | 13 |
| **LightGBM v2, decoded 16 / 20 + merge H/A spells under 3 s, current candidate** | **0.635** | 0.832 | 12 |

On held-out DFL matches (degraded to look like the ATD): defaults 0.204, tuned 0.542, smooth50 0.279.
LightGBM, leave-one-match-out on all 7 DFL matches (decoder 16/20): raw argmax 0.317, decoded 0.686 (per match 0.589 to 0.737, std 0.043), D recall 0.97; with the 3 s merge 0.6945 (+0.0085 on all 7, +0.006 when the setting is chosen on the other 6 matches). Short possessions are the weak point: of true spells under 5 s only about a third are matched.
The switch 16 / dead 20 setting was chosen after seeing the v2 public-window result, so it is a disclosed judgement call and not independent evidence (see `CONTEXT.md`).
The untuned HMM over-segments about 2.5× on both degraded DFL and the real ATD, so the degraded DFL is a reasonable proxy.

---

## Important notes about files

### `tactical_adt.txt`

- Each row has multiple tracks that represent the position of a player or the ball.
- Each track has 3 values: `<position_x>:<position_y>:<speed>`.
- Tracks are not tied to one player (if the algorithm loses a player it can assign a different track to the same player once they are tracked again).
- All track values are already normalised to field position.
- Homography is already applied.
- **The ball speed (3rd value) is unusable**: present on only 7.8% of visible frames and capped at ~7 m/s. Always compute speed from positions (`HMM.py` and `pipeline/` do).
- **The ball is missing in long blocks, not frame by frame**: ~150 gaps in the graded halves, median 10.7 s, mean 17 s. Gaps cluster near the goal lines/touchlines and while players move slowly (stoppages), so a missing ball is itself a clue. A player-only model predicts "ball missing" with AUC 0.82–0.85.

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

Statistical and probabilistic state model (`starter/tested_models/HMM.py`).

- Implemented, and now tuned (see [HMM tuning](#hmm-tuning)).
- With its original constants it scored below the basic model they already provided (spellF1 0.231 vs 0.326 on the public window); tuned it scores 0.490.
- Known weakness: **dead ball**. On the public window it recalls only ~25% of true dead-ball frames (most are labelled as a team in possession).
- Parameters that can be tuned are listed below. "Code default" is what `HMM.py` has as constants; "Tuned v2" is `pipeline/hmm_params_v2.json`.

### Transitions: how sticky each state is

| Parameter | Code default | Tuned v2 | Effect |
|---|---|---|---|
| `SECONDS_BETWEEN_SWITCHES` (`--switch-seconds`) | 16.9 s | 133.3 s | Expected time between H↔A changes. Higher means longer possessions and fewer splits, but slower reaction to real turnovers. |
| `SECONDS_BETWEEN_STOPS` | 36.2 s | 32.9 s | Live→dead. Higher means D is entered less readily. |
| `SECONDS_OF_DEAD_BALL` | 20.1 s | 20.4 s | Dead→live. Lower ends D runs sooner. |

### Evidence for H vs A

| Parameter | Code default | Tuned v2 | Effect |
|---|---|---|---|
| `CLOSER_SLOPE` | 0.6 per m | 0.22 | Confidence per metre of "who is closer". Higher means the evidence outweighs the transition penalty more often (more flicker). |
| `LOOSE_BALL_DISTANCE` | 6 m | 1.6 m | How fast the evidence fades when nobody is near the ball. Lower means a ball with nobody nearby counts for less. |

### Evidence for D

| Parameter | Code default | Tuned v2 | Effect |
|---|---|---|---|
| `STILL_SPEED` / `STILL_FRAMES` | 0.7 m/s / 25 | 1.59 m/s / 38 | What counts as a still ball, and for how long. |
| `OFF_PITCH_MARGIN` | 0.01 | (not tuned) | How far outside 0–1 counts as out of play. |
| `D_LOGLIK_STRONG` / `_STILL` / `_PLAY` | 3.0 / 1.5 / −1.0 | 0.66 / 0.0 / −0.95 | Strength of the D cues. `_PLAY` is the penalty for D when the ball is moving on the pitch. Tuned `_STILL` = 0: the still-ball cue is switched off. |
| `D_LOGLIK_MISSING` | 0.3 | 1.11 | Lean towards D when the ball is not tracked at all. |
| `MIN_DEAD_SECONDS` (`--min-dead-seconds`) | 2 s | 6.5 s | D runs shorter than this are replaced by the previous label. |

### Mid-air → X

| Parameter | Code default | Tuned v2 | Effect |
|---|---|---|---|
| `SPEED_LAG_FRAMES` | 5 | (not tuned) | Window for computing ball speed. Larger is smoother but blurs short events. |
| `FLIGHT_SPEED` / `FLIGHT_DISTANCE` | 10 m/s / 3 m | 16.2 m/s / 2.1 m, **off** | What looks airborne. The search switched the whole mid-air→X step off (`--keep-flight-as-d`). |
| `AMBIGUOUS_LLR` | 0.3 | 0.35, **off** | Coin-flip team evidence → X. Also switched off (`--keep-ambiguous-as-d`). |

### HMM tuning

The old tuner (`starter/tested_models/tune_hmm_from_dfl.py`) only turned DFL event-duration means into the three transition times. The new one (`pipeline/tune_hmm.py`) runs the real HMM on **degraded DFL matches** and searches all parameters against the project's own metric (`possession_metrics.evaluate`, i.e. spellF1):

- tune on DFL matches 0–3 (2 degradation seeds); held out matches 4–6 with a fresh seed; nothing from the competition match's labels is used;
- reference labels = DFL ball status/possession, debounced so spells shorter than N = 3 s are merged (N is a parameter, chosen a priori from the possession definition, not from the public ground truth);
- trial 0 = code defaults, then random search and local refinement; round 1 (400 trials) then round 2 with widened ranges and a warm start (500 trials);
- result: held-out DFL spellF1 0.204 → 0.542; public window 0.231 → 0.490. The two rounds are within noise of each other.

Caveats: the parameters are weakly identified (several settings score about the same; the switch time drifted to 133 s, near its upper bound), and the gain is in possession structure, not in macroF1 or dead-ball detection. The HMM looks saturated; the next lever is a learned model.

---

## LightGBM

`pipeline/features.py` (142 features, one builder for DFL and the ATD), `pipeline/train_lgb.py`.

- **Features** (centred windows, offline): player-only (`p_`: centroids, spread, lines, speed, stillness at 1 / 4 / 12 s), ball where visible (`b_`, `bw_`), last-seen and next-seen ball values with their age (`bl_`, `bn_`, they bridge the 10 to 17 s ball gaps), and how much of the surrounding time the ball is visible (`v_`). Ball speed is recomputed from positions. Attack direction is inferred per half and x is mirrored, so home always attacks towards +x.
- **Training**: leave-one-match-out on the 7 degraded DFL matches (6 matches x 3 degradation seeds, every 5th frame). Multiclass H / A / D, 300 rounds, 31 leaves, lr 0.05. Native `lightgbm.train` API.
- **No thresholds**: the model outputs `P(H), P(A), P(D)` per frame and a Viterbi decoder picks the label path with the lowest `sum(-log p) + transition costs`. Switching H<->A costs 16 and live<->D costs 20 (nats), so a label changes only when the evidence over a stretch of frames outweighs the cost. Decoding takes the out-of-fold `spellF1` from 0.317 to 0.686.
- **Debounce N = 3 s** for the training labels (sweep over 2 to 5 s, scored against every reference N: N=2 0.692, N=3 0.689, N=4 0.669, N=5 0.657 mean `spellF1`).

What did not help (all recorded in `CONTEXT.md`): an ensemble of 3 models (0.690 vs 0.689), a bias on the D probability (best at 0 on DFL), wider decoder grids (a plateau for dead cost 32 to 80), and round 3 domain alignment (speed dead-band, relative-speed and set-piece features, adversarial pruning and weights: adversarial AUC 0.98 to 0.82, but public-window D recall 0.56 vs 0.61, so it is switched off by default: `degrade.dead_band=False`, `features.EXTRA=False`).

Known weakness: dead-ball recall on the real match is about 0.6 (precision about 0.9), while on DFL out-of-fold it is 0.97. The predicted D share on the full ATD (44.7%) matches the DFL truth (44.8%), so the gap is about where D is placed or how it is labelled, not how much of it there is. Next test: a held-out dataset with real broadcast missingness (SkillCorner open data is the candidate; not yet checked).

---

# Additional data (DFL / IDSSE)

Open dataset with 7 Bundesliga matches (match info, official events, 25 fps TRACAB positions incl. the ball):
[repo](https://github.com/spoho-datascience/idsse-data) · data: https://doi.org/10.6084/m9.figshare.28196177

Other games, so allowed as training/validation data under the rules. We use it to build per-frame labels and training tracking:

- **Labels**: every DFL ball frame has `BallStatus` (alive/dead) and `BallPossession` (1 home / 2 away). Label = `D` if dead, else the team. This is the same logic the competition ground truth uses.
- **Raw spells are touch-level** (median ~3 s, 3061 spells over 7 matches); `pipeline/labels.py` `debounce()` merges spells shorter than N seconds. Possessions / mean spell over the 7 matches: N=0 3061 / 7.2 s, 1 2551 / 8.7 s, 2 1964 / 11.3 s, 3 1634 / 13.5 s, 5 1281 / 17.3 s, 8 1036 / 21.4 s.
- **Degradation** (`pipeline/degrade.py`): DFL is converted to 0–1 coordinates and anonymous per-team slots, players get rare dropouts plus correlated "view loss" episodes, and the ball-missing mask is sampled from a model fitted on the *unlabelled* ATD (P(ball missing | player-only features)). Matches the ATD on ball-missing share (39% vs 41%), missing-run length (median 11.4 s vs 10.7 s), tracked players and player speed.
- **Domain gap**: a classifier still separates degraded DFL from the ATD (AUC 0.84; per-match normalisation does not change it). The tells are team compactness, ball speed and the like, probably real league/tracker differences. Treat as a known gap and validate downstream.

Layout of `pipeline/`: `ingest_dfl.py`, `labels.py`, `atd.py`, `player_features.py`, `degrade.py`, `check_domain.py`, `tune_hmm.py`, `hmm_params_v2.json`, `features.py`, `train_lgb.py`, `sweep_n.py`, `adversarial.py`. More detail in `CONTEXT.md`.

**Credit / licence**: the DFL data is CC-BY 4.0. Credit the *Deutsche Fußball Liga (DFL)* and cite: Bassek, M., Rein, R., Weber, H., Memmert, D. (2025). *An integrated dataset of spatiotemporal and event data in elite soccer.* Scientific Data, 12(1), 195. https://doi.org/10.1038/s41597-025-04505-y

---

## To dos

- [✓] Look for any other available datasets of football matches that have data labelled by frame (human labelled if possible).
- [✓] Try to improve the amount of dead balls the HMM generates (idk how atm). *Partly: tuned, but dead-ball recall is still ~25%.*
- [✓] Calibrate/tune the HMM on other games (DFL) instead of the public window.
- [✓] Canonical pipeline: DFL → ATD-like training data (ingest, labels, degradation, domain check).
- [✓] Feature builder (ball-free + ball features, attack direction inferred so it works in both halves).
- [✓] LightGBM, leave-one-match-out on DFL, scored with spellF1; then Viterbi decoding of its probabilities (HMM tuned as fallback).
- [✓] Sweep the debounce N (2–5 s) with a fixed reference; fix one value (N = 3) and document it.
- [ ] Validate on a second labelled dataset with real broadcast missingness (SkillCorner open data?) to measure dead-ball recall.
- [ ] Reproducible repo + technical document (training data: DFL, who/what/how much; possession rules; limitations).
