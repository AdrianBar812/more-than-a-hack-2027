# Project Context — More than a Hack 2027 (Frame-level Ball Possession)

Snapshot written 2026-11-13. **Submission deadline: 2026-11-17** (results before 2026-11-26).
Organisers: Metrica Sports, FC Barcelona (Barça Innovation Hub), Mobile World Capital. Repo is authoritative over the overview docs; `evaluation/possession_metrics.py` is authoritative over EVALUATION.md.

## 1. The task
For every one of **199,319 frames** (25 fps) of the *tactical* video of **FC Barcelona (H, home) vs Athletic Club (A, away)**, LaLiga 24-25 MD2, output one label:

| label | meaning |
|---|---|
| `H` / `A` | home / away team in possession, ball alive |
| `D` | dead ball (not in play). Dead-ball *ownership* is NOT asked |
| `X` | abstain / no answer |

- Output: `frame,label` CSV, header exact, rows `0…199318` in order, no blanks (write `X` to abstain). Validate: `python evaluation/validate_submission.py file.csv`.
- Graded frames: 158,857 only. 1st half 11185–90651, 2nd half 110808–190197. Everything else is ignored (X is free there).
- Ground truth is derived mechanically from *corrected* tracking: `D` where ball marked dead, else the team tracking names as owner. 3 states only; no "contested".
- A **possession** = maximal run of H or of A. `H H D D H H` = two possessions. Loose/contested balls stay with the last controlling team. Touches, deflections, aerial duels without control ≠ change of possession.

## 2. Scoring (what to optimise)
- **Rank metric: `spellF1`** — possessions matched 1-to-1 by temporal IoU, F1 averaged over IoU thresholds 0.3–0.7. Constant guess = 0. 1-frame offset is free; 1 s late → 0.65. Noise band ≈ ±0.04.
- Tie-break: `macroF1` (mean F1 over H, A, D; X counts as a miss; noise ≈ 0.01).
- **X semantics:** for spellF1 the scorer forward-fills X with the last non-X label in the same half (back-fill at half start). So X inside a possession is free/safe; X across a hand-over bleeds the old label — guess at transitions.
- Published floor: `baseline_smooth50` spellF1 **0.2433** (macroF1 0.6724). Others: raw 0.0486/0.6532, smooth25 0.1996/0.6752, smooth125 0.2242/0.6356 (full match).
- **Key lesson:** per-frame metrics reward flicker-free labelling; spellF1 punishes over-segmentation. Public 5-min window picks the wrong smoothing (smooth125 looks best there, worst of the smoothed ones on the full match). **Do not tune on the public window** (17 possessions, ~6 pts per missed possession).
- Local eval: `python evaluation/run_evaluation.py [--results dir] [--out dir] [--bootstrap 1000]` — scores every CSV in `evaluation/results/` against public GT.

## 3. Rules (hard constraints)
- Labels must come from an algorithm. **No hand labels on the provided game** (not for submission, training, or validation). Watching video to understand/debug is fine; turning it into frame-specific rules/constants is not.
- **Hand-labelled/event data from OTHER games is fine in any amount** (must be accessible to organisers and documented).
- No private GT, and **no human-produced data about the provided game from any source** (event data, corrected tracking, etc.).
- Public 5 min GT (frames 11185–18684; H 3454, A 2577, D 1469, 17 possessions) = pipeline sanity check only, not training/tuning.
- Repo must regenerate the exact submission file. Data must not be committed (`data/` is gitignored); don't publish derived tracking exports.
- LLMs allowed, but **do not upload any competition data/video/extracts to third-party hosted LLMs.**
- Teams 3–5 members. Any method, any language; improving/replacing provided tracking with your own algorithms is allowed.

## 4. Deliverables (single Google Form, link emailed to participants)
1. `<team_name>.csv` possession file (~1.9 MB)
2. Public code repo (pinned deps, install + run command, no data)
3. `<team_name>.pdf` (or .md) technical doc: approach/why, assets used and dropped, **training data (source, labeller, amount — none from provided game)**, possession rules implemented, limitations, reproduction.
Needs a Google account. One response per team; last one counts. No live leaderboard.

## 5. Provided data (`data/`, gitignored)
Currently present locally: `data/tactical/` {`BAR-ATH_tactical_atd.txt`, `_atd.xml`, `_homography.csv`, `_smart_tagging.xml`}. `data/broadcast/` and the mp4 videos are **not** present locally (`data/additional_data`, `data/pipeline` are empty dirs). `games.py` resolves paths (`asset_paths()`, `half_video_blocks()`, `MTAH_DATA_DIR` env override; pitch 105.3×68.0 m).

- **ATD (tracking, Metrica CV, uncorrected):** per frame, 46 track slots (tactical; 51 broadcast) + ball. Coordinates **normalised 0–1** (multiply by 105.3 × 68 for m), speed in m/s. Raw line: `frame_id:x,y,s;x,y,s;…:ballx,bally,speed`, `NaN` = missing. `frame_id` = video_frame + 1. Slots are tracklets, not players (no IDs, can swap/vanish) but each slot always belongs to one team (first team in XML = home). ~20.5 tracked objects on average in play. **Ball missing on 41% of graded frames** (visible 59.3% graded / 56.9% all).
- **Smart Tagging (automatic, not ground truth):** 638 SportsCode intervals (seconds from video start → frame = round(s·25)). Codes: Build Up, Progression, Creating Chances, Defending in X third, Attacking/Defensive Transition, Set Pieces (59: throw-in 26, corner 12, goal kick 10, free kick 5, kick-off 6), Goals (4), Long Ball, Ball in Final Third/Box, Players in the Box. Tags: Team (`N/A` for 36 incl. *all throw-ins*), Half, Type, Side, Direction, Max players in box (string). Set pieces = restarts = ends of dead-ball stretches; phases are not possessions.
- **Homography:** 3×3 per frame mapping 0–1 pitch → pixels; tactical 100% valid; broadcast 65.8%. `starter/load_homography.py`.
- **Broadcast:** own timeline (164,488 frames), cuts/replays, **not synchronised** to tactical, no mapping provided.

## 6. Repo layout
- Docs: `README.md` (team notes — our own), `OVERVIEW.md`, `CHALLENGE.md`, `DATA.md`, `EVALUATION.md`, `RULES.md`, `SUBMISSION.md`, `FAQ.md`, `CONTEXT.md` (this).
- `games.py` — path/frames/halves helpers.
- `starter/`: `baseline.py` (nearest-player-to-ball + majority vote, numpy only; ball missing → D), `load_atd.py`, `load_homography.py`, `load_smart_tagging.py`, `utils.py`, `make_dummy_submission.py`, notebooks 01–03.
- `starter/tested_models/` (our work): `HMM.py` (3-state H/A/D Viterbi, heuristic evidence, `--switch-seconds`, `--out`), `explain_hmm.py`, `tune_hmm_from_dfl.py` (calibrates transitions from DFL events), `lightGBM.py` (**empty, 0 lines**), `notes.txt`.
- `evaluation/`: `possession_metrics.py` (metric spec), `run_evaluation.py`, `validate_submission.py`, `gt/` (public GT + meta), `results/` (baselines + `scoreboard.csv`, `reports/`), `report/`.
- Env: `.venv` (py3.14), `requirements.txt` (numpy, pandas), `requirements-starter.txt` (kloppy etc.), `.mise.toml`.

## 7. Our progress so far
Team-notes ideas table: (a) heuristics + HMM/Viterbi (no external labels), (b) LightGBM on window features + smoothing (needs labels from other games), (c) GRU/1D-CNN/transformer + smoothing (most data/tuning).

| run (public 5 min) | macroF1 | spellF1 | #poss (GT 17) |
|---|---|---|---|
| `hmm_switch40` | 0.6418 | 0.2533 | 43 |
| `hmm_switch16.9` | 0.6414 | 0.2313 | 47 |

- HMM status: macroF1 slightly higher than baseline but spellF1 ≈ baseline-level; tuning didn't help. Known main flaw: **flags airborne balls as dead (D)**; produces many short possessions (43–47 vs 17 true). Parameters (see README): `SECONDS_BETWEEN_SWITCHES` 12, `_STOPS` 40, `SECONDS_OF_DEAD_BALL` 10, `CLOSER_SLOPE` 0.6, `LOOSE_BALL_DISTANCE` 6, `STILL_SPEED` 0.7/`STILL_FRAMES` 25, `OFF_PITCH_MARGIN` 0.01, `D_LOGLIK_*` 3.0/1.5/−0.5, `SPEED_LAG_FRAMES` 5, `FLIGHT_SPEED` 10/`FLIGHT_DISTANCE` 3 (mid-air → X).
- Public-window numbers above are sanity checks only; HMM transition priors were calibrated from DFL (allowed: other games).
- TODO: try stronger models (LightGBM / sequence NN); then write the technical doc and the reproducible repo.

## 8. Additional dataset: DFL / IDSSE (`/home/bark/Downloads/28196177/`)
Bassek et al. 2025, *Scientific Data* 12:195 (figshare 28196177; CC-BY 4.0 — **must credit DFL and cite the paper** and say so in the tech doc). Clone of `spoho-datascience/idsse-data` is in `idsse-data/` (`data_processing.py`, `visualzation.py`, `data_summary.ipynb`; deps floodlight 0.5.0, pandas 2.2.2, matplotlib, seaborn; its own `.venv` at `28196177/.venv`).

- **7 Bundesliga matches** (1. & 2. Bundesliga; e.g. Köln–Bayern MD34 2022/23, DFL-MAT-J03WMX/WN1/WOH/WOY/WPY/WQQ/WR9). ~2.5 GB in `data/`.
- Per match, 3 XML files: `DFL_02_01_matchinformation_*` (teams, players, pitch 105×68, kickoff), `DFL_03_02_events_raw_*` (official event data, ~0.6–0.8 MB), `DFL_04_03_positions_raw_observed_*` (TRACAB 25 fps positions for players, referee, ball, ~350–420 MB each; `<Frame N T X Y D S A M>`, **metres, origin at pitch centre**, `FrameSet GameSection=firstHalf/secondHalf`, per `PersonId`/`TeamId`; ball has its own FrameSet incl. ball status and possession info).
- Why it matters: **legit external data from other games** — event data (possession changes, restarts/stoppages, ball status) + high-quality tracking → can derive pseudo-GT possession labels (H/A/D) to train LightGBM/NN or calibrate HMM priors. Caveats: it's *not* Metrica ATD — need to degrade it to look like ATD (drop ball ~41%, track fragmentation/ID-less slots, noise, normalise to 0–1 with 105.3×68, flip attack direction/home-away conventions) to avoid train/test domain gap. Different league/teams, hand-corrected-ish event data is OK since it's other games.
- Already used: `tune_hmm_from_dfl.py` (possession duration/stoppage stats → HMM transition params).
- Not yet used: training labels from DFL event+position data for a supervised model.

## 9. Gotchas checklist
- Row count must be 199,319 (≈164k rows ⇒ you indexed broadcast).
- `frame_id − 1` for ATD; `round(sec·25)` for Smart Tagging; homography already 0-based.
- Scale x by 105.3 and y by 68 before distances.
- Smart Tagging case: titles/codes differ in case; `Max Players in the box` is a string.
- Don't tune on public GT; use DFL-derived labels/bootstraps to choose hyperparameters.
- Aim for far fewer possessions than the raw approaches (true mean ≈ 14 s on public window); use X inside possessions during ball dropouts, guess at transitions.
- Don't send data/code containing data extracts to hosted LLMs.

## 10. Pipeline findings (2026-11-13) — code in `pipeline/`
- `pipeline/ingest_dfl.py` → `data/pipeline/dfl_raw/<match>.npz` (7 matches, ~72 s). DFL ball frames carry `BallStatus` (1 alive/0 dead) and `BallPossession` (1 home/2 away, verified vs nearest player). Labels = D if dead else team.
- `pipeline/labels.py`: `debounce(lab, min_spell_seconds)` merges short H/A spells. Raw DFL spells are touch-level (median ~3 s). Sweep over 7 matches (possessions / mean s): N=0 3061/7.2, 1 2551/8.7, 2 1964/11.3, 3 1634/13.5, 5 1281/17.3, 8 1036/21.4. N is a hyperparameter, default 3 s set a priori; do NOT pick it on the public GT.
- `pipeline/atd.py` → canonical ball [T,3], players [T,46,3], team [46] (cached in `data/pipeline/atd_tactical.npz`).
- **ATD ball speed is unusable**: present on only 7.8% of visible frames, max 7.3 m/s, corr 0.06 with finite-difference speed. Always recompute speed from positions (finite-diff p50 5.1, p99 24.6 m/s). `starter/tested_models/HMM.py` relies on the reported speed (STILL/FLIGHT cues) — likely part of its weakness.
- **ATD ball is missing in long blocks, not per-frame**: 150 missing runs over the graded halves, median 10.7 s, mean 17 s; visible runs median 13.8 s. Loss/regain cluster near goal lines (~40–44% vs 19% overall) and touchlines (42–46% vs 32%); players move slower while the ball is missing (1.2 vs 2.1 m/s) → missingness correlates with stoppages.
- Player-only features (`pipeline/player_features.py`) predict ball-missing with AUC 0.82–0.85 cross-half (single feature mean player speed: 0.84). So the DFL degradation must be conditional on player behaviour (fit on unlabelled ATD, apply to DFL), not an i.i.d./unconditional dropout.
- Env note: `.venv` has numpy/pandas/scipy/lightgbm but no scikit-learn → use `lightgbm.train` (native API), not `LGBMClassifier`.

### Degradation + domain check (2026-11-13)
- `pipeline/degrade.py` (`--seeds 3`): DFL -> canonical (0-1 coords, anonymous slots) -> player dropout (rare per-slot + correlated "view loss" episodes) -> ball-missing mask sampled from a LightGBM P(missing | player-only features) fitted on the unlabelled ATD (`data/pipeline/missing_model.txt`) -> `data/pipeline/dfl_canon/<match>_s<seed>.npz` (21 files). Position noise is ~0 on purpose: clean DFL player speeds already match ATD (median 1.9/1.6 vs 1.5 m/s) and independent per-frame noise inflates finite-diff speed by ~10 m/s.
- Matches the ATD on: ball-missing share (39% vs 41%), missing-run length (median 11.4 s / mean 17.1 s vs 10.7 / 17.2), tracked players (p10/50/90 18/22/22), player speed.
- `pipeline/check_domain.py`: adversarial AUC (degraded DFL vs ATD, orientation-free features, split by match/half) = **0.84**; per-match z-scoring (`--normalise`) leaves it at 0.84. Remaining tells: team spread (`sx_h`, `sx_a`), `ball_speed`, `abs_cx_mid` -> likely real league/team/tracker differences. Treat as a known gap: validate downstream, and be wary of features with high adversarial importance.
- Known approximations: missing runs are geometric (ATD heavier tailed); no ball false positives/jumps modelled; ball visibility is not conditioned on true ball state (only on player behaviour).

### HMM tuning on degraded DFL (2026-11-13) — `pipeline/tune_hmm.py`
- Runs the real `HMM.py` on degraded DFL (matches 0-3, seeds 0-1 to tune; matches 4-6 with a fresh seed held out), scored with `possession_metrics.evaluate`; reference labels = DFL debounced at N=3 s. 400 trials (defaults, 200 random, 200 local refinement), ~10 min on 16 workers. Results in `data/pipeline/hmm_tuned.json`, log in `data/pipeline/tune_hmm.log`.
- Held-out DFL spellF1: HMM defaults 0.204, **tuned 0.520**, smooth50 baseline 0.279 (also 0.476/0.532 at reference N=2/5 s). Untuned HMM over-segments ~2.5x on both degraded DFL and the real ATD -> degraded DFL is a reasonable proxy.
- Sanity check on the real ATD public window (run once, not used to choose anything): defaults 0.231 (47 spells; reproduces the old `hmm_switch16.9` exactly), **tuned 0.490 (14 spells vs 17 true)**, smooth50 0.326. macroF1 0.631 vs 0.641 (flat). Full ATD match with tuned params: 278 possessions, mean 12.7 s.
- Tuned values: switch 51 s, stops 48 s, dead 31 s, CLOSER_SLOPE 0.20, LOOSE_BALL_DISTANCE 2.0 (lower bound), STILL_SPEED 1.77, STILL_FRAMES 59, D_LOGLIK_STRONG 1.0 (lower bound), D_LOGLIK_STILL 0.3 (lower bound), D_LOGLIK_PLAY -0.88, D_LOGLIK_MISSING 1.05, AMBIGUOUS_LLR 1.0 (upper bound), MIN_DEAD_SECONDS 6.0 (upper bound), flight->X and ambiguous->X switched OFF. Several sit on search bounds -> widen and re-run before trusting them.
- Not yet wired into `HMM.py` (needs a `--params` option) and no submission CSV written from it.

### HMM tuning round 2 + `--params` (2026-11-13)
- `HMM.py` now has `--params <json> [--name N]` (applies `best_params` from `pipeline/tune_hmm.py`: constants, switch time, min-dead, flight/ambiguous flags). Verified: with `data/pipeline/hmm_tuned.json` the CLI reproduces the tuner exactly (public window spellF1 0.4903, macroF1 0.6312, 14 possessions; file passes `validate_submission.py`). Run the scorer without `-I` (it imports a sibling module).
- Round 2 (`data/pipeline/hmm_tuned_v2.json`, widened ranges, warm-started, 500 trials): held-out DFL spellF1 0.542 (v1 0.520); public window spellF1 0.4903 (identical to v1), macroF1 0.6302. Converged — differences are within noise. Parameters are weakly identified (switch time drifted to 133 s, near the 150 s bound; D_LOGLIK_STILL hit 0.0, i.e. still-ball cue off; flight/ambiguous->X off).
- Remaining weakness of both: dead-ball recall ~0.25 on the public window (D F1 0.35; 796 of 1469 true D frames labelled H). Next lever is a learned model (LightGBM) with player-only features, not more HMM tuning.
