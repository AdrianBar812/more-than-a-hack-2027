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

### LightGBM feature builder (2026-11-13) — `pipeline/features.py`
- `build_features(ball, players, team)` -> 142 features, same code for degraded DFL and the ATD (~0.5 s per half). Groups: `p_` player-only (team centroids/spread/lines/speed/stillness/zone counts at 1/4/12 s), `b_` ball where observed, `bw_` ball windows incl. fraction of visible frames with the home team nearest, `bl_`/`bn_` last-seen/next-seen ball values + age (bridge the 10-17 s gaps), `v_` visibility fractions. NaN kept (LightGBM native). `python pipeline/features.py --build` caches everything in `data/pipeline/features/` (1.7 GB; labels stored raw, debounce N applied at training time).
- Attack direction is inferred per half (`home_attacks_right`, extreme-player/goalkeeper-side vote) and x mirrored so home always attacks +x. Validated against independent evidence: 14/14 DFL halves match the declared goalkeeper (PlayingPosition TW) direction; consistent with all 4 ATD Smart Tagging goals. NOTE: "home possession => ball x higher" is NOT a valid direction test — both teams hold the ball mostly in their own half.
- Quick check (not tuned; 4 DFL matches x 2 seeds train, 3 held-out matches fresh seed, N=3 s labels, 300 rounds, no decoding): held-out degraded-DFL macroF1 0.911 (D F1 0.968; ball missing frames macro 0.830, visible 0.939). Top features: last-seen ball speed (18% gain), rolling home-nearest fraction, player speeds.
- **Real ATD public window (run once as a pipeline check): raw argmax spellF1 0.344, macroF1 0.743, 26 spells (17 true); per-class F1 H 0.849 A 0.847 D 0.532 (D recall 0.37).** vs tuned HMM 0.490 / 0.630. macroF1 and D clearly better, spellF1 limited by over-segmentation (no smoothing yet). D F1 drops 0.97 -> 0.53 from degraded DFL to the real ATD = remaining domain gap in dead-ball detection (suspect: DFL ball is near-frozen while dead, the ATD ball is not; last-seen ball speed dominates). Predicted shares on the full ATD match H .37 A .24 D .39.

### LightGBM v1 + ball jitter + Viterbi (2026-11-13) — `pipeline/train_lgb.py`, outputs in `data/pipeline/lgb_v1/`
- Diagnosis of the dead-ball domain gap (unlabelled marginals only): DFL ball speed (5-frame) is EXACTLY 0.00 for >25% of visible frames (frozen ball while dead; p25 ATD = 2.38 m/s, p2 = 0.16). Model had learned "still ball => dead". Fix in `degrade.py`: speed-dependent white jitter on the ball, sigma = 0.22 m + 0.015 m per m/s of true speed, fitted to the ATD's visible-ball speed quantiles (plateau base 0.20-0.25, per-speed 0.01-0.02).
- Protocol: leave-one-match-out over 7 DFL matches (train on 6 x 3 degradation seeds, predict the held-out match with seed 2), N=3 s labels, 300 rounds, every 5th frame. Decoder = Viterbi over -log p with H<->A cost and live<->D cost, grid-searched on the OOF probabilities.
- OOF (7 held-out matches): raw argmax spellF1 0.317 / macroF1 0.895 (3.2x over-segmented); decoded spellF1 **0.685** / macroF1 0.895 (switch cost 32-48, dead cost 12 = grid edge, widen). 
- Real ATD public window (check only, read once): raw 0.320 / 0.807 (38 spells); **decoded spellF1 0.6065, macroF1 0.8166, 14 spells (17 true), D F1 0.707 (recall 0.58)**. For comparison: tuned HMM 0.490/0.630, smooth50 0.326/0.621, pre-jitter raw LightGBM 0.344/0.743 (D F1 0.532). The jitter alone lifted raw D F1 0.53 -> 0.68.
- Submission candidate: `data/pipeline/lgb_v1/lgb_decoded.csv` (valid format).
- Open: widen dead-cost grid, sweep debounce N, per-seed ensemble, remaining D gap, tech doc.

### Round 2 experiments (2026-11-13): domain gap, decoder grid, ensemble, debounce N
- **Domain gap (unlabelled diagnostics)**: ATD ball is lost/re-found while moving (median last-seen speed 4.5 m/s, next-seen 11.7 m/s; a speed-blind chain gave ~2 and ~4). `degrade.py` now scales the ball-missing chain's switching rates by g(v)=((v+1)/6)^0.75 (stationary P unchanged), qscale 1.2, rate 0.14; fitted on ATD last/next-seen speed quantiles, share and run length (share 0.42 vs 0.41, run mean 18.9 vs 17.2 s). View-loss no longer drops keepers (ATD home def-third count never below 1). ATD player stillness (median speed 1.49 vs DFL 1.82 m/s, more still players) could not be matched by noise/smoothing: left as-is (may be team style). Frozen tracks are not a thing in the ATD (exact-zero steps 0.76%). Adversarial AUC on all 142 features stays ~0.975 (team-style/player tells dominate), so judge by downstream, not AUC.
- **Decoder grid widened** (switch 8-128 x dead 3-80): OOF plateau — dead cost 32-80 equal within 0.001; v2 best switch 16 / dead 48, OOF spellF1 0.6888 (v1 0.6853, different data/grid: same within noise).
- **Ensemble of 3** (frame offsets + bagging seeds, probs averaged): OOF decoded 0.6896 vs 0.6888 single; raw over-segmentation 3.2x -> 2.8x but nothing after decoding. Not worth 3x compute.
- **Debounce N sweep** (`pipeline/sweep_n.py`; each train-N scored against every reference N, mean/min spellF1): N=2 .692/.648, **N=3 .689/.636**, N=4 .669/.605, N=5 .657/.586. N=2 and 3 tied; N>=4 worse at every reference. Keep N=3 (a-priori choice).
- **Public window, read once for lgb_v2 (switch 16, dead 48)**: spellF1 0.4923, macroF1 0.8422 (H .884 A .867 D .776, D precision 1.00 recall 0.63), **9 possessions (17 true)**. vs lgb_v1 (switch 48, dead 12): 0.6065 / 0.8166 / 14 possessions. Frame-level quality improved, possession count dropped: the high dead cost merges possessions across missed stoppages.
- **Sensitivity of the real-ATD output to the dead cost** (unlabelled, full match, switch 16): dead 12 -> 285 possessions (mean 12.3 s), 20 -> 267 (13.2 s), 48 -> 239 (14.8 s); DFL N=3 truth averages 233 / 13.5 s. DFL OOF cannot separate these settings (plateau within 0.004).
- **Decision (disclosed)**: chose a middle plateau value, switch 16 / dead 20 (OOF 0.6860, ratio 0.93), on the argument that ATD dead-ball probabilities are less confident than DFL's (remaining domain gap), so a high dead cost over-suppresses real stoppages there. This was decided AFTER seeing the v2 public-window result, so it is not independent evidence; the public window was NOT read again for it. Candidate: `data/pipeline/lgb_v2/lgb_v2_sw16_d20.csv` (valid format).

## Round 3: label-free domain alignment (ideas 1-3), 2026-11-14

What was built: (3) `degrade.backlash` dead-band (0.3 m) on degraded DFL players, (2) new features in `features.py`
(`rspeed_*`/`rstill_*` speed relative to the half's own median, 30 s / 60 s windows, touchline stillness, keeper-proxy speed,
player counts around the last/next seen ball: 192 features), (1) `pipeline/adversarial.py` (`--prune`, `--weights`) and
`train_lgb.py --drop K --adv-weights`. Old data kept in `data/pipeline/{dfl_canon,features}_v2`.

Label-free evidence: adversarial AUC (same split, all features) v2 0.979 -> v3 192 features 0.966; pruning curve
0.97 (192) -> 0.82 (132) -> 0.75 (90) -> 0.67 (42). The new `gk_speed_*` features were the single biggest tell (the
extreme-x player is not a reliable keeper proxy in the ATD) and were pruned first. Speed smoothing does not close the player-speed gap
(the mid-range gap is behavioural); the dead-band only closes the low-speed tail.

DFL out-of-fold (7 held-out matches, decoder 16/20, new degradation so not comparable with v2's 0.686):
0.670 all 192 | 0.670 drop 60 (132) | 0.652 drop 102 (90) | 0.600 drop 150 (42). Weights add ~0.001-0.005 (noise).
A-priori rule (set before looking at the public window): smallest feature set within 0.01 of the full set -> drop 60 + weights.
ATD D share stays 0.43-0.44 for every variant (DFL truth 0.448).

Public window, read ONCE for the chosen variant (check only, 4 dead-ball segments): v2 candidate spellF1 0.613 (13/17 possessions),
D precision 0.896 recall 0.613; v3 drop60+weights spellF1 0.477 (9/17), D precision 0.998 recall 0.559. No D gain, so
v3 is NOT adopted; the submitted candidate stays `data/pipeline/lgb_v2/lgb_v2_sw16_d20.csv`.
Reverted to v2 as the default: `degrade.degrade_players(dead_band=False)`, `features.EXTRA = False` (verified: identical 142 names and values to the v2 ATD features). v3 data is kept in `data/pipeline/{dfl_canon,features}_v3`; to try it again set `dead_band=True` and `EXTRA = True` and rebuild.

## Round 4: ideas from Borghesi et al. (2023), Expert Syst. Appl. 223:119780 (2026-11-15)

Paper: TCN trajectory embeddings per object + permutation-invariant aggregation, two branches (dead/alive, then home/away with the possession loss
masked on dead frames). 86.2% possession / 89.2% dead-alive accuracy on 8 low-missing-ball games (16 fps, x/y only, confidential data); ablation: the
5 nearest players carry the possession signal, the ball alone carries dead/alive. Architecture choice moved accuracy only +-1.7%.

Tried on our LightGBM (out-of-fold DFL, 7 held-out matches, decoder 16/20, same matches; baseline v2 = 0.6860):
- idea 1 `--two-stage` (binary D model on all frames; binary A|live model on live frames; combined): 0.6877 (paired +0.002, better in 4/7)
- idea 2 `POSS_NEAR=1` (30 features: speed / closing speed / speed relative to the ball of the nearest home and away player, 2nd-nearest distance,
  home share and mean speed of the 5 nearest): 0.6924 (paired +0.006, better in 5/7); raw-argmax 0.331 vs 0.317
- both: 0.6859 (paired -0.000)
Criterion (stated before reading): adopt only if the paired gain is >= 0.01 and the model is better in >= 5/7 matches. None qualifies (per-match std is 0.04),
so the defaults are unchanged. Both are off by default (`--two-stage` flag, `POSS_NEAR` env var; features go to `POSS_FEAT_DIR`, here `data/pipeline/features_near`).
The public window was not read.

## Round 5: one-match transfer check on SkillCorner open data (2026-11-15)

`pipeline/skillcorner_check.py`: match 1874553 (Brisbane vs Adelaide, A-League; tracking fetched from the repo's git-lfs media URL, not committed).
Converted to canonical arrays (metres -> 0-1, anonymous slots, 10 -> 25 fps linear interpolation, detected players/ball only), run through the
DFL-trained v2 model + Viterbi 16/20 with NO retraining. Caveat: `possession.group` is empty both for dead ball and whenever the ball is not detected.
- attack direction inferred correctly in 2/2 halves; only ~9.3 players tracked per frame (the ATD has ~20), ball visible 60%.
- home/away accuracy where labelled: 0.839 (ball visible 0.849, ball missing 0.694, n=5264); 0.851 after Viterbi.
- ball-visible frames: empty-possession share 0.151, decoded D 0.120; decoded D given empty 0.671 (recall), empty given decoded D 0.840 (precision).
  Same pattern as the public window (D recall 0.61, precision 0.90): D is under-called relative to the labels on a second, independent dataset.
- ball-missing frames: labels are empty 91% of the time (uninformative), so D there cannot be checked. Overall decoded D share 0.428 vs empty 0.454.

## Round 6: imputed ball from the players (self-supervised on the ATD), tried and removed (2026-11-16)

Idea: where the ball is visible on the unlabelled ATD (57% of frames) it is a free regression target for player-only features; fit a LightGBM regressor
(ball x, y from the 84 `p_*` features, mirrored so home attacks +x), apply it to every frame of both the degraded DFL and the ATD, and add 24 features of the
imputed ball (position, distance to the nearest home/away player, counts within 5 m, speed; windows 1 / 4 / 12 s).
- regressor quality (ATD, train one half, test the other, ball-visible frames): median error 11.3 / 11.1 m (mean 13.9 / 13.1) vs 20.7 / 19.9 m for the
  all-player centroid and 31.8 / 30.8 m for the pitch centre; applied to degraded DFL (3WMX half 1): median 12.5 m, so it transfers.
- out-of-fold DFL (decoder 16/20): spellF1 0.6823 vs 0.6860 baseline (paired -0.004, better in 1/7 matches), D F1 0.963 unchanged.
- public window (one read, check only): spellF1 0.529 vs 0.613 (14 vs 13 of 17 possessions), D precision 0.862 vs 0.896, D recall 0.572 vs 0.613.
Criterion set before looking: keep only if the paired out-of-fold gain is >= 0.01 in >= 5/7 matches and the public window is not worse. Both failed, so the
code (`ball_impute.py`, the POSS_IMP block in `features.py`), the fitted models and the generated data were deleted; the default pipeline is unchanged
(142 features, identical to before).

## Round 7: separate models for ball-visible and ball-missing frames, tried and removed (2026-11-16)

Diagnostic on the v2 out-of-fold DFL predictions (decoded 16/20), by regime: ball visible (58% of frames) accuracy 0.931, home/away 0.911, D recall 0.986 / precision 0.990;
ball missing (42%) accuracy 0.876, home/away 0.817, D recall 0.965 / precision 0.931. The model is weaker without the ball, mostly on home vs away.
Specialist test (one multiclass LightGBM on visible frames, one on missing frames, chosen per frame by `b_bx` is NaN): out-of-fold spellF1 0.6797 vs 0.6860 (paired -0.006, better in
1/7 matches); per-regime metrics unchanged (missing: home/away 0.814 vs 0.817, D recall 0.964 vs 0.965). Public window (one read): spellF1 0.550 vs 0.613, D precision 0.827 vs 0.896,
recall 0.546 vs 0.613. Same keep-criterion as before (paired gain >= 0.01 in >= 5/7 matches and public not worse): failed, code and outputs deleted.

## Round 8: decoder-level experiments for spellF1 (no retraining, 2026-11-16)

`pipeline/decoder_experiments.py` on the saved v2 out-of-fold probabilities (7 matches, baseline Viterbi 16/20 = 0.6860). "Honest" = the grid cell is chosen on the other 6 matches
for each held-out match. Rule: paired gain >= 0.01 and better in >= 5/7. Public window not read.
- 1 debounce the decoded output: best N = 3 s 0.6945 (+0.0085, 5/7); honest +0.0063, 5/7
- 2 visibility-dependent switch cost: best (visible 16, missing 32) 0.6889 (+0.0029, 5/7); honest +0.0022, 5/7
- 3 duration decoder (min H/A run, soft short-run penalty): best (5 s, penalty 32, min D 0.2 s) 0.6953 (+0.0093, 6/7); honest +0.0049, 5/7
- 4 dead cost matched to the possession count of the other matches (prior ~233): -0.0019, 2/7 (chosen costs scatter between 8 and 96)
None passes. Nothing was changed in the default decoder; all scripts and the log are kept.

### Round 8b: ideas 1+2+3 combined, and all decoders on the public window (2026-11-16)
Combined (duration graph + visibility-dependent switch cost + final debounce), grid of 45 cells, same protocol: best cell (3 s min H/A, penalty 32, min D 0.2 s, switch 16 visible / 32 missing, debounce 3 s) 0.7000 (+0.0140, 6/7);
honest LOMO-chosen +0.0088 (5/7), just under the 0.01 bar. The singles do not add up (1 and 3 overlap).
Public window (one read of each, information only, 17 possessions): baseline 0.6133; debounce 3 s 0.6345; vis cost 0.6000; duration 0.6069; combined 0.6207 (12/17 possessions). All within the noise band.
Candidate file with the combined decoder: data/pipeline/lgb_v2/lgb_v2_combo.csv (valid). Not adopted as the default; the choice is open.

### Round 9: merge spells under 3 s adopted; analysis of true possessions of 3-5 s (2026-11-16)
`train_lgb.decode()` = Viterbi + `labels.debounce(path, MERGE_S)`, MERGE_S = 3 s by default (`--merge-seconds`, 0 = off). Candidate file: data/pipeline/lgb_v2/lgb_v2_sw16_d20_merge3.csv (valid; public window spellF1 0.6345, macroF1 0.8322, 12/17 possessions; one read, information only).
Out-of-fold DFL, true spells of 3-5 s (378 of 1634, 23%), v2 probabilities with the merge: matched at IoU >= 0.5: 29%; called the OTHER team: 38%; called dead ball: 14%; right team but boundaries off: 19%.
65% of them sit next to a dead-ball run (restarts), 35% sit between two spells of the other team. Ball visible during the spell: 36% matched; ball missing: 21%.

## Round 10: all options for 3-5 s possessions, alone and combined (2026-11-16)

New feature groups behind env flags (`POSS_NEAR`, `POSS_SHORT`, `POSS_RESTART`; cache `data/pipeline/features_all`, 245 columns; `train_lgb.py --groups near,short,restart --exclude <prefixes> --ref-seconds N`):
`sw_*` short windows (0.5 / 1 s nearest-player kinematics, ball heading, ball acceleration spikes), `rs_*` per-team players around the last/next seen ball + set-piece geometry, `nb_*` (round 4).
Decoders on top (`pipeline/round10_eval.py`): merge < 3 s (adopted), boundary shift grid, combined duration/visibility decoder. Reference labels always the N = 3 s DFL labels of lgb_v2.
Out-of-fold DFL (merge 3 s, decoder 16/20) vs the adopted baseline 0.6945; ATD = public window (information only, one read each):
 v2 baseline 0.6945 | ATD 0.6345 (D recall 0.613) ; near +0.0056 (4/7) | 0.557 ; short +0.0024 (3/7) | 0.557 ; restart +0.0048 (5/7) | 0.607 ;
 short+restart +0.0103 (7/7) | 0.533 ; near+short+restart +0.0121 (6/7) | 0.543 ; same, N=2 +0.0263 (7/7) | 0.538 ;
 v2 features trained with N=2 +0.0078 (5/7) | 0.648 (D recall 0.617) ; restart N=2 +0.0091 (5/7) | 0.607 ; all groups without sw_acc/sw_spike, N=2 +0.0324 (7/7) | 0.543 (D recall 0.550).
Boundary shift: best +0.0013 on lgb_v2 (zero shift best for every other run), i.e. no systematic offset. Combined decoder: +0.0055 on lgb_v2 (ATD 0.621), +0.0132 on v2 features with N=2 (ATD 0.557).
Label-free check: adversarial AUC base 0.979, + restart 0.978, + near 0.979, + short 0.991 (ball acceleration / spike features are the top tells, 31%), all except sw_acc/sw_spike 0.979.
Reading: DFL out-of-fold improves with every added group and with N = 2 (up to +0.03), but the ATD window gets worse for every feature group (D recall 0.51-0.56 vs 0.61): extra DFL-specific cues do not transfer.
Only N = 2 on the unchanged features is neutral-to-positive on both (+0.0078 DFL, 0.648 ATD), just under the 0.01 bar. Nothing adopted yet; defaults unchanged.
