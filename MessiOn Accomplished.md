# Technical document: ball possession (H / A / D) for FC Barcelona vs Athletic Club

Team: **<team_name>** (replace before submitting; the file must be named `<team_name>.md` or `.pdf`)

## 1. Summary

Per-frame possession labels for the 199,319 tactical-ATD frames, produced by an algorithm only:

1. 142 hand-written features per frame from the tactical tracking (players and ball).
2. A LightGBM multiclass model (H / A / D) trained on seven Bundesliga matches that were degraded to look like the ATD.
3. Viterbi decoding of the per-frame probabilities (switch cost 16, dead cost 20), then a merge of home/away spells shorter than 3 s.

The submitted file is `<team_name>.csv`. No label of the competition match was used to train, tune or select anything.

## 2. Approach and why

The ranking metric is `spellF1` (possession spells matched by overlap), so a per-frame classifier is not enough: raw argmax output scores `spellF1` 0.32 on held-out DFL matches while the same probabilities, decoded, score 0.69. The frame accuracy is high (about 0.82), but short flips between teams break spells.

- **Features** (`pipeline/features.py`, one builder for DFL and ATD, so train and test are processed identically): team centroids, spread, lines and speed at 1 s, 4 s and 12 s; ball distance to each team, ball speed and region where the ball is observed; windowed means of those signals; the last-seen and next-seen ball position, speed and age (these bridge the 10-17 s gaps where the ATD loses the ball); how much of the surrounding time the ball is visible. Home is always mirrored to attack towards +x (the direction is inferred from the data).
- **Model** (`pipeline/train_lgb.py`): LightGBM multiclass, 300 rounds, 31 leaves, learning rate 0.05, min 200 rows per leaf. Leave-one-match-out for model selection.
- **Decoder**: Viterbi over -log p with a cost for H<->A switches (16) and for entering or leaving D (20), then H/A spells shorter than 3 s are merged into a neighbour (dead-ball runs are never changed). Costs were chosen on the out-of-fold DFL score. The 3 s merge was adopted because it gave +0.006 out of fold with the grid cell chosen on the other six matches.
- **Why not a neural network or an HMM alone**: the tuned HMM from the starter scored 0.49 `spellF1` on the public window; LightGBM plus decoding scores 0.63. A sequence network was considered but the data (7 matches) and the domain gap make a small, well-regularised model the safer choice.

## 3. Assets used, tried and dropped

Used: tactical tracking (ATD), the DFL / IDSSE dataset for training, the starter scorer and validator.

Tried and **dropped** (all recorded in `CONTEXT.md`, each was good on DFL and did not help on the ATD):

| Idea | DFL out of fold | ATD public window (check only) | Decision |
|---|---|---|---|
| Domain alignment (speed dead-band, relative speed, adversarial pruning and weights) | adversarial AUC 0.98 -> 0.82 | D recall 0.56 vs 0.61 | off |
| Nearest-player, short-window (ball acceleration) and restart-geometry feature groups | +0.002 to +0.012 | `spellF1` 0.53-0.61 vs 0.63 | off |
| Imputed ball positions in gaps | no gain | n/a | removed |
| Separate model for ball-missing frames | no gain | n/a | removed |
| Two-stage model, 3-model ensemble, D probability bias | no gain | n/a | not used |
| Shifting spell boundaries | best +0.001 | n/a | not used |
| Minimum dead-ball length 2 / 2.5 / 3 s | -0.0001 / n/a / -0.0002 | 0.607 / 0.529 / 0.459 | not used |
| Cheaper dead-ball entry when the 5 players nearest the ball are slow | -0.0003 (honest) | 0.61-0.62 | not used |
| Training on SkillCorner broadcast data | one-match check only: H/A accuracy 84-85%, but no usable dead-ball label | n/a | not used for training |

## 4. Training data

- **DFL / IDSSE** (7 Bundesliga matches, CC-BY 4.0). Credit: Deutsche Fussball Liga (DFL). Cite: Bassek, M., Rein, R., Weber, H., Memmert, D. (2025). *An integrated dataset of spatiotemporal and event data in elite soccer.* Scientific Data, 12(1), 195. https://doi.org/10.1038/s41597-025-04505-y
  - Labelled by DFL (ball status and possessing team on every event, produced by DFL's data operations, not by us). Labels: alive ball with the possessing team -> H or A; dead ball or no ball -> D. Spells shorter than 3 s are merged for training (the debounce N).
  - **Degraded to look like the ATD** (`pipeline/degrade.py`): 0-1 coordinates, anonymous per-team slots, player dropouts and view-loss episodes, ball jitter, and a ball-missing mask sampled from a model fitted on the *unlabelled* ATD tracking (P(ball missing | player-only features)). No ATD label is involved. The degraded copies match the ATD on ball-missing share (39% vs 41%) and missing-run length.
- **SkillCorner open data**: one match used only for a transfer check, not for training.
- **Nothing from the provided match is used for training or selection.** No hand labels were made.

## 5. Possession rules implemented

The rules are those of the DFL data, learned by the model, not hand-coded:

- **D (dead):** ball out of play, or no ball status; long stretches where the ball is missing and players are standing still are learned as D. Dead-ball runs are never merged or removed by the decoder.
- **H / A:** the team DFL assigns possession to while the ball is alive. Loose balls and challenges follow DFL's assignment (the team of the last touch). Home/away spells shorter than 3 s are merged into the neighbouring spell, which approximates a "controlled possession" of a few seconds. Two short exceptions: a spell between two dead-ball runs is kept as is.
- **X:** not used (the scorer fills it with the last label anyway).

## 6. Results (for orientation)

Out of fold on the 7 DFL matches (degraded): `spellF1` 0.6945 (raw argmax 0.32). On the public 5-minute window of the competition match, used **only as a pipeline check and read many times for information; nothing was selected from it**: `spellF1` 0.6345, `macroF1` 0.8322, 12 predicted possessions against 17, D recall 0.61. The public window has only 17 possessions, so differences of a few hundredths are within noise (about +-0.04).

## 7. Limitations and what we would do with more time

- **Domain gap**: a classifier separates degraded DFL from the ATD with AUC 0.98. ATD players are slower (median 1.49 vs 1.82 m/s) and the ATD loses the ball while it moves, so D recall is about 0.61 on the public window against 0.97 on DFL. Features that raise the DFL score (ball acceleration, nearest-player cues) lower the ATD score.
- **Ball missing in 43% of the ATD frames** (41% in the graded halves). On the public window the model still misses part of every long stoppage, including stoppages where the ball is visible.
- **Short possessions**: possessions under 5 s match the truth in only about 35% of cases, and about 38% of short spells are called as the other team.
- **Small evaluation**: seven training matches and one 17-possession window.
- **Next**: labelled broadcast-style data closer to the ATD (the SkillCorner labels have no clean dead-ball state), a sequence model once there is more data, and a better ball-missing model for stoppages.

## 8. Reproduction

```bash
pip install -r requirements.txt -r requirements-pipeline.txt
# data: place the ATD (tactical) bundle in data/ as described in README.md; download the DFL/IDSSE data (figshare 28196177, CC-BY 4.0)
python pipeline/ingest_dfl.py --data-dir <DFL data dir>
python pipeline/degrade.py --seeds 3
python pipeline/features.py --build
python pipeline/train_lgb.py --ref-seconds 3 --out-dir data/pipeline/final --costs 16 20
cp data/pipeline/final/lgb_decoded.csv <team_name>.csv
python evaluation/validate_submission.py <team_name>.csv
```

All seeds are fixed. The match data and the DFL data are not in the repository.
