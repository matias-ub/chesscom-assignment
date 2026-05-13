# Chess Game Outcome Prediction

Take-home assignment for a Senior ML Engineer position at Chess.com. Predicts the outcome of Titled Tuesday blitz games (win / draw / loss for White) using **only pre-game information**.

## Quickstart

```bash
uv run jupyter nbconvert --to notebook --execute notebook.ipynb
# or
uv run python main.py
```

First run fetches data from the Chess.com PubAPI and caches it locally. All subsequent runs are fully offline.

## Project structure

```
├── src/
│   ├── data_collection.py   # API fetching + caching
│   ├── features.py          # Feature engineering
│   ├── modeling.py          # Training + evaluation
│   └── utils.py             # Shared helpers
├── notebook.ipynb           # Main reproducible notebook
├── data/                    # Cached API responses (gitignored, regenerable)
└── sprints/                 # Sprint-by-sprint implementation notes
```

## Data split

| Split | Tournament | Games |
|---|---|---|
| Train | February 10, 2026 | 2025 |
| Test | March 10, 2026 | 2038 |

**Temporal split** — the model trains on February and predicts March. This mirrors the production scenario: you always predict future tournaments from past data.

A random split would leak within-tournament information in two ways. First, `tournament_points` for a player in round 9 are a direct function of their results in rounds 1–8 — if those rounds are split randomly across train and val, the model can infer current form from future games. Second, the same player appears across multiple rounds: a random split lets the model learn player-specific patterns from some of their games and exploit them when predicting others, which isn't available at prediction time.

Cross-validation within training also uses a round-based split (rounds 1–8 train, 9–11 val) for the same reason.

## Features

All 28 features are available before the game starts. Three groups:

**Rating** — `white_rating`, `black_rating`, `rating_diff`, `rating_avg`, `abs_rating_diff`, `elo_expected_score`

**Tournament context** — `round_number`, `white_tournament_points`, `black_tournament_points`, `tournament_points_diff` (cumulative points per player before this round, computed round-by-round to avoid leakage)

**Player titles** (binary, one per title per color) — GM, IM, FM, WGM, WIM, CM, NM, WFM, WCM

## Model quality assessment

### Results on March test set

| Model | Log-loss ↓ | Accuracy |
|---|---|---|
| Elo baseline | **0.7352** | 69.7% |
| Logistic Regression | 0.7631 | 69.9% |
| XGBoost | 0.7824 | **70.4%** |

**The Elo baseline is hard to beat.** It achieves the best log-loss of all three models — its probability estimates are better calibrated than either learned model. Logistic Regression and XGBoost match its accuracy but not their probability quality.

**XGBoost overfits.** Train log-loss of 0.45 vs test log-loss of 0.78 is a clear sign of overfitting. With ~2000 training games and 28 features, the model memorizes training patterns that don't transfer across tournaments. The top feature by importance is `rating_diff` (0.20), confirming that the model mostly rediscovers the Elo formula.

**Draws are essentially unpredictable from pre-game data.** All models score F1 ≈ 0 on the draw class in the test set. Draws represent ~8% of games and depend on in-game tactical decisions, not on who the players are before the game starts. Any model that reliably predicts draws would need post-game or in-game features, which defeat the purpose.

The honest conclusion: **rating difference explains most of the predictable variance**, and the ceiling for improvement with only pre-game features is low. The models do no worse than Elo and marginally better on accuracy, but the gap is not meaningful.

## What I would do next

**More data first.** The single biggest lever is training on multiple Titled Tuesday tournaments instead of one. ~2000 games is not enough to learn patterns beyond rating difference.

**Player historical stats** via `/pub/player/{username}/stats` — blitz win rate, recent form, and crucially, performance as White vs Black specifically. Some players score significantly above or below their rating with one color.

**Head-to-head history.** Certain matchups have established patterns (e.g., one GM consistently winning against another regardless of rating). With enough data this becomes a useful feature.

**Calibration.** The Elo baseline wins on log-loss because it produces better-calibrated probabilities. Applying isotonic regression or Platt scaling to the XGBoost outputs would likely close that gap without changing the model architecture.

**Ordinal regression.** Win > draw > loss has a natural ordering that multinomial logistic regression ignores. An ordinal model (e.g., `mord` library) might produce better-calibrated probabilities for the three classes.

**Strength of schedule** (average rating of prior opponents in the tournament). Tested and removed — with ~2000 training games it increased overfitting without improving test metrics. Worth revisiting with more data.
