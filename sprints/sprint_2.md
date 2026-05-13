# Sprint 2 — Feature Engineering (Phase 2)

## Goal
Transform raw game records into a clean feature matrix using only pre-game information. No post-game data (PGN, moves, ECO, end_time, duration) is ever touched.

## What was built

### `src/features.py`

- `map_outcome(white_result)` — maps the API result string to the 3-class target: `2` (white wins), `1` (draw), `0` (black wins). Draw strings: `agreed`, `stalemate`, `repetition`, `insufficient`, `50move`, `timevsinsufficient`.
- `_elo_expected(white_rating, black_rating)` — standard Elo formula: `1 / (1 + 10^((black - white) / 400))`.
- `_add_tournament_points(games)` — core leakage-safe logic: processes rounds in ascending order, assigns cumulative points to each game **before** that round runs, then updates the running total. Round 1 always gets 0.0 for both players.
- `build_features(games)` — main entry point; returns a `pd.DataFrame` with all features + target column.
- `get_feature_columns(df)` — returns the list of model input columns, excluding identifiers (`white_username`, `black_username`, `game_url`) and the target (`outcome`).

## Feature matrix

| Group | Features |
|---|---|
| Rating | `white_rating`, `black_rating`, `rating_diff`, `rating_avg`, `abs_rating_diff`, `elo_expected_score` |
| Tournament context | `round_number`, `white_tournament_points`, `black_tournament_points`, `tournament_points_diff` |
| Title binary | `white_is_{T}` / `black_is_{T}` for T in GM, IM, FM, WGM, WIM, CM, NM, WFM, WCM |

Total: **28 feature columns** + 3 identifier columns + 1 target = 32 columns.

## Validation results (February / train tournament)

| Metric | Value |
|---|---|
| Shape | (2025, 32) |
| Feature columns | 28 |
| White wins | 987 (48.7%) |
| Draws | 164 (8.1%) |
| Black wins | 874 (43.2%) |

**Anti-leakage check passed**: all round-1 tournament points are 0.0.

Tournament points at round 5 (sample): players showing 4.0 pts — consistent with 4 wins from 4 prior rounds.

## Design decisions

- **Round-by-round accumulation**: games are sorted by `round_number`, points assigned before updating — no leakage possible.
- **Extra titles included**: CM, NM, WFM, WCM appear in the data beyond the spec's GM/IM/FM/WGM/WIM — all encoded as binary features since they may carry discriminative signal.
- **Identifiers kept in DataFrame** for debugging and EDA, excluded from `get_feature_columns()` so they never reach the model.

## What's next
Phase 3 — Modeling (`src/modeling.py`): Elo baseline → Logistic Regression → XGBoost, with temporal train/test split and round-based cross-validation.
