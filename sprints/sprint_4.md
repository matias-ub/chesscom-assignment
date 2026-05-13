# Sprint 4 — Notebook (Phase 4)

## Goal
Wire all src/ modules into a clean, fully executable notebook with EDA, visualizations, and the final discussion. Must run end-to-end with `uv run jupyter nbconvert --to notebook --execute notebook.ipynb`.

## What was built

### `build_notebook.py`
Generator script that produces `notebook.ipynb` using nbformat. Kept separate from the notebook itself so the cell source stays readable Python rather than raw JSON.

Run once to regenerate: `uv run python build_notebook.py`

### `notebook.ipynb`
Six sections, all heavy logic delegated to `src/`:

**Setup** — imports, logging, matplotlib inline, seaborn theme.

**1. Data Collection** — calls `fetch_tournament_games` for train and test, shows game count and a sample table (pgn excluded).

**2. EDA** — four plots:
- Outcome class distribution (train vs test side by side)
- Rating KDE distributions by outcome (white and black rating)
- Boxplot of rating difference (White − Black) by outcome
- KDE of Elo expected score by outcome

**3. Feature Engineering** — feature matrix shape and `.describe()` table, full correlation heatmap of numeric features (mask removed after visual artifact with seaborn).

**4. Modeling & Evaluation** — calls `run_pipeline`, displays train and test comparison tables side by side, confusion matrix heatmaps (one per model), XGBoost feature importance bar chart, and Logistic Regression coefficient table.

**5. Discussion & Next Steps** — written inline as markdown (see content below).

## Validation
Notebook executes clean end-to-end:
```
[NbConvertApp] Writing 565737 bytes to notebook.ipynb
```
No errors, all outputs and plots embedded.

## Visual fix
Removed `mask=corr.abs() < 0.05` from the correlation heatmap — seaborn's masked heatmap leaves background squares misaligned with the grid. Full correlation matrix shown instead.

## Discussion highlights (from notebook section 5)

**Temporal split rationale**: February → train, March → test. A random split would leak within-tournament player form and inflate test metrics.

**Model quality**: Elo baseline (log-loss 0.7352) is hard to beat. Logistic Regression is worse on log-loss despite marginally better accuracy — its probabilities are less calibrated. XGBoost overfits heavily (train log-loss 0.45 vs test 0.78) with only ~2000 training games.

**Draw prediction**: F1 ≈ 0 for all models on the test set. Draws are 8% of games and determined by in-game tactics, not pre-game information.

**Top feature**: `rating_diff` dominates XGBoost importances, confirming the Elo formula already captures most available signal.

**Next steps identified**: player historical stats, head-to-head history, more training tournaments, calibration analysis, ordinal regression, opening repertoire tendency features.

## Design decision logged
Considered adding average-rating-of-wins/draws/losses as features. Rejected for this sprint: high sparsity in early rounds, redundant with existing `tournament_points` features given Swiss pairing, and XGBoost already overfitting with 28 features on ~2000 games. Strength-of-schedule (`avg_opp_rating`) noted as a cleaner alternative if more features are needed.

## What's next
Phase 5 — `README.md` writeup covering data split, model assessment, and next steps.
