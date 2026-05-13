# Sprint 5 — README & Final Wiring (Phase 5 + 6)

## Goal
Write the project README, add the `main.py` entry point, and close out the project as a deliverable.

## What was built

### `README.md`
Covers everything the spec asked for:
- Quickstart (`uv run python main.py` and `uv run jupyter nbconvert ...`)
- Project structure
- Data split rationale (temporal split, why random would leak, why CV is round-based)
- Feature table (28 features across 3 groups)
- Model quality assessment with honest conclusions
- What I would do next (more data, player stats, head-to-head, calibration, ordinal regression)

### `main.py`
Replaced the uv-generated stub with a real entry point that runs the full pipeline end-to-end and prints train/test comparison tables. Mirrors what the notebook does but without visualization.

### Notebook regenerated
Re-executed `build_notebook.py` + `nbconvert` to ensure notebook outputs are consistent with the final feature set (schedule strength removed).

## Feature experiment: strength of schedule
Implemented `_add_schedule_strength` in `src/features.py` — average rating of opponents faced before each round. Tested against both models:

| | Without | With |
|---|---|---|
| XGBoost test log-loss | 0.7824 | 0.7857 |
| XGBoost test accuracy | 0.7041 | 0.7007 |
| XGBoost train log-loss | 0.4523 | 0.3952 |

Feature made overfitting worse with no test improvement. Removed. Documented in README under "What I would do next" as worth revisiting with more training data.

## CV design rationale documented
Random CV is invalid here because the same player appears across multiple rounds — the model would learn player-specific patterns from some rounds and exploit them in others, which isn't available at prediction time. Round-based split (1–8 train, 9–11 val) mirrors the temporal structure of the data.

## Final numbers

| Model | Test log-loss | Test accuracy |
|---|---|---|
| Elo baseline | 0.7352 | 69.7% |
| Logistic Regression | 0.7631 | 69.9% |
| XGBoost | 0.7824 | 70.4% |

Project complete and pushable as a take-home submission.
