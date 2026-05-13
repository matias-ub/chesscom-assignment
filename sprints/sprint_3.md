# Sprint 3 — Modeling (Phase 3)

## Goal
Train three models on the February tournament and evaluate them on the March tournament. Establish the Elo baseline, add Logistic Regression as the interpretable model, and XGBoost as the boosted model.

## What was built

### `src/modeling.py`

- `train_elo_baseline(df_train, feature_cols)` — grid-searches the best (low, high) thresholds on the CV validation fold (rounds 9-11), estimates draw rate from training data. Returns a params dict.
- `predict_elo(df, params)` — converts `elo_expected_score` to class predictions and a 3-class probability matrix using `P(draw) = draw_rate`, `P(white_win) = (1 - draw_rate) * E`, `P(black_win) = (1 - draw_rate) * (1 - E)`.
- `train_logistic(df_train, feature_cols)` — sklearn Pipeline with StandardScaler + LogisticRegression (lbfgs, C=1.0, max_iter=1000). `multi_class` param dropped — removed in scikit-learn 1.8.
- `train_xgboost(df_train, feature_cols)` — XGBClassifier with `multi:softprob`, uses rounds 1-8/9-11 split for CV eval set, then refits on full training data.
- `evaluate(name, y_true, y_pred, y_proba)` — returns log-loss, accuracy, confusion matrix, and full classification report.
- `build_comparison_table(results)` — DataFrame with all models × all metrics.
- `plot_confusion_matrices(results)` — matplotlib figure, one heatmap per model.
- `plot_feature_importance(model, feature_cols)` — horizontal bar chart of XGBoost top-N features.
- `run_pipeline(df_train, df_test, feature_cols)` — orchestrates all three models; returns train AND test comparisons, trained models, and params.

## Results

### TRAIN (February)
| Model | Log-loss | Accuracy | F1 draw |
|---|---|---|---|
| Elo baseline | 0.7484 | 0.7002 | 0.000 |
| Logistic Regression | 0.7242 | 0.7037 | 0.012 |
| XGBoost | **0.4523** | **0.8356** | 0.348 |

### TEST (March)
| Model | Log-loss | Accuracy | F1 draw |
|---|---|---|---|
| Elo baseline | 0.7352 | 0.6968 | 0.000 |
| Logistic Regression | 0.7631 | 0.6992 | 0.000 |
| XGBoost | 0.7824 | **0.7041** | 0.012 |

### XGBoost top 5 features
| Feature | Importance |
|---|---|
| rating_diff | 0.1994 |
| abs_rating_diff | 0.0526 |
| round_number | 0.0476 |
| black_tournament_points | 0.0458 |
| white_is_GM | 0.0434 |

## Key observations

- **XGBoost overfits heavily**: log-loss 0.45 on train vs 0.78 on test. With only ~2000 games and 28 features, the model memorizes training patterns that don't generalize.
- **Elo baseline is hard to beat**: Logistic Regression and XGBoost both match or barely beat it on accuracy, and LR is actually worse on log-loss on test. This is expected — rating difference explains most of the variance.
- **Draw prediction is essentially unsolved**: all models have F1 ≈ 0 for draws on the test set. Draws are rare (8%) and hard to distinguish from pre-game info alone.
- **Elo thresholds**: low=0.44, high=0.50 — the high threshold collapsing to 0.50 means the model learned to almost never predict draws, which is rational given their low frequency.

## Design decisions

- CV split is temporal by round (1-8 train, 9-11 val) — no random shuffling to avoid leaking within-tournament player form.
- Train metrics included to make overfitting visible, not just test metrics.
- Logger prints per-model progress during training; tables printed once by the validation script only (no duplicate output).

## What's next
Phase 4 — Notebook (`notebook.ipynb`): wire all modules into a clean, executable notebook with EDA, visualizations, and the final discussion section.
