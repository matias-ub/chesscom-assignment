"""Generates notebook.ipynb from code. Run once: uv run python build_notebook.py"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13.3"},
}

def md(src): return nbf.v4.new_markdown_cell(src)
def code(src): return nbf.v4.new_code_cell(src)

cells = []

# ── Introduction ──────────────────────────────────────────────────────────────
cells.append(md("""\
# Chess Game Outcome Prediction

This notebook predicts the outcome of Titled Tuesday blitz games (win / draw / loss for White) using **only pre-game information**: player ratings, titles, and cumulative tournament standings.

**Data**: Chess.com PubAPI — February 10, 2026 tournament (train) and March 10, 2026 tournament (test).
**Approach**: Elo baseline → Logistic Regression → XGBoost, evaluated with a strict temporal split."""))

# ── Setup ─────────────────────────────────────────────────────────────────────
cells.append(md("## Setup"))
cells.append(code("""\
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import setup_logging
setup_logging()

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 120)
sns.set_theme(style="whitegrid", palette="muted")
%matplotlib inline"""))

# ── 1. Data Collection ────────────────────────────────────────────────────────
cells.append(md("""\
## 1. Data Collection

All API responses are cached locally in `data/raw/` after the first run — the notebook runs fully offline on subsequent executions."""))

cells.append(code("""\
from src.data_collection import fetch_tournament_games

games_train = fetch_tournament_games("train")
games_test  = fetch_tournament_games("test")

print(f"Train (February): {len(games_train)} games")
print(f"Test  (March):    {len(games_test)} games")"""))

cells.append(code("""\
# Sample rows — pgn excluded (post-game, never used as feature)
sample_cols = ["round_number", "white_username", "white_rating", "white_title",
               "black_username", "black_rating", "black_title", "white_result"]
pd.DataFrame(games_train)[sample_cols].head(6)"""))

# ── 2. EDA ────────────────────────────────────────────────────────────────────
cells.append(md("""\
## 2. Exploratory Data Analysis"""))

cells.append(code("""\
from src.features import build_features, get_feature_columns, map_outcome

df_train = build_features(games_train)
df_test  = build_features(games_test)
feature_cols = get_feature_columns(df_train)

label_map = {0: "Black wins", 1: "Draw", 2: "White wins"}
df_train["outcome_label"] = df_train["outcome"].map(label_map)
df_test["outcome_label"]  = df_test["outcome"].map(label_map)"""))

cells.append(code("""\
# Outcome class distribution
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (df, title) in zip(axes, [(df_train, "Train (February)"), (df_test, "Test (March)")]):
    counts = df["outcome_label"].value_counts().reindex(["Black wins", "Draw", "White wins"])
    ax.bar(counts.index, counts.values, color=["#e07b7b", "#aaaaaa", "#7baee0"])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 10, f"{v}\\n({v/len(df)*100:.1f}%)", ha="center", fontsize=9)
    ax.set_title(title)
    ax.set_ylabel("Games")
    ax.set_ylim(0, counts.max() * 1.2)
fig.suptitle("Outcome Distribution", fontsize=13, fontweight="bold")
fig.tight_layout()
plt.show()"""))

cells.append(code("""\
# Rating distributions by outcome
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, col, label in zip(axes,
        ["white_rating", "black_rating"],
        ["White rating", "Black rating"]):
    for outcome_label, color in zip(["Black wins", "Draw", "White wins"],
                                    ["#e07b7b", "#aaaaaa", "#7baee0"]):
        subset = df_train[df_train["outcome_label"] == outcome_label][col]
        subset.plot.kde(ax=ax, label=outcome_label, color=color, linewidth=2)
    ax.set_title(f"{label} distribution by outcome")
    ax.set_xlabel("Rating")
    ax.legend()
fig.tight_layout()
plt.show()"""))

cells.append(code("""\
# Outcome vs rating difference
fig, ax = plt.subplots(figsize=(8, 4))
order = ["Black wins", "Draw", "White wins"]
colors = ["#e07b7b", "#aaaaaa", "#7baee0"]
df_train["rating_diff_plot"] = df_train["white_rating"] - df_train["black_rating"]
sns.boxplot(data=df_train, x="outcome_label", y="rating_diff_plot",
            order=order, palette=colors, ax=ax)
ax.axhline(0, linestyle="--", color="black", alpha=0.4)
ax.set_title("Rating difference (White − Black) by outcome")
ax.set_xlabel("")
ax.set_ylabel("Rating diff")
fig.tight_layout()
plt.show()"""))

cells.append(code("""\
# Elo expected score distribution by outcome
fig, ax = plt.subplots(figsize=(8, 4))
for outcome_label, color in zip(["Black wins", "Draw", "White wins"],
                                 ["#e07b7b", "#aaaaaa", "#7baee0"]):
    subset = df_train[df_train["outcome_label"] == outcome_label]["elo_expected_score"]
    subset.plot.kde(ax=ax, label=outcome_label, color=color, linewidth=2)
ax.set_title("Elo expected score distribution by outcome")
ax.set_xlabel("Elo expected score (White)")
ax.legend()
fig.tight_layout()
plt.show()"""))

# ── 3. Feature Engineering ────────────────────────────────────────────────────
cells.append(md("""\
## 3. Feature Engineering

Features are grouped into three families — all computed from pre-game data only."""))

cells.append(code("""\
print(f"Feature matrix shape: {df_train[feature_cols].shape}")
print(f"\\nFeature groups:")
print(f"  Rating features     : white_rating, black_rating, rating_diff, rating_avg, abs_rating_diff, elo_expected_score")
print(f"  Tournament context  : round_number, white/black_tournament_points, tournament_points_diff")
print(f"  Title binary (×2)   : GM, IM, FM, WGM, WIM, CM, NM, WFM, WCM")
df_train[feature_cols].describe().round(2)"""))

cells.append(code("""\
# Feature correlation heatmap (numeric features only)
numeric_features = ["white_rating", "black_rating", "rating_diff", "rating_avg",
                    "abs_rating_diff", "elo_expected_score", "round_number",
                    "white_tournament_points", "black_tournament_points",
                    "tournament_points_diff", "outcome"]
corr = df_train[numeric_features].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, ax=ax)
ax.set_title("Feature correlations")
fig.tight_layout()
plt.show()"""))

# ── 4. Modeling ───────────────────────────────────────────────────────────────
cells.append(md("""\
## 4. Modeling & Evaluation

**Split strategy**: Train on February tournament, test on March tournament.
This is a strict temporal split — no data from the test period is ever seen during training.
Cross-validation within training uses round number as the split criterion (rounds 1–8 train, 9–11 val)."""))

cells.append(code("""\
from src.modeling import run_pipeline, plot_confusion_matrices, plot_feature_importance

output = run_pipeline(df_train, df_test, feature_cols)"""))

cells.append(code("""\
print("=== TRAIN performance (February) ===")
display(output["comparison_train"])

print("\\n=== TEST performance (March) ===")
display(output["comparison_test"])"""))

cells.append(code("""\
fig = plot_confusion_matrices(output["results_test"])
fig.suptitle("Confusion matrices — Test set (March)", fontsize=13, fontweight="bold", y=1.02)
plt.show()"""))

cells.append(code("""\
fig = plot_feature_importance(output["xgb_model"], feature_cols, top_n=15)
plt.show()"""))

cells.append(code("""\
# Logistic Regression coefficients — interpretability check
lr = output["lr_model"].named_steps["lr"]
coef_df = pd.DataFrame(
    lr.coef_,
    index=["black_wins", "draw", "white_wins"],
    columns=feature_cols,
).T.round(3)
print("Logistic Regression coefficients (top 10 by |white_wins| coef):")
coef_df.reindex(coef_df["white_wins"].abs().sort_values(ascending=False).index).head(10)"""))

# ── 5. Discussion ─────────────────────────────────────────────────────────────
cells.append(md("""\
## 5. Discussion & Next Steps

### Data split
The train/test split is **temporal**: February 2026 tournament → train, March 2026 tournament → test. This mirrors the production scenario where a model trained on past tournaments predicts future ones. A random split would leak within-tournament player form and inflate test metrics. Cross-validation within training is also round-based for the same reason.

### Model quality assessment

| Model | Test log-loss | Test accuracy |
|---|---|---|
| Elo baseline | 0.7352 | 69.7% |
| Logistic Regression | 0.7631 | 69.9% |
| XGBoost | 0.7824 | 70.4% |

**The Elo baseline is remarkably hard to beat.** Logistic Regression is actually *worse* on log-loss than the Elo baseline, meaning its probability estimates are less calibrated even though its accuracy is marginally higher. XGBoost achieves the best accuracy but its train log-loss (0.45) vs test log-loss (0.78) reveals significant overfitting — it memorizes training patterns that don't generalize across tournaments.

**Draw prediction is essentially unsolved.** Draws represent only ~8% of games and are nearly impossible to predict from pre-game information alone. All models collapse to near-zero F1 on draws in the test set. This is expected: whether two equally-rated players agree to a draw depends on tactical nuances of the position, not on who they are before the game starts.

**Rating difference dominates.** XGBoost's top feature by a large margin is `rating_diff`, confirming that the Elo formula already captures most of the predictive signal available before a game starts.

### What I would do next
- **Fetch player historical stats** via `/pub/player/{username}/stats` — blitz win rate, recent form over last 30 days, performance as White vs Black specifically
- **Head-to-head history** between the paired players (some matchups have well-established patterns)
- **More training data** — stack multiple Titled Tuesday tournaments rather than training on a single event (~2000 games is small for a 28-feature model)
- **Calibration analysis** — reliability diagrams to check if predicted probabilities match empirical frequencies; the Elo baseline likely needs isotonic regression calibration
- **Ordinal regression** — win > draw > loss has a natural ordering that multinomial logistic regression ignores
- **Opening repertoire features** — not the opening played, but each player's *tendency* (e.g., percentage of games as White that reach open games) can be computed from historical data without using the current game's moves"""))

nb.cells = cells
with open("notebook.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("notebook.ipynb written successfully")
