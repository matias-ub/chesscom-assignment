"""Model training and evaluation for chess outcome prediction.

Train/test split: temporal (February → March).
CV split within training: rounds 1-8 train, rounds 9-11 validation.
"""

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE = 42
CV_TRAIN_ROUNDS = list(range(1, 9))   # rounds 1-8
CV_VAL_ROUNDS = list(range(9, 12))    # rounds 9-11
CLASS_NAMES = ["black_wins", "draw", "white_wins"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Elo baseline
# ---------------------------------------------------------------------------

def _tune_elo_thresholds(elo_scores: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    """Grid-search the best (low, high) thresholds on the validation fold."""
    best_acc = -1.0
    best = (0.4, 0.6)
    for low in np.arange(0.30, 0.50, 0.02):
        for high in np.arange(0.50, 0.72, 0.02):
            if low >= high:
                continue
            preds = np.where(elo_scores > high, 2, np.where(elo_scores < low, 0, 1))
            acc = accuracy_score(y_true, preds)
            if acc > best_acc:
                best_acc = acc
                best = (round(low, 2), round(high, 2))
    return best


def _elo_proba(elo_scores: np.ndarray, draw_rate: float) -> np.ndarray:
    """Convert scalar Elo expected scores to 3-class probability matrix."""
    p_draw = np.full(len(elo_scores), draw_rate)
    p_white = (1 - draw_rate) * elo_scores
    p_black = (1 - draw_rate) * (1 - elo_scores)
    return np.column_stack([p_black, p_draw, p_white])


def train_elo_baseline(
    df_train: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, Any]:
    """Fit the Elo baseline by tuning thresholds on the CV validation fold."""
    cv_train = df_train[df_train["round_number"].isin(CV_TRAIN_ROUNDS)]
    cv_val = df_train[df_train["round_number"].isin(CV_VAL_ROUNDS)]

    low, high = _tune_elo_thresholds(
        cv_val["elo_expected_score"].values,
        cv_val["outcome"].values,
    )
    draw_rate = float((df_train["outcome"] == 1).mean())

    logger.info("Elo baseline thresholds: low=%.2f high=%.2f | draw_rate=%.3f", low, high, draw_rate)
    return {"low": low, "high": high, "draw_rate": draw_rate}


def predict_elo(
    df: pd.DataFrame,
    params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (class predictions, probability matrix) for the Elo baseline."""
    elo = df["elo_expected_score"].values
    low, high, draw_rate = params["low"], params["high"], params["draw_rate"]
    preds = np.where(elo > high, 2, np.where(elo < low, 0, 1))
    proba = _elo_proba(elo, draw_rate)
    return preds, proba


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------

def train_logistic(
    df_train: pd.DataFrame,
    feature_cols: list[str],
) -> Pipeline:
    """Train a multinomial logistic regression with standard scaling."""
    X = df_train[feature_cols].values
    y = df_train["outcome"].values

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=RANDOM_STATE,
            C=1.0,
        )),
    ])
    pipe.fit(X, y)
    logger.info("Logistic regression trained on %d samples", len(y))
    return pipe


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

def train_xgboost(
    df_train: pd.DataFrame,
    feature_cols: list[str],
) -> XGBClassifier:
    """Train an XGBoost classifier with temporal CV for early stopping reference."""
    cv_train = df_train[df_train["round_number"].isin(CV_TRAIN_ROUNDS)]
    cv_val = df_train[df_train["round_number"].isin(CV_VAL_ROUNDS)]

    X_cv_train = cv_train[feature_cols].values
    y_cv_train = cv_train["outcome"].values
    X_cv_val = cv_val[feature_cols].values
    y_cv_val = cv_val["outcome"].values

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        max_depth=4,
        n_estimators=200,
        learning_rate=0.1,
        min_child_weight=5,
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
        verbosity=0,
    )
    model.fit(
        X_cv_train, y_cv_train,
        eval_set=[(X_cv_val, y_cv_val)],
        verbose=False,
    )

    # Refit on full training set with the same hyperparameters
    X_train = df_train[feature_cols].values
    y_train = df_train["outcome"].values
    model.fit(X_train, y_train, verbose=False)

    logger.info("XGBoost trained on %d samples", len(y_train))
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, Any]:
    """Compute all evaluation metrics for one model."""
    ll = log_loss(y_true, y_proba)
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    logger.info("%s — log_loss=%.4f  accuracy=%.4f", name, ll, acc)
    return {
        "name": name,
        "log_loss": round(ll, 4),
        "accuracy": round(acc, 4),
        "report": report,
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def plot_confusion_matrices(results: list[dict[str, Any]]) -> plt.Figure:
    """Return a figure with one confusion matrix heatmap per model."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        cm = res["confusion_matrix"]
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
        )
        ax.set_title(res["name"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.tight_layout()
    return fig


def plot_feature_importance(model: XGBClassifier, feature_cols: list[str], top_n: int = 15) -> plt.Figure:
    """Return a horizontal bar chart of XGBoost feature importances."""
    importances = model.feature_importances_
    idx = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(np.array(feature_cols)[idx], importances[idx], color="steelblue")
    ax.set_title(f"XGBoost — top {top_n} feature importances")
    ax.set_xlabel("Importance (gain)")
    fig.tight_layout()
    return fig


def build_comparison_table(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Return a DataFrame comparing log-loss and accuracy across all models."""
    rows = []
    for res in results:
        rep = res["report"]
        rows.append({
            "Model": res["name"],
            "Log-loss": res["log_loss"],
            "Accuracy": res["accuracy"],
            "F1 black_wins": round(rep["black_wins"]["f1-score"], 4),
            "F1 draw": round(rep["draw"]["f1-score"], 4),
            "F1 white_wins": round(rep["white_wins"]["f1-score"], 4),
        })
    return pd.DataFrame(rows).set_index("Model")


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, Any]:
    """
    Train all models on df_train, evaluate on df_test.

    Returns a dict with trained models, evaluation results, and helper objects.
    """
    y_test = df_test["outcome"].values

    # 1. Elo baseline
    elo_params = train_elo_baseline(df_train, feature_cols)
    elo_pred, elo_proba = predict_elo(df_test, elo_params)
    elo_result = evaluate("Elo baseline", y_test, elo_pred, elo_proba)

    # 2. Logistic Regression
    lr_model = train_logistic(df_train, feature_cols)
    X_test = df_test[feature_cols].values
    lr_proba = lr_model.predict_proba(X_test)
    lr_pred = lr_model.predict(X_test)
    lr_result = evaluate("Logistic Regression", y_test, lr_pred, lr_proba)

    # 3. XGBoost
    xgb_model = train_xgboost(df_train, feature_cols)
    xgb_proba = xgb_model.predict_proba(X_test)
    xgb_pred = xgb_model.predict(X_test)
    xgb_result = evaluate("XGBoost", y_test, xgb_pred, xgb_proba)

    # Also evaluate on train to surface overfitting
    X_train = df_train[feature_cols].values
    y_train = df_train["outcome"].values

    _, elo_proba_train = predict_elo(df_train, elo_params)
    elo_pred_train = np.where(
        df_train["elo_expected_score"].values > elo_params["high"], 2,
        np.where(df_train["elo_expected_score"].values < elo_params["low"], 0, 1),
    )
    elo_train = evaluate("Elo baseline", y_train, elo_pred_train, elo_proba_train)
    lr_train = evaluate("Logistic Regression", y_train, lr_model.predict(X_train), lr_model.predict_proba(X_train))
    xgb_train = evaluate("XGBoost", y_train, xgb_model.predict(X_train), xgb_model.predict_proba(X_train))

    results_test = [elo_result, lr_result, xgb_result]
    results_train = [elo_train, lr_train, xgb_train]

    comparison_test = build_comparison_table(results_test)
    comparison_train = build_comparison_table(results_train)

    return {
        "results_test": results_test,
        "results_train": results_train,
        "comparison_test": comparison_test,
        "comparison_train": comparison_train,
        "elo_params": elo_params,
        "lr_model": lr_model,
        "xgb_model": xgb_model,
        "feature_cols": feature_cols,
    }
