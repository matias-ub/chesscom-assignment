from src.utils import setup_logging
from src.data_collection import fetch_tournament_games
from src.features import build_features, get_feature_columns
from src.modeling import run_pipeline

setup_logging()

print("Loading data...")
games_train = fetch_tournament_games("train")
games_test = fetch_tournament_games("test")

print("Building features...")
df_train = build_features(games_train)
df_test = build_features(games_test)
feature_cols = get_feature_columns(df_train)

print(f"\nTrain: {len(df_train)} games | Test: {len(df_test)} games")
print(f"Features: {len(feature_cols)}")

print("\nTraining models...")
output = run_pipeline(df_train, df_test, feature_cols)

print("\n--- TRAIN results (February) ---")
print(output["comparison_train"].to_string())

print("\n--- TEST results (March) ---")
print(output["comparison_test"].to_string())

print("\n--- Elo baseline thresholds ---")
print(output["elo_params"])

print("\n--- XGBoost top 5 features ---")
import numpy as np
imp = output["xgb_model"].feature_importances_
top5_idx = np.argsort(imp)[-5:][::-1]
for i in top5_idx:
    print(f"  {feature_cols[i]:<35} {imp[i]:.4f}")
