from src.utils import setup_logging
from src.data_collection import fetch_tournament_games
from src.features import build_features, get_feature_columns

setup_logging()

games = fetch_tournament_games("train")
df = build_features(games)
feature_cols = get_feature_columns(df)

print(f"\n--- Feature matrix ---")
print(f"Shape          : {df.shape}")
print(f"Feature cols   : {len(feature_cols)}")
print(f"\nOutcome dist (0=black, 1=draw, 2=white):")
print(df["outcome"].value_counts().sort_index().rename({0: "black_wins", 1: "draw", 2: "white_wins"}))

print(f"\nRating features (first 3 rows):")
print(df[["white_rating", "black_rating", "rating_diff", "elo_expected_score"]].head(3).to_string())

print(f"\nTournament points at round 5 (should be > 0 for most players):")
r5 = df[df["round_number"] == 5][["white_username", "white_tournament_points", "black_tournament_points"]].head(5)
print(r5.to_string(index=False))

print(f"\nTournament points at round 1 (must all be 0.0):")
r1 = df[df["round_number"] == 1]["white_tournament_points"]
assert r1.eq(0.0).all(), "ERROR: round 1 has non-zero tournament points!"
print("All round-1 tournament points are 0.0 -- OK")

print(f"\nSample title features:")
title_cols = [c for c in df.columns if c.startswith("white_is_") or c.startswith("black_is_")]
print(df[title_cols].sum().to_string())
