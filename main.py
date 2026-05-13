"""Entry point for the chess outcome prediction pipeline."""

from src.utils import setup_logging
from src.data_collection import fetch_tournament_games
from src.features import build_features, get_feature_columns
from src.modeling import run_pipeline


def main() -> None:
    setup_logging()

    games_train = fetch_tournament_games("train")
    games_test  = fetch_tournament_games("test")

    df_train = build_features(games_train)
    df_test  = build_features(games_test)
    feature_cols = get_feature_columns(df_train)

    output = run_pipeline(df_train, df_test, feature_cols)

    print("\n=== TRAIN (February) ===")
    print(output["comparison_train"].to_string())
    print("\n=== TEST (March) ===")
    print(output["comparison_test"].to_string())


if __name__ == "__main__":
    main()
