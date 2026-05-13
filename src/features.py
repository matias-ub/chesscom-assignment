"""Feature engineering for chess game outcome prediction.

All features use only information available BEFORE the game starts.
Post-game data (PGN, moves, ECO, end_time, duration) is never used.
"""

import logging
from collections import defaultdict
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Draw result strings from the Chess.com API
DRAW_RESULTS = {"agreed", "stalemate", "repetition", "insufficient", "50move", "timevsinsufficient"}

# Titles to encode as binary features
TITLE_FEATURES = ["GM", "IM", "FM", "WGM", "WIM", "CM", "NM", "WFM", "WCM"]


def map_outcome(white_result: str) -> int:
    """Map white's result string to a 3-class label.

    Returns 2 (white wins), 1 (draw), or 0 (black wins).
    """
    if white_result == "win":
        return 2
    if white_result in DRAW_RESULTS:
        return 1
    return 0


def _elo_expected(white_rating: int, black_rating: int) -> float:
    """Expected score for white using the standard Elo formula."""
    return 1.0 / (1.0 + 10 ** ((black_rating - white_rating) / 400))


def _add_tournament_points(games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Compute cumulative tournament points per player BEFORE each round.

    Mutates each game dict in-place, adding:
      - white_tournament_points
      - black_tournament_points

    Games must already be sorted by round_number (ascending).
    """
    # Group games by round so we can process round-by-round
    rounds: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for game in games:
        rounds[game["round_number"]].append(game)

    cumulative: dict[str, float] = defaultdict(float)

    for round_num in sorted(rounds.keys()):
        round_games = rounds[round_num]

        # Assign points BEFORE this round runs
        for game in round_games:
            game["white_tournament_points"] = cumulative[game["white_username"].lower()]
            game["black_tournament_points"] = cumulative[game["black_username"].lower()]

        # Update cumulative totals with this round's results
        for game in round_games:
            outcome = map_outcome(game["white_result"])
            w = game["white_username"].lower()
            b = game["black_username"].lower()
            if outcome == 2:       # white wins
                cumulative[w] += 1.0
            elif outcome == 1:     # draw
                cumulative[w] += 0.5
                cumulative[b] += 0.5
            else:                  # black wins
                cumulative[b] += 1.0

    return games


def _add_schedule_strength(games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Compute average opponent rating faced per player BEFORE each round.

    Mutates each game dict in-place, adding:
      - white_avg_opp_rating
      - black_avg_opp_rating

    Round 1 has no prior opponents; falls back to the player's own rating.
    Games must already be sorted by round_number (ascending).
    """
    rounds: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for game in games:
        rounds[game["round_number"]].append(game)

    opp_ratings: dict[str, list[int]] = defaultdict(list)

    for round_num in sorted(rounds.keys()):
        round_games = rounds[round_num]

        for game in round_games:
            w = game["white_username"].lower()
            b = game["black_username"].lower()
            game["white_avg_opp_rating"] = (
                sum(opp_ratings[w]) / len(opp_ratings[w]) if opp_ratings[w] else game["white_rating"]
            )
            game["black_avg_opp_rating"] = (
                sum(opp_ratings[b]) / len(opp_ratings[b]) if opp_ratings[b] else game["black_rating"]
            )

        # Update opponent history after assigning features
        for game in round_games:
            w = game["white_username"].lower()
            b = game["black_username"].lower()
            opp_ratings[w].append(game["black_rating"])
            opp_ratings[b].append(game["white_rating"])

    return games


def build_features(games: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Transform a list of raw game dicts into a feature DataFrame.

    Each row is one game. The 'outcome' column is the target variable.
    Features only use pre-game information.
    """
    games = sorted(games, key=lambda g: g["round_number"])
    _add_tournament_points(games)

    rows = []
    for game in games:
        wr = game["white_rating"]
        br = game["black_rating"]

        row: dict[str, Any] = {
            # --- identifiers (not used as model features) ---
            "white_username": game["white_username"],
            "black_username": game["black_username"],
            "game_url": game["game_url"],

            # --- core rating features ---
            "white_rating": wr,
            "black_rating": br,
            "rating_diff": wr - br,
            "rating_avg": (wr + br) / 2,
            "abs_rating_diff": abs(wr - br),
            "elo_expected_score": _elo_expected(wr, br),

            # --- tournament context ---
            "round_number": game["round_number"],
            "white_tournament_points": game["white_tournament_points"],
            "black_tournament_points": game["black_tournament_points"],
            "tournament_points_diff": game["white_tournament_points"] - game["black_tournament_points"],

            # --- target ---
            "outcome": map_outcome(game["white_result"]),
        }

        # --- title binary features ---
        wt = game.get("white_title", "").upper()
        bt = game.get("black_title", "").upper()
        for title in TITLE_FEATURES:
            row[f"white_is_{title}"] = int(wt == title)
            row[f"black_is_{title}"] = int(bt == title)

        rows.append(row)

    df = pd.DataFrame(rows)
    logger.info(
        "Built feature matrix: %d rows x %d cols | outcome dist: %s",
        len(df),
        df.shape[1],
        df["outcome"].value_counts().to_dict(),
    )
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of columns to use as model input features (excludes identifiers and target)."""
    exclude = {"white_username", "black_username", "game_url", "outcome"}
    return [c for c in df.columns if c not in exclude]
