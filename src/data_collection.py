"""API fetching and caching for Chess.com Titled Tuesday tournaments."""

import logging
import time
from pathlib import Path
from typing import Any

import requests

from src.utils import REQUEST_DELAY, USER_AGENT, get, load_cache, save_cache

logger = logging.getLogger(__name__)

API_BASE = "https://api.chess.com"
DATA_RAW = Path("data/raw")

TOURNAMENTS = {
    "train": "titled-tuesday-blitz-february-10-2026-6221327",
    "test": "titled-tuesday-blitz-march-10-2026-6277141",
}


def _make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def _fetch_tournament(tournament_id: str, session: requests.Session) -> dict[str, Any]:
    """Fetch and cache the top-level tournament metadata."""
    cache_path = DATA_RAW / tournament_id / "tournament.json"
    cached = load_cache(cache_path)
    if cached is not None:
        logger.info("Tournament cache hit: %s", tournament_id)
        return cached

    url = f"{API_BASE}/pub/tournament/{tournament_id}"
    logger.info("Fetching tournament: %s", url)
    data = get(url, session)
    save_cache(cache_path, data)
    time.sleep(REQUEST_DELAY)
    return data


def _fetch_round(round_url: str, round_idx: int, tournament_id: str, session: requests.Session) -> dict[str, Any]:
    """Fetch and cache a round endpoint (which may list groups)."""
    cache_path = DATA_RAW / tournament_id / f"round_{round_idx:02d}.json"
    cached = load_cache(cache_path)
    if cached is not None:
        return cached

    logger.info("Fetching round %d: %s", round_idx, round_url)
    data = get(round_url, session)
    save_cache(cache_path, data)
    time.sleep(REQUEST_DELAY)
    return data


def _fetch_group(group_url: str, round_idx: int, group_idx: int, tournament_id: str, session: requests.Session) -> dict[str, Any]:
    """Fetch and cache a group endpoint that contains the actual games."""
    cache_path = DATA_RAW / tournament_id / f"round_{round_idx:02d}_group_{group_idx:02d}.json"
    cached = load_cache(cache_path)
    if cached is not None:
        return cached

    logger.info("Fetching round %d group %d: %s", round_idx, group_idx, group_url)
    data = get(group_url, session)
    save_cache(cache_path, data)
    time.sleep(REQUEST_DELAY)
    return data


def _parse_game(game: dict[str, Any], round_number: int) -> dict[str, Any] | None:
    """
    Extract a flat record from a raw game dict.

    Returns None if the game is missing essential fields.
    """
    white = game.get("white", {})
    black = game.get("black", {})

    white_username = white.get("username")
    black_username = black.get("username")
    white_result = white.get("result")
    white_rating = white.get("rating")
    black_rating = black.get("rating")

    if not all([white_username, black_username, white_result, white_rating, black_rating]):
        logger.debug("Skipping game with missing fields: %s", game.get("url"))
        return None

    return {
        "round_number": round_number,
        "white_username": white_username,
        "black_username": black_username,
        "white_rating": int(white_rating),
        "black_rating": int(black_rating),
        "white_result": white_result,
        "black_result": black.get("result"),
        "game_url": game.get("url", ""),
        "time_control": game.get("time_control", ""),
        # Store raw pgn for reference only — never used as a feature
        "pgn": game.get("pgn", ""),
    }


def fetch_player_titles(usernames: list[str], session: requests.Session | None = None) -> dict[str, str]:
    """
    Fetch and cache player profiles for a list of usernames.

    Returns a dict mapping lowercase username → title string (e.g. 'GM', 'IM', '').
    """
    if session is None:
        session = _make_session()

    titles: dict[str, str] = {}
    total = len(usernames)

    for i, username in enumerate(usernames, start=1):
        key = username.lower()
        cache_path = DATA_RAW / "players" / f"{key}.json"
        data = load_cache(cache_path)
        if data is None:
            url = f"{API_BASE}/pub/player/{key}"
            logger.info("Fetching player profile (%d/%d): %s", i, total, url)
            try:
                data = get(url, session)
                save_cache(cache_path, data)
            except requests.RequestException:
                logger.warning("Could not fetch profile for %s", username)
                data = {}
            time.sleep(REQUEST_DELAY)
        titles[key] = data.get("title", "") or ""

    return titles


def fetch_tournament_games(split: str) -> list[dict[str, Any]]:
    """
    Fetch all games for a given split ('train' or 'test').

    Returns a flat list of parsed game records, one per game.
    """
    if split not in TOURNAMENTS:
        raise ValueError(f"split must be one of {list(TOURNAMENTS)}, got {split!r}")

    tournament_id = TOURNAMENTS[split]
    session = _make_session()

    tournament_data = _fetch_tournament(tournament_id, session)

    round_urls: list[str] = []
    # The API returns rounds nested under a "rounds" key as a list of URLs
    for rd in tournament_data.get("rounds", []):
        if isinstance(rd, str):
            round_urls.append(rd)
        elif isinstance(rd, dict):
            url = rd.get("@id") or rd.get("url", "")
            if url:
                round_urls.append(url)

    logger.info("Found %d rounds for tournament %s", len(round_urls), tournament_id)

    games: list[dict[str, Any]] = []

    for round_idx, round_url in enumerate(round_urls, start=1):
        round_data = _fetch_round(round_url, round_idx, tournament_id, session)

        # A round may directly contain games or list group URLs
        if "games" in round_data:
            raw_games = round_data["games"]
            for game in raw_games:
                parsed = _parse_game(game, round_idx)
                if parsed:
                    games.append(parsed)

        elif "groups" in round_data:
            group_urls: list[str] = []
            for grp in round_data["groups"]:
                if isinstance(grp, str):
                    group_urls.append(grp)
                elif isinstance(grp, dict):
                    url = grp.get("@id") or grp.get("url", "")
                    if url:
                        group_urls.append(url)

            for group_idx, group_url in enumerate(group_urls, start=1):
                group_data = _fetch_group(group_url, round_idx, group_idx, tournament_id, session)
                for game in group_data.get("games", []):
                    parsed = _parse_game(game, round_idx)
                    if parsed:
                        games.append(parsed)
        else:
            logger.warning("Round %d has neither 'games' nor 'groups' keys", round_idx)

    logger.info("Collected %d raw games for split '%s'", len(games), split)

    # Enrich with player titles (fetched from profiles, cached)
    all_usernames = list({g["white_username"] for g in games} | {g["black_username"] for g in games})
    logger.info("Fetching profiles for %d unique players", len(all_usernames))
    titles = fetch_player_titles(all_usernames, session)

    for game in games:
        game["white_title"] = titles.get(game["white_username"].lower(), "")
        game["black_title"] = titles.get(game["black_username"].lower(), "")

    logger.info("Collected %d games for split '%s'", len(games), split)
    return games
