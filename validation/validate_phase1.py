from src.utils import setup_logging
from src.data_collection import fetch_tournament_games
from collections import Counter

setup_logging()
games = fetch_tournament_games("train")

rounds = sorted(set(g["round_number"] for g in games))
players = set(g["white_username"] for g in games) | set(g["black_username"] for g in games)
titles = [g["white_title"] for g in games] + [g["black_title"] for g in games]
title_dist = Counter(t for t in titles if t)

print(f"Total games   : {len(games)}")
print(f"Rounds        : {rounds}")
print(f"Unique players: {len(players)}")
print(f"Title dist    : {dict(title_dist)}")
print(f"Sample game   : {games[0]['white_username']} ({games[0]['white_title']}) vs {games[0]['black_username']} ({games[0]['black_title']})")
print("OK - all from cache, no HTTP calls")
