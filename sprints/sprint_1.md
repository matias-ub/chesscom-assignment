# Sprint 1 — Data Collection (Phase 1)

## Goal
Fetch, parse, and cache all game data from the two Titled Tuesday tournaments used for training and testing.

## What was built

### `src/utils.py`
Shared helpers used across all modules:
- `setup_logging()` — consistent log format with timestamps
- `get(url, session, retries)` — HTTP GET with exponential-backoff retries and the required `User-Agent` header
- `load_cache(path)` / `save_cache(path, data)` — read/write JSON cache files, creates parent dirs automatically

### `src/data_collection.py`
Main data fetching module:
- `fetch_tournament_games(split)` — fetches all games for `'train'` (February) or `'test'` (March) tournament
  - Traverses the API hierarchy: tournament → rounds → groups → games
  - Handles both `games` and `groups` response shapes (the API returns groups for these tournaments)
  - Caches every response as JSON under `data/raw/<tournament-id>/`
  - Parses each game into a flat record with: round_number, usernames, ratings, results, time_control, game_url
- `fetch_player_titles(usernames, session)` — fetches `/pub/player/{username}` profiles and caches them under `data/raw/players/`
  - Called automatically from `fetch_tournament_games` after collecting all games
  - Enriches each game record with `white_title` and `black_title`

### `validate_phase1.py`
Quick smoke test — runs entirely from cache (no HTTP calls after first run):
```bash
uv run python validate_phase1.py
```

## Validation results (February / train tournament)
| Metric | Value |
|---|---|
| Total games | 2025 |
| Rounds | 1 – 11 |
| Unique players | 450 |
| Title distribution | FM: 1035, GM: 869, CM: 741, IM: 637, NM: 439, WFM: 127, WIM: 86, WCM: 86, WGM: 30 |

## API notes
- All requests use `User-Agent: ChessPrediction/1.0 (take-home exercise)`
- 0.5s delay between requests to be respectful
- Tournament data: titles are **not** included in game objects — they require a separate `GET /pub/player/{username}` call (fetched once and cached)
- Extra titles beyond the spec's GM/IM/FM/WGM/WIM: CM, NM, WFM, WCM also appear in the data — all captured

## Cache structure
```
data/raw/
├── titled-tuesday-blitz-february-10-2026-6221327/
│   ├── tournament.json
│   ├── round_01.json ... round_11.json
│   └── round_01_group_01.json ... round_11_group_01.json
├── titled-tuesday-blitz-march-10-2026-6277141/   (fetched on first test run)
│   └── ...
└── players/
    └── {username}.json   (~450 files)
```

## What's next
Phase 2 — Feature Engineering (`src/features.py`): rating features, Elo expected score, cumulative tournament points, title binary encoding, and target variable mapping.
