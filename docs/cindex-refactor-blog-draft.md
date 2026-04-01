# Refactoring cindex With cindex: A CLI-Guided Session

## Why this experiment

We wanted to answer two questions in one pass:

1. Can cindex identify high-value refactors in its own codebase?
2. Is the CLI workflow usable enough to narrate as a reproducible engineering story?

This session focused on the top two opportunities from an earlier report:

- Decompose `create_app` in the API server.
- Deduplicate sqlite-vector extension loading helpers.

## Environment assumptions

- Active server and live indexer running on `localhost:8000`.
- SQLite index used for this run: `src/.cindex/db.sqlite`.
- Python test command used for verification:

```powershell
c:/Users/cliff/projects/mcindi/cindex/.venv/Scripts/python.exe -m pytest tests/test_server.py tests/test_query.py -q
```

## Command log: intelligence -> response

### 1) Baseline complexity snapshot

```powershell
C:\Users\cliff\projects\mcindi\cindex\.venv\Scripts\cindex query sql --sqlite-db src/.cindex/db.sqlite --sql "SELECT COUNT(*) AS total_functions FROM vertices WHERE label='function';"
```

Result:

- `total_functions = 203`

```powershell
C:\Users\cliff\projects\mcindi\cindex\.venv\Scripts\cindex query sql --sqlite-db src/.cindex/db.sqlite --sql "SELECT CASE WHEN CAST(json_extract(properties_json, '$.complexity') AS INTEGER) >= 15 THEN '>=15' WHEN CAST(json_extract(properties_json, '$.complexity') AS INTEGER) BETWEEN 10 AND 14 THEN '10-14' WHEN CAST(json_extract(properties_json, '$.complexity') AS INTEGER) BETWEEN 5 AND 9 THEN '5-9' ELSE '1-4' END AS band, COUNT(*) AS count FROM vertices WHERE label='function' AND CAST(json_extract(properties_json, '$.complexity') AS INTEGER) >= 1 GROUP BY band ORDER BY CASE band WHEN '>=15' THEN 1 WHEN '10-14' THEN 2 WHEN '5-9' THEN 3 ELSE 4 END;"
```

Result:

- `>=15: 3`
- `10-14: 15`
- `5-9: 47`
- `1-4: 138`

Response:

- Proceed with first two planned refactors.

### 2) Verify `create_app` hotspot status

```powershell
C:\Users\cliff\projects\mcindi\cindex\.venv\Scripts\cindex query sql --sqlite-db src/.cindex/db.sqlite --sql "SELECT json_extract(properties_json, '$.name') AS name, json_extract(properties_json, '$.path') AS path, json_extract(properties_json, '$.line') AS line, CAST(json_extract(properties_json, '$.complexity') AS INTEGER) AS complexity FROM vertices WHERE label='function' AND json_extract(properties_json, '$.name')='create_app' AND json_extract(properties_json, '$.path') LIKE '%server%app.py';"
```

Result:

- `create_app` complexity is now `5` at `server/app.py:285`.

```powershell
C:\Users\cliff\projects\mcindi\cindex\.venv\Scripts\cindex query sql --sqlite-db src/.cindex/db.sqlite --sql "SELECT COUNT(*) AS server_app_functions FROM vertices WHERE label='function' AND json_extract(properties_json, '$.path') LIKE '%server%app.py';"
```

Result:

- `server_app_functions = 19` (module-level handlers now counted directly instead of closure-nested under one large factory body).

Response:

- Keep decomposition architecture: `AppConfig` + module-level handlers + route registration helper.

### 3) Verify sqlite-vector helper deduplication

```powershell
C:\Users\cliff\projects\mcindi\cindex\.venv\Scripts\cindex query sql --sqlite-db src/.cindex/db.sqlite --sql "SELECT json_extract(properties_json, '$.name') AS name, json_extract(properties_json, '$.path') AS path, json_extract(properties_json, '$.line') AS line, CAST(json_extract(properties_json, '$.complexity') AS INTEGER) AS complexity FROM vertices WHERE label='function' AND json_extract(properties_json, '$.name') IN ('_load_sqlite_vector_extension', '_try_load_sqlite_vector_extension') ORDER BY name;"
```

Result:

- `_load_sqlite_vector_extension` in `sqlite_store.py` remains complexity `10` (real implementation).
- `_try_load_sqlite_vector_extension` in `sqlite_store.py` is complexity `1` (compat alias).

Response:

- Consider dedup complete for MVP: one real loader implementation.

### 4) Semantic UX check while refactoring

```powershell
C:\Users\cliff\projects\mcindi\cindex\.venv\Scripts\cindex query similar "sqlite vector extension loading helper" --sqlite-db src/.cindex/db.sqlite -k 5 --output json
```

Result (top hits included):

- `_load_sqlite_vector_extension`
- `_try_load_sqlite_vector_extension`
- `_try_initialize_sqlite_vector`

Response:

- Confirms `query similar` is practical for finding neighboring refactor targets and not just exact names.

## What changed in code

- `src/cindex/server/app.py`
  - Reduced `create_app` complexity by lifting route handlers to module scope.
  - Added `AppConfig` state object and centralized route registration.
- `src/cindex/services/indexing/sqlite_store.py`
  - Consolidated sqlite-vector loader logic in `_load_sqlite_vector_extension`.
  - Kept `_try_load_sqlite_vector_extension` as a compatibility alias.
- `src/cindex/services/indexing/query.py`
  - Removed duplicate extension loading code and reused shared loader.
- `src/cindex/services/indexing/graph_query.py`
  - Restored JSON extraction in SQL for compatibility with simple test schemas.
- `src/cindex/cli/commands/serve.py`
  - Preserved expected argument behavior while still using resolved paths for model consistency checks.

## Validation

Focused suite passed after changes:

- `32 passed in 6.80s` for `tests/test_server.py` and `tests/test_query.py`.

## Usability notes for the CLI

What worked well:

- SQL mode made hotspot verification fast and explicit.
- Semantic mode quickly found related helper functions by intent phrase.
- Re-running the same command pattern after edits gave an easy before/after loop.

What to improve:

- Default DB path confusion: in this workspace, index lived at `src/.cindex/db.sqlite`, while CLI default looked under repo root (`.cindex/db.sqlite`).
- For blog/tutorial readability, showing the active DB path prominently at query time would reduce friction.

## Before/after highlight

- `create_app`: complexity dropped from `33` to `5`.
- sqlite-vector loader duplication: two complexity-10 implementations -> one complexity-10 implementation + one complexity-1 alias.

## Next likely target

From the latest top complexity list, likely next candidates are:

- `cli/commands/index.py::run` (19)
- adapter `_visit` methods in `javascript.py` and `php.py` (18-19)

The first is likely the better next move for behavior-safe simplification.
