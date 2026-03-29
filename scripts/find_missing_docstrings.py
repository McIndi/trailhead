from __future__ import annotations

import collections
import sqlite3
from pathlib import Path

Q_MODULES = """
SELECT
  'module' AS symbol_kind,
  json_extract(v.properties_json, '$.name') AS name,
  json_extract(v.properties_json, '$.path') AS path,
  NULL AS line
FROM vertices v
WHERE v.label = 'module'
  AND (
    json_extract(v.properties_json, '$.docstring') IS NULL
    OR TRIM(json_extract(v.properties_json, '$.docstring')) = ''
  )
ORDER BY path, name;
"""

Q_FUNCTIONS = """
SELECT
  'function' AS symbol_kind,
  json_extract(v.properties_json, '$.name') AS name,
  json_extract(v.properties_json, '$.path') AS path,
  json_extract(v.properties_json, '$.line') AS line
FROM vertices v
WHERE v.label = 'function'
  AND NOT EXISTS (
    SELECT 1 FROM edges e
    WHERE e.label = 'has_method' AND e.in_v_id = v.id
  )
  AND (
    json_extract(v.properties_json, '$.docstring') IS NULL
    OR TRIM(json_extract(v.properties_json, '$.docstring')) = ''
  )
ORDER BY path, line, name;
"""

Q_METHODS = """
SELECT
  'method' AS symbol_kind,
  json_extract(v.properties_json, '$.name') AS name,
  json_extract(v.properties_json, '$.path') AS path,
  json_extract(v.properties_json, '$.line') AS line
FROM vertices v
WHERE v.label = 'function'
  AND EXISTS (
    SELECT 1 FROM edges e
    WHERE e.label = 'has_method' AND e.in_v_id = v.id
  )
  AND (
    json_extract(v.properties_json, '$.docstring') IS NULL
    OR TRIM(json_extract(v.properties_json, '$.docstring')) = ''
  )
ORDER BY path, line, name;
"""


def main() -> int:
    db = Path(".cindex") / "db.sqlite"
    if not db.exists():
        print(f"database not found: {db}")
        return 1

    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        rows = [
            *conn.execute(Q_MODULES).fetchall(),
            *conn.execute(Q_FUNCTIONS).fetchall(),
            *conn.execute(Q_METHODS).fetchall(),
        ]

    print(f"total_missing={len(rows)}")
    counts = collections.Counter(row["symbol_kind"] for row in rows)
    for kind in sorted(counts):
        print(f"kind={kind} count={counts[kind]}")

    print("--- rows ---")
    for row in rows:
        kind = row["symbol_kind"] or ""
        path = row["path"] or ""
        line = row["line"] or ""
        name = row["name"] or ""
        print(f"{kind}\t{path}\t{line}\t{name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
