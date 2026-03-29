from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import urllib.request

from cindex.services.indexing.query import execute_sql_query
from cindex.services.indexing.query_templates import get_query_template


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a built-in starter SQL query")
    parser.add_argument("name", help="Starter query name")
    parser.add_argument("--db", default=".cindex/db.sqlite", help="Path to SQLite DB")
    parser.add_argument("--limit", type=int, default=25, help="Rows to print")
    parser.add_argument(
        "--api-url",
        default=None,
        help="If set, execute SQL through API /query/sql instead of direct SQLite",
    )
    return parser


def run_via_api(api_url: str, sql: str) -> tuple[list[str], list[dict[str, object]]]:
    payload = json.dumps({"sql": sql}).encode("utf-8")
    req = urllib.request.Request(
        f"{api_url.rstrip('/')}/query/sql",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return list(body.get("columns", [])), list(body.get("rows", []))


def main() -> int:
    args = build_parser().parse_args()
    template = get_query_template(args.name)
    if args.api_url:
        columns, rows = run_via_api(args.api_url, template.sql)
    else:
        columns, rows = execute_sql_query(Path(args.db), template.sql)

    print(f"template={template.name}")
    print(f"title={template.title}")
    print(f"rows={len(rows)}")

    if "symbol_kind" in columns:
        by_kind = Counter(str(row.get("symbol_kind", "")) for row in rows)
        for kind in sorted(by_kind):
            print(f"kind={kind} count={by_kind[kind]}")

    print("--- preview ---")
    for row in rows[: max(args.limit, 0)]:
        print("\t".join(str(row.get(col, "")) for col in columns))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
