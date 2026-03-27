"""Graph-oriented query helpers for SQLite-backed code indexes."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any



def search_vertices(
    db_path: Path,
    *,
    name: str | None = None,
    label: str | None = None,
    path_contains: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Find vertices by simple property filters."""
    if limit < 1:
        raise ValueError("limit must be at least 1")

    clauses = ["1 = 1"]
    params: list[Any] = []
    if name:
        clauses.append("json_extract(properties_json, '$.name') LIKE ?")
        params.append(f"%{name}%")
    if label:
        clauses.append("label = ?")
        params.append(label)
    if path_contains:
        clauses.append("COALESCE(json_extract(properties_json, '$.path'), '') LIKE ?")
        params.append(f"%{path_contains}%")

    sql = f"""
    SELECT id, label, properties_json
    FROM vertices
    WHERE {' AND '.join(clauses)}
    ORDER BY label, COALESCE(json_extract(properties_json, '$.path'), ''), COALESCE(json_extract(properties_json, '$.line'), 0), COALESCE(json_extract(properties_json, '$.name'), id)
    LIMIT {limit}
    """

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return [_vertex_from_row(row) for row in conn.execute(sql, tuple(params)).fetchall()]



def traverse_graph(
    db_path: Path,
    *,
    vertex_id: str,
    direction: str = "both",
    depth: int = 1,
    edge_labels: list[str] | None = None,
    max_vertices: int = 100,
) -> dict[str, Any]:
    """Traverse outward from a vertex id and return a small subgraph."""
    if direction not in {"out", "in", "both"}:
        raise ValueError("direction must be one of: out, in, both")
    if depth < 1:
        raise ValueError("depth must be at least 1")
    if max_vertices < 1:
        raise ValueError("max_vertices must be at least 1")

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        center_row = conn.execute(
            "SELECT id, label, properties_json FROM vertices WHERE id = ?",
            (vertex_id,),
        ).fetchone()
        if center_row is None:
            raise ValueError(f"Unknown vertex id: {vertex_id}")

        visited_vertices = {vertex_id}
        frontier = {vertex_id}
        edge_rows: dict[str, dict[str, Any]] = {}

        for _ in range(depth):
            if not frontier or len(visited_vertices) >= max_vertices:
                break

            rows = _fetch_edges(conn, frontier, direction=direction, edge_labels=edge_labels)
            next_frontier: set[str] = set()
            for row in rows:
                edge_rows[str(row["id"])] = {
                    "edge_id": str(row["id"]),
                    "label": str(row["label"]),
                    "out_v_id": str(row["out_v_id"]),
                    "in_v_id": str(row["in_v_id"]),
                    "properties": json.loads(row["properties_json"]),
                }
                for candidate in (str(row["out_v_id"]), str(row["in_v_id"])):
                    if candidate not in visited_vertices and len(visited_vertices) < max_vertices:
                        next_frontier.add(candidate)
                        visited_vertices.add(candidate)

            frontier = next_frontier

        vertex_rows = _fetch_vertices(conn, visited_vertices)
        return {
            "center_vertex_id": vertex_id,
            "direction": direction,
            "depth": depth,
            "vertices": [_vertex_from_row(row) for row in vertex_rows],
            "edges": list(edge_rows.values()),
        }



def _fetch_edges(
    conn: sqlite3.Connection,
    frontier: set[str],
    *,
    direction: str,
    edge_labels: list[str] | None,
) -> list[sqlite3.Row]:
    placeholders = ", ".join("?" for _ in frontier)
    clauses: list[str] = []
    params: list[Any] = []

    if direction in {"out", "both"}:
        clauses.append(f"out_v_id IN ({placeholders})")
        params.extend(sorted(frontier))
    if direction in {"in", "both"}:
        clauses.append(f"in_v_id IN ({placeholders})")
        params.extend(sorted(frontier))

    sql = f"SELECT id, label, out_v_id, in_v_id, properties_json FROM edges WHERE ({' OR '.join(clauses)})"
    if edge_labels:
        label_placeholders = ", ".join("?" for _ in edge_labels)
        sql += f" AND label IN ({label_placeholders})"
        params.extend(edge_labels)

    return conn.execute(sql, tuple(params)).fetchall()



def _fetch_vertices(conn: sqlite3.Connection, vertex_ids: set[str]) -> list[sqlite3.Row]:
    placeholders = ", ".join("?" for _ in vertex_ids)
    sql = f"SELECT id, label, properties_json FROM vertices WHERE id IN ({placeholders})"
    return conn.execute(sql, tuple(sorted(vertex_ids))).fetchall()



def _vertex_from_row(row: sqlite3.Row) -> dict[str, Any]:
    properties = json.loads(row["properties_json"])
    return {
        "vertex_id": str(row["id"]),
        "label": str(row["label"]),
        "properties": properties,
        "name": properties.get("name"),
        "path": properties.get("path"),
        "line": properties.get("line"),
    }
