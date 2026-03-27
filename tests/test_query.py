from __future__ import annotations

import sqlite3
import struct


class TestExecuteSqlQuery:
    def test_execute_sql_query_returns_columns_and_rows(self, tmp_path):
        from cindex.services.indexing.query import execute_sql_query

        db = tmp_path / "query.db"
        with sqlite3.connect(db) as conn:
            conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)")
            conn.execute("INSERT INTO items(name) VALUES ('alpha')")

        columns, rows = execute_sql_query(db, "SELECT id, name FROM items")

        assert columns == ["id", "name"]
        assert rows == [{"id": 1, "name": "alpha"}]

    def test_execute_sql_query_rejects_non_read_only_sql(self, tmp_path):
        from cindex.services.indexing.query import execute_sql_query

        db = tmp_path / "query.db"
        with sqlite3.connect(db) as conn:
            conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)")

        try:
            execute_sql_query(db, "DELETE FROM items")
        except ValueError as exc:
            assert "read-only" in str(exc)
        else:
            raise AssertionError("Expected ValueError for non-read-only SQL")


class TestGraphQueries:
    def test_search_vertices_filters_rows(self, tmp_path):
        from cindex.services.indexing.graph_query import search_vertices

        db = tmp_path / "query.db"
        with sqlite3.connect(db) as conn:
            conn.execute(
                "CREATE TABLE vertices (id TEXT PRIMARY KEY, label TEXT NOT NULL, properties_json TEXT NOT NULL)"
            )
            conn.execute(
                "INSERT INTO vertices(id, label, properties_json) VALUES (?, ?, ?)",
                ("m1", "module", '{"name": "alpha", "path": "/tmp/a.py"}'),
            )
            conn.execute(
                "INSERT INTO vertices(id, label, properties_json) VALUES (?, ?, ?)",
                ("f1", "function", '{"name": "alpha_fn", "path": "/tmp/a.py", "line": 2}'),
            )

        rows = search_vertices(db, name="alpha", label="function")

        assert len(rows) == 1
        assert rows[0]["vertex_id"] == "f1"
        assert rows[0]["name"] == "alpha_fn"

    def test_traverse_graph_returns_neighbors(self, tmp_path):
        from cindex.services.indexing.graph_query import traverse_graph

        db = tmp_path / "query.db"
        with sqlite3.connect(db) as conn:
            conn.execute(
                "CREATE TABLE vertices (id TEXT PRIMARY KEY, label TEXT NOT NULL, properties_json TEXT NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE edges (id TEXT PRIMARY KEY, label TEXT NOT NULL, out_v_id TEXT NOT NULL, in_v_id TEXT NOT NULL, properties_json TEXT NOT NULL)"
            )
            conn.execute(
                "INSERT INTO vertices(id, label, properties_json) VALUES (?, ?, ?)",
                ("m1", "module", '{"name": "alpha", "path": "/tmp/a.py"}'),
            )
            conn.execute(
                "INSERT INTO vertices(id, label, properties_json) VALUES (?, ?, ?)",
                ("f1", "function", '{"name": "alpha_fn", "path": "/tmp/a.py", "line": 2}'),
            )
            conn.execute(
                "INSERT INTO vertices(id, label, properties_json) VALUES (?, ?, ?)",
                ("e1", "external", '{"name": "pathlib"}'),
            )
            conn.execute(
                "INSERT INTO edges(id, label, out_v_id, in_v_id, properties_json) VALUES (?, ?, ?, ?, ?)",
                ("d1", "defines", "m1", "f1", '{}'),
            )
            conn.execute(
                "INSERT INTO edges(id, label, out_v_id, in_v_id, properties_json) VALUES (?, ?, ?, ?, ?)",
                ("i1", "imports", "m1", "e1", '{}'),
            )

        subgraph = traverse_graph(db, vertex_id="m1", direction="out", depth=1)

        assert subgraph["center_vertex_id"] == "m1"
        assert {vertex["vertex_id"] for vertex in subgraph["vertices"]} == {"m1", "f1", "e1"}
        assert {edge["edge_id"] for edge in subgraph["edges"]} == {"d1", "i1"}


class TestFindSimilarVertices:
    def test_find_similar_vertices_returns_ranked_rows(self, tmp_path, monkeypatch):
        from cindex.services.indexing.query import find_similar_vertices

        db = tmp_path / "query.db"
        with sqlite3.connect(db) as conn:
            conn.execute(
                """
                CREATE TABLE vertices (
                    id TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    properties_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE vertex_embeddings (
                    vertex_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    source_text TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    embedding BLOB NOT NULL
                )
                """
            )
            conn.execute(
                "INSERT INTO vertices(id, label, properties_json) VALUES (?, ?, ?)",
                ("v1", "function", '{"name": "alpha", "path": "/tmp/a.py", "line": 10}'),
            )
            conn.execute(
                "INSERT INTO vertex_embeddings(vertex_id, model_name, source_text, dimension, embedding) VALUES (?, ?, ?, ?, ?)",
                (
                    "v1",
                    "sentence-transformers/all-MiniLM-L6-v2",
                    "function alpha /tmp/a.py:10",
                    3,
                    struct.pack("<3f", 0.1, 0.2, 0.3),
                ),
            )

        monkeypatch.setattr(
            "cindex.services.indexing.query.generate_embedding",
            lambda text, model_name, cache_folder=None: [0.1, 0.2, 0.3],
        )

        rows = find_similar_vertices(
            db,
            "find alpha",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            k=5,
        )

        assert len(rows) == 1
        assert rows[0]["label"] == "function"
        assert rows[0]["name"] == "alpha"


class TestQueryCommand:
    def test_query_sql_command_prints_json(self, tmp_path, capsys):
        from cindex.cli import app as cli

        db = tmp_path / "query.db"
        with sqlite3.connect(db) as conn:
            conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)")
            conn.execute("INSERT INTO items(name) VALUES ('alpha')")

        rc = cli.main(
            [
                "query",
                "sql",
                "--sqlite-db",
                str(db),
                "--sql",
                "SELECT id, name FROM items",
                "--output",
                "json",
            ]
        )

        assert rc == 0
        assert '"name": "alpha"' in capsys.readouterr().out

    def test_query_similar_command_prints_json(self, tmp_path, monkeypatch, capsys):
        from cindex.cli import app as cli

        db = tmp_path / "query.db"
        db.touch()

        monkeypatch.setattr(
            "cindex.cli.commands.query.find_similar_vertices",
            lambda *args, **kwargs: [
                {
                    "vertex_id": "v1",
                    "label": "function",
                    "name": "persist_graph",
                    "path": "/tmp/file.py",
                    "line": 12,
                    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                    "distance": 0.123,
                }
            ],
        )

        rc = cli.main(
            [
                "query",
                "similar",
                "sqlite persistence",
                "--sqlite-db",
                str(db),
                "--output",
                "json",
            ]
        )

        assert rc == 0
        assert '"label": "function"' in capsys.readouterr().out