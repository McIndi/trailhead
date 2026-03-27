from __future__ import annotations

from pathlib import Path
import sqlite3
import struct

from fastapi.testclient import TestClient


class TestModelStore:
    def test_get_embedding_model_reuses_loaded_instance(self, monkeypatch):
        from cindex.services.embeddings import get_embedding_model

        created: list[tuple[str, str | None]] = []

        class FakeModel:
            def __init__(self, model_name: str, cache_folder: str | None = None) -> None:
                created.append((model_name, cache_folder))
                self.model_name = model_name

            def encode(self, text):
                return [0.1, 0.2, 0.3]

        monkeypatch.setattr(
            "sentence_transformers.SentenceTransformer",
            FakeModel,
        )

        model1 = get_embedding_model("model-a", cache_folder="/tmp/cache")
        model2 = get_embedding_model("model-a", cache_folder="/tmp/cache")

        assert model1 is model2
        assert created == [("model-a", "/tmp/cache")]


class TestServerApp:
    def test_health_endpoint_reports_status(self):
        from cindex.server.app import create_app

        configured_db = str((Path("/tmp/test.db")).resolve())
        client = TestClient(create_app(sqlite_db="/tmp/test.db", preload_default_model=False))
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["sqlite_db"] == configured_db
        assert response.json()["indexer_enabled"] is False

    def test_embed_endpoint_uses_embedding_service(self, monkeypatch):
        from cindex.server.app import create_app

        monkeypatch.setattr(
            "cindex.server.app.generate_embedding",
            lambda text, model, cache_folder=None: [0.1, 0.2, 0.3],
        )

        client = TestClient(create_app(preload_default_model=False))
        response = client.post(
            "/embed",
            json={"text": "hello", "model": "model-a"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["dimensions"] == 3
        assert body["embedding"] == [0.1, 0.2, 0.3]

    def test_query_sql_endpoint_returns_rows(self, tmp_path):
        from cindex.server.app import create_app

        db = tmp_path / "api.db"
        with sqlite3.connect(db) as conn:
            conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)")
            conn.execute("INSERT INTO items(name) VALUES ('alpha')")

        client = TestClient(create_app(sqlite_db=str(db), preload_default_model=False))
        response = client.post(
            "/query/sql",
            json={"sql": "SELECT id, name FROM items"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["columns"] == ["id", "name"]
        assert body["rows"] == [{"id": 1, "name": "alpha"}]

    def test_graph_vertices_endpoint_returns_matches(self, tmp_path):
        from cindex.server.app import create_app

        db = tmp_path / "api.db"
        with sqlite3.connect(db) as conn:
            conn.execute(
                "CREATE TABLE vertices (id TEXT PRIMARY KEY, label TEXT NOT NULL, properties_json TEXT NOT NULL)"
            )
            conn.execute(
                "INSERT INTO vertices(id, label, properties_json) VALUES (?, ?, ?)",
                ("f1", "function", '{"name": "alpha", "path": "/tmp/a.py", "line": 10}'),
            )

        client = TestClient(create_app(sqlite_db=str(db), preload_default_model=False))
        response = client.post(
            "/graph/vertices",
            json={"name": "alp", "label": "function"},
        )

        assert response.status_code == 200
        assert response.json()["count"] == 1
        assert response.json()["rows"][0]["vertex_id"] == "f1"

    def test_graph_traverse_endpoint_returns_subgraph(self, tmp_path):
        from cindex.server.app import create_app

        db = tmp_path / "api.db"
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
                "INSERT INTO edges(id, label, out_v_id, in_v_id, properties_json) VALUES (?, ?, ?, ?, ?)",
                ("d1", "defines", "m1", "f1", '{}'),
            )

        client = TestClient(create_app(sqlite_db=str(db), preload_default_model=False))
        response = client.post(
            "/graph/traverse",
            json={"vertex_id": "m1", "direction": "out", "depth": 1},
        )

        assert response.status_code == 200
        assert {vertex["vertex_id"] for vertex in response.json()["vertices"]} == {"m1", "f1"}

    def test_query_similar_endpoint_returns_matches(self, tmp_path, monkeypatch):
        from cindex.server.app import create_app

        db = tmp_path / "api.db"
        with sqlite3.connect(db) as conn:
            conn.execute(
                "CREATE TABLE vertices (id TEXT PRIMARY KEY, label TEXT NOT NULL, properties_json TEXT NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE vertex_embeddings (vertex_id TEXT PRIMARY KEY, model_name TEXT NOT NULL, source_text TEXT NOT NULL, dimension INTEGER NOT NULL, embedding BLOB NOT NULL)"
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

        client = TestClient(create_app(sqlite_db=str(db), preload_default_model=False))
        response = client.post(
            "/query/similar",
            json={
                "text": "find alpha",
                "k": 5,
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["count"] == 1
        assert body["rows"][0]["name"] == "alpha"

    def test_query_similar_endpoint_returns_bad_request_for_invalid_model(self, tmp_path, monkeypatch):
        from cindex.server.app import create_app

        db = tmp_path / "api.db"
        with sqlite3.connect(db) as conn:
            conn.execute(
                "CREATE TABLE vertices (id TEXT PRIMARY KEY, label TEXT NOT NULL, properties_json TEXT NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE vertex_embeddings (vertex_id TEXT PRIMARY KEY, model_name TEXT NOT NULL, source_text TEXT NOT NULL, dimension INTEGER NOT NULL, embedding BLOB NOT NULL)"
            )
            conn.execute(
                "INSERT INTO vertex_embeddings(vertex_id, model_name, source_text, dimension, embedding) VALUES (?, ?, ?, ?, ?)",
                ("v1", "model-a", "alpha", 3, struct.pack("<3f", 0.1, 0.2, 0.3)),
            )

        monkeypatch.setattr(
            "cindex.services.indexing.query.generate_embedding",
            lambda text, model_name, cache_folder=None: (_ for _ in ()).throw(
                OSError("Repo id must use alphanumeric chars")
            ),
        )

        client = TestClient(create_app(sqlite_db=str(db), preload_default_model=False))
        response = client.post(
            "/query/similar",
            json={
                "text": "find alpha",
                "model": ".cindex/db.sqlite",
                "k": 5,
            },
        )

        assert response.status_code == 400
        assert "Repo id must use alphanumeric chars" in response.json()["detail"]

    def test_ui_uses_configured_database_path(self, tmp_path):
        from cindex.server.app import create_app

        db = tmp_path / "ui.db"
        client = TestClient(create_app(sqlite_db=str(db), preload_default_model=False))

        response = client.get("/ui")

        assert response.status_code == 200
        assert str(db.resolve()) in response.text
        assert 'id="sql-db"' not in response.text
        assert 'id="sim-model"' not in response.text


class TestServeCommand:
    def test_serve_command_invokes_uvicorn(self, monkeypatch):
        from cindex.cli.commands import serve

        captured: dict[str, object] = {}
        sentinel_app = object()

        def fake_create_app(*, default_model, cache_dir, sqlite_db, watch_directory, preload_default_model, run_indexer):
            captured["default_model"] = default_model
            captured["cache_dir"] = cache_dir
            captured["sqlite_db"] = sqlite_db
            captured["watch_directory"] = watch_directory
            captured["preload_default_model"] = preload_default_model
            captured["run_indexer"] = run_indexer
            return sentinel_app

        def fake_run(app, host, port):
            captured["app"] = app
            captured["host"] = host
            captured["port"] = port

        monkeypatch.setattr("cindex.server.app.create_app", fake_create_app)
        monkeypatch.setattr("uvicorn.run", fake_run)

        rc = serve.run(
            type(
                "Args",
                (),
                {
                    "directory": ".",
                    "host": "127.0.0.1",
                    "port": 9000,
                    "model": "model-a",
                    "cache_dir": None,
                    "sqlite_db": "data.db",
                    "no_preload": False,
                },
            )()
        )

        assert rc == 0
        assert captured["app"] is sentinel_app
        assert captured["default_model"] == "model-a"
        assert captured["sqlite_db"] == "data.db"
        assert captured["run_indexer"] is True
