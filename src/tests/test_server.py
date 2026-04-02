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


class TestRenderUiTemplate:
    def test_substitutes_both_placeholders(self):
        from cindex.server.app import _render_ui_template

        result = _render_ui_template(
            "model={{DEFAULT_MODEL}} db={{SQLITE_DB}}",
            {"DEFAULT_MODEL": "my-model", "SQLITE_DB": "/tmp/db"},
        )
        assert result == "model=my-model db=/tmp/db"

    def test_no_double_substitution_when_value_contains_other_placeholder(self):
        """A model name containing {{SQLITE_DB}} must not cause the db path to appear."""
        from cindex.server.app import _render_ui_template

        result = _render_ui_template(
            "model={{DEFAULT_MODEL}} db={{SQLITE_DB}}",
            {"DEFAULT_MODEL": "tricky{{SQLITE_DB}}model", "SQLITE_DB": "/real/db"},
        )
        assert "/real/db" not in result.split("db=")[1] or result.count("/real/db") == 1
        assert "tricky{{SQLITE_DB}}model" in result
        assert result == "model=tricky{{SQLITE_DB}}model db=/real/db"


class TestRateLimit:
    def test_requests_within_limit_succeed(self):
        from cindex.server.app import create_app

        client = TestClient(create_app(preload_default_model=False, rate_limit=5))
        for _ in range(5):
            assert client.get("/api/health").status_code == 200

    def test_requests_over_limit_receive_429(self):
        from cindex.server.app import create_app

        client = TestClient(create_app(preload_default_model=False, rate_limit=3))
        responses = [client.get("/api/health").status_code for _ in range(5)]
        assert responses[:3] == [200, 200, 200]
        assert 429 in responses[3:]

    def test_rate_limit_zero_disables_limiting(self):
        from cindex.server.app import create_app

        client = TestClient(create_app(preload_default_model=False, rate_limit=0))
        for _ in range(10):
            assert client.get("/api/health").status_code == 200


class TestCors:
    def test_cors_headers_absent_without_configuration(self):
        from cindex.server.app import create_app

        client = TestClient(create_app(preload_default_model=False))
        response = client.get("/api/health", headers={"Origin": "http://evil.example.com"})
        assert "access-control-allow-origin" not in response.headers

    def test_cors_headers_present_for_allowed_origin(self):
        from cindex.server.app import create_app

        client = TestClient(
            create_app(
                preload_default_model=False,
                cors_origins=["http://localhost:3000"],
            )
        )
        response = client.get("/api/health", headers={"Origin": "http://localhost:3000"})
        assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"

    def test_cors_headers_absent_for_unlisted_origin(self):
        from cindex.server.app import create_app

        client = TestClient(
            create_app(
                preload_default_model=False,
                cors_origins=["http://localhost:3000"],
            )
        )
        response = client.get("/api/health", headers={"Origin": "http://evil.example.com"})
        assert response.headers.get("access-control-allow-origin") != "http://evil.example.com"


class TestInputValidation:
    def test_graph_vertices_rejects_oversized_name_filter(self, tmp_path):
        from cindex.server.app import create_app

        db = tmp_path / "api.db"
        with sqlite3.connect(db) as conn:
            conn.execute(
                "CREATE TABLE vertices (id TEXT PRIMARY KEY, label TEXT NOT NULL, properties_json TEXT NOT NULL)"
            )

        client = TestClient(create_app(sqlite_db=str(db), preload_default_model=False, rate_limit=0))
        response = client.get("/api/graph/vertices", params={"name": "a" * 201})
        assert response.status_code == 422

    def test_graph_vertices_accepts_name_at_max_length(self, tmp_path):
        from cindex.server.app import create_app

        db = tmp_path / "api.db"
        with sqlite3.connect(db) as conn:
            conn.execute(
                "CREATE TABLE vertices (id TEXT PRIMARY KEY, label TEXT NOT NULL, properties_json TEXT NOT NULL)"
            )

        client = TestClient(create_app(sqlite_db=str(db), preload_default_model=False, rate_limit=0))
        response = client.get("/api/graph/vertices", params={"name": "a" * 200})
        assert response.status_code == 200

    def test_sql_endpoint_rejects_oversized_query(self, tmp_path):
        from cindex.server.app import create_app

        db = tmp_path / "api.db"
        db.touch()

        client = TestClient(create_app(sqlite_db=str(db), preload_default_model=False, rate_limit=0))
        response = client.post("/api/query/sql", json={"sql": "SELECT 1" + " " * 10_000})
        assert response.status_code == 422


class TestServerApp:
    def test_health_endpoint_reports_status(self):
        from cindex.server.app import create_app

        configured_db = str((Path("/tmp/test.db")).resolve())
        client = TestClient(create_app(sqlite_db="/tmp/test.db", preload_default_model=False))
        response = client.get("/api/health")

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
            "/api/embed",
            json={"text": "hello"},
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
            "/api/query/sql",
            json={"sql": "SELECT id, name FROM items"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["columns"] == ["id", "name"]
        assert body["rows"] == [{"id": 1, "name": "alpha"}]

    def test_query_templates_endpoint_lists_templates(self, tmp_path):
        from cindex.server.app import create_app

        db = tmp_path / "api.db"
        with sqlite3.connect(db) as conn:
            conn.execute(
                "CREATE TABLE vertices (id TEXT PRIMARY KEY, label TEXT NOT NULL, properties_json TEXT NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE edges (id TEXT PRIMARY KEY, label TEXT NOT NULL, out_v_id TEXT NOT NULL, in_v_id TEXT NOT NULL, properties_json TEXT NOT NULL)"
            )

        client = TestClient(create_app(sqlite_db=str(db), preload_default_model=False))
        response = client.get("/api/query/templates")

        assert response.status_code == 200
        body = response.json()
        assert body["count"] >= 6
        names = {template["name"] for template in body["templates"]}
        assert "missing_docstrings" in names
        assert "symbols_not_represented_by_tests" in names
        assert all("category" in template for template in body["templates"])

    def test_query_template_run_endpoint_returns_rows(self, tmp_path):
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

        client = TestClient(create_app(sqlite_db=str(db), preload_default_model=False))
        response = client.post("/api/query/templates/missing_docstrings/run")

        assert response.status_code == 200
        body = response.json()
        assert body["template"]["name"] == "missing_docstrings"
        assert body["template"]["category"] == "quality"
        assert body["columns"] == [
            "symbol_kind",
            "path",
            "line",
            "name",
            "docstring",
            "source",
            "vertex_id",
        ]
        assert body["rows"][0]["symbol_kind"] == "module"
        assert body["rows"][0]["name"] == "alpha"

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
        response = client.get(
            "/api/graph/vertices",
            params={"name": "alp", "label": "function"},
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
        response = client.get(
            "/api/graph/traverse",
            params={"vertex_id": "m1", "direction": "out", "depth": 1},
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
        response = client.get(
            "/api/query/similar",
            params={"text": "find alpha", "k": 5},
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
        response = client.get(
            "/api/query/similar",
            params={"text": "find alpha", "k": 5},
        )

        assert response.status_code == 400
        assert "Repo id must use alphanumeric chars" in response.json()["detail"]

    def test_ui_uses_configured_database_path(self, tmp_path):
        from cindex.server.app import create_app

        db = tmp_path / "ui.db"
        client = TestClient(create_app(sqlite_db=str(db), preload_default_model=False))

        response = client.get("/")

        assert response.status_code == 200
        assert str(db.resolve()) in response.text
        assert 'id="sql-db"' not in response.text
        assert 'id="sim-model"' not in response.text
        assert "Visualize" in response.text
        assert "Starter Queries" in response.text
        assert "titleizeCategory" in response.text


class TestServeCommand:
    def test_serve_command_invokes_uvicorn(self, monkeypatch):
        from cindex.cli.commands import serve

        captured: dict[str, object] = {}
        sentinel_app = object()

        def fake_create_app(*, default_model, cache_dir, sqlite_db, watch_directory, preload_default_model, run_indexer, cors_origins, rate_limit):
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
                    "allow_any_model": True,
                    "cors_origins": None,
                    "rate_limit": 120,
                },
            )()
        )

        assert rc == 0
        assert captured["app"] is sentinel_app
        assert captured["default_model"] == "model-a"
        assert captured["sqlite_db"] == "data.db"
        assert captured["run_indexer"] is True
