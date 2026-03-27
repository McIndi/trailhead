from __future__ import annotations

from pathlib import Path
import sqlite3


class TestReindexFile:
    def test_reindex_file_replaces_existing_rows_and_tracks_deletes(self, tmp_path, monkeypatch):
        from cindex.services.indexing.sqlite_store import reindex_file

        source = tmp_path / "example.py"
        db = tmp_path / "graph.db"
        source.write_text("def alpha():\n    pass\n")

        monkeypatch.setattr(
            "cindex.services.indexing.sqlite_store.generate_embeddings",
            lambda texts, model_name, cache_folder=None: [[0.1, 0.2, 0.3] for _ in texts],
        )
        monkeypatch.setattr(
            "cindex.services.indexing.sqlite_store._try_initialize_sqlite_vector",
            lambda conn, dimension: True,
        )

        vertex_count, embedding_count = reindex_file(db, source, model_name="model-a")
        assert vertex_count == 2
        assert embedding_count == 2

        source.write_text("def beta():\n    pass\n")
        vertex_count, embedding_count = reindex_file(db, source, model_name="model-a")
        assert vertex_count == 2
        assert embedding_count == 2

        with sqlite3.connect(db) as conn:
            names = [
                row[0]
                for row in conn.execute(
                    "SELECT json_extract(properties_json, '$.name') FROM vertices WHERE label = 'function'"
                ).fetchall()
            ]
            indexed_files = conn.execute("SELECT path FROM indexed_files").fetchall()

        assert names == ["beta"]
        assert indexed_files == [(str(source.resolve()),)]

        source.unlink()
        vertex_count, embedding_count = reindex_file(db, source, model_name="model-a")
        assert vertex_count == 0
        assert embedding_count == 0

        with sqlite3.connect(db) as conn:
            remaining = conn.execute(
                "SELECT COUNT(*) FROM vertices WHERE json_extract(properties_json, '$.path') = ?",
                (str(source.resolve()),),
            ).fetchone()[0]
            indexed_files = conn.execute("SELECT COUNT(*) FROM indexed_files").fetchone()[0]

        assert remaining == 0
        assert indexed_files == 0


class TestLiveIndexer:
    def test_synchronize_reindexes_changed_new_and_deleted_files(self, tmp_path, monkeypatch):
        from cindex.services.indexing.live_indexer import LiveIndexer
        from cindex.services.indexing.sqlite_store import SCHEMA_SQL

        root = tmp_path / "src"
        root.mkdir()
        changed_file = root / "changed.py"
        new_file = root / "new.py"
        deleted_file = root / "deleted.py"
        changed_file.write_text("def changed():\n    pass\n")
        new_file.write_text("def fresh():\n    pass\n")

        db = tmp_path / "graph.db"
        with sqlite3.connect(db) as conn:
            conn.executescript(SCHEMA_SQL)
            conn.execute(
                "INSERT INTO indexed_files(path, mtime_ns) VALUES (?, ?)",
                (str(changed_file.resolve()), 0),
            )
            conn.execute(
                "INSERT INTO indexed_files(path, mtime_ns) VALUES (?, ?)",
                (str(deleted_file.resolve()), 123),
            )

        seen_paths: list[Path] = []

        def fake_reindex_file(db_path, path, **kwargs):
            seen_paths.append(path.resolve())
            return (0, 0)

        monkeypatch.setattr("cindex.services.indexing.live_indexer.reindex_file", fake_reindex_file)

        service = LiveIndexer(root=root, db_path=db, model_name="model-a")
        service.synchronize()

        assert set(seen_paths) == {
            changed_file.resolve(),
            new_file.resolve(),
            deleted_file.resolve(),
        }
