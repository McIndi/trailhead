"""Tests for the indexing service (property graph model + tree-sitter parser)."""
from __future__ import annotations

import sqlite3

import pytest


# ── PropertyGraph unit tests ──────────────────────────────────────────────────

class TestPropertyGraph:
    def test_add_vertex_stores_label_and_properties(self):
        from cindex.services.indexing.graph import PropertyGraph

        graph = PropertyGraph()
        v = graph.add_vertex("module", name="mymodule", path="/foo/mymodule.py")
        assert v.label == "module"
        assert v.properties["name"] == "mymodule"
        assert v.properties["path"] == "/foo/mymodule.py"
        assert v.id is not None

    def test_add_edge_stores_label_and_endpoints(self):
        from cindex.services.indexing.graph import PropertyGraph

        graph = PropertyGraph()
        v1 = graph.add_vertex("module", name="a")
        v2 = graph.add_vertex("class", name="MyClass")
        e = graph.add_edge("defines", v1, v2)
        assert e.label == "defines"
        assert e.out_v.id == v1.id
        assert e.in_v.id == v2.id

    def test_vertices_filter_by_label(self):
        from cindex.services.indexing.graph import PropertyGraph

        graph = PropertyGraph()
        graph.add_vertex("module", name="a")
        graph.add_vertex("class", name="Foo")
        graph.add_vertex("class", name="Bar")
        assert len(graph.vertices("module")) == 1
        assert len(graph.vertices("class")) == 2
        assert len(graph.vertices()) == 3

    def test_edges_filter_by_label(self):
        from cindex.services.indexing.graph import PropertyGraph

        graph = PropertyGraph()
        v1 = graph.add_vertex("module", name="m")
        v2 = graph.add_vertex("class", name="C")
        v3 = graph.add_vertex("function", name="f")
        ext = graph.add_vertex("external", name="os")
        graph.add_edge("defines", v1, v2)
        graph.add_edge("defines", v1, v3)
        graph.add_edge("imports", v1, ext)
        assert len(graph.edges("defines")) == 2
        assert len(graph.edges("imports")) == 1
        assert len(graph.edges()) == 3

    def test_out_edges_filtered_by_label(self):
        from cindex.services.indexing.graph import PropertyGraph

        graph = PropertyGraph()
        mod = graph.add_vertex("module", name="m")
        cls = graph.add_vertex("class", name="C")
        fn = graph.add_vertex("function", name="f")
        graph.add_edge("defines", mod, cls)
        graph.add_edge("defines", mod, fn)
        assert len(graph.out_edges(mod, "defines")) == 2
        assert len(graph.out_edges(cls, "defines")) == 0

    def test_in_edges_filtered_by_label(self):
        from cindex.services.indexing.graph import PropertyGraph

        graph = PropertyGraph()
        mod = graph.add_vertex("module", name="m")
        cls = graph.add_vertex("class", name="C")
        graph.add_edge("defines", mod, cls)
        assert len(graph.in_edges(cls, "defines")) == 1
        assert len(graph.in_edges(mod, "defines")) == 0

    def test_vertex_ids_are_unique(self):
        from cindex.services.indexing.graph import PropertyGraph

        graph = PropertyGraph()
        vertices = [graph.add_vertex("module", name=str(i)) for i in range(100)]
        assert len({v.id for v in vertices}) == 100

    def test_get_vertex_returns_vertex_by_id(self):
        from cindex.services.indexing.graph import PropertyGraph

        graph = PropertyGraph()
        v = graph.add_vertex("module", name="m")
        assert graph.get_vertex(v.id) is v

    def test_get_vertex_returns_none_for_missing_id(self):
        from cindex.services.indexing.graph import PropertyGraph

        graph = PropertyGraph()
        assert graph.get_vertex("nonexistent") is None

    def test_out_edges_no_label_filter_returns_all(self):
        from cindex.services.indexing.graph import PropertyGraph

        graph = PropertyGraph()
        a = graph.add_vertex("module", name="a")
        b = graph.add_vertex("external", name="b")
        c = graph.add_vertex("class", name="c")
        graph.add_edge("imports", a, b)
        graph.add_edge("defines", a, c)
        assert len(graph.out_edges(a)) == 2


# ── Parser tests ──────────────────────────────────────────────────────────────

class TestParsePythonFile:
    def test_parses_module_docstring(self, tmp_path):
        from cindex.services.indexing.graph import PropertyGraph
        from cindex.services.indexing.parser import parse_python_file

        f = tmp_path / "docmod.py"
        f.write_text('"""module docs"""\n\nimport os\n')
        graph = PropertyGraph()
        module_v = parse_python_file(f, graph)

        assert module_v.properties["docstring"] == "module docs"

    def test_parses_module_vertex(self, tmp_path):
        from cindex.services.indexing.graph import PropertyGraph
        from cindex.services.indexing.parser import parse_python_file

        f = tmp_path / "mymodule.py"
        f.write_text("")
        graph = PropertyGraph()
        module_v = parse_python_file(f, graph)
        assert module_v.label == "module"
        assert module_v.properties["name"] == "mymodule"
        assert module_v.properties["path"] == str(f)

    def test_parses_top_level_function(self, tmp_path):
        from cindex.services.indexing.graph import PropertyGraph
        from cindex.services.indexing.parser import parse_python_file

        f = tmp_path / "funcs.py"
        f.write_text("def greet(name):\n    return 'hi'\n")
        graph = PropertyGraph()
        parse_python_file(f, graph)

        funcs = graph.vertices("function")
        assert len(funcs) == 1
        assert funcs[0].properties["name"] == "greet"
        assert funcs[0].properties["line"] == 1

    def test_function_source_and_docstring_are_captured(self, tmp_path):
        from cindex.services.indexing.graph import PropertyGraph
        from cindex.services.indexing.parser import parse_python_file

        f = tmp_path / "funcs.py"
        f.write_text(
            "def greet(name):\n"
            "    \"\"\"Say hi\"\"\"\n"
            "    message = f'hi {name}'\n"
            "    return message\n"
        )
        graph = PropertyGraph()
        parse_python_file(f, graph)

        func = graph.vertices("function")[0]
        assert func.properties["docstring"] == "Say hi"
        assert "message = f'hi {name}'" in func.properties["source"]

    def test_method_source_and_docstring_are_captured(self, tmp_path):
        from cindex.services.indexing.graph import PropertyGraph
        from cindex.services.indexing.parser import parse_python_file

        f = tmp_path / "animal.py"
        f.write_text(
            "class Dog:\n"
            "    def bark(self):\n"
            "        \"\"\"Bark loudly\"\"\"\n"
            "        return 'woof'\n"
        )
        graph = PropertyGraph()
        parse_python_file(f, graph)

        method = graph.vertices("function")[0]
        assert method.properties["docstring"] == "Bark loudly"
        assert "return 'woof'" in method.properties["source"]

    def test_parses_class(self, tmp_path):
        from cindex.services.indexing.graph import PropertyGraph
        from cindex.services.indexing.parser import parse_python_file

        f = tmp_path / "classes.py"
        f.write_text("class Dog:\n    pass\n")
        graph = PropertyGraph()
        parse_python_file(f, graph)

        classes = graph.vertices("class")
        assert len(classes) == 1
        assert classes[0].properties["name"] == "Dog"
        assert classes[0].properties["line"] == 1

    def test_parses_class_methods(self, tmp_path):
        from cindex.services.indexing.graph import PropertyGraph
        from cindex.services.indexing.parser import parse_python_file

        f = tmp_path / "animal.py"
        f.write_text(
            "class Dog:\n"
            "    def bark(self):\n"
            "        pass\n"
            "    def sit(self):\n"
            "        pass\n"
        )
        graph = PropertyGraph()
        parse_python_file(f, graph)

        methods = graph.vertices("function")
        assert len(methods) == 2
        assert {v.properties["name"] for v in methods} == {"bark", "sit"}

    def test_method_edges_from_class(self, tmp_path):
        from cindex.services.indexing.graph import PropertyGraph
        from cindex.services.indexing.parser import parse_python_file

        f = tmp_path / "animal.py"
        f.write_text("class Dog:\n    def bark(self):\n        pass\n")
        graph = PropertyGraph()
        parse_python_file(f, graph)

        classes = graph.vertices("class")
        assert len(classes) == 1
        method_edges = graph.out_edges(classes[0], "has_method")
        assert len(method_edges) == 1
        assert method_edges[0].in_v.properties["name"] == "bark"

    def test_module_defines_edge_to_class(self, tmp_path):
        from cindex.services.indexing.graph import PropertyGraph
        from cindex.services.indexing.parser import parse_python_file

        f = tmp_path / "mod.py"
        f.write_text("class Foo:\n    pass\n")
        graph = PropertyGraph()
        module_v = parse_python_file(f, graph)

        define_edges = graph.out_edges(module_v, "defines")
        assert len(define_edges) == 1
        assert define_edges[0].in_v.label == "class"

    def test_module_defines_edge_to_top_level_function(self, tmp_path):
        from cindex.services.indexing.graph import PropertyGraph
        from cindex.services.indexing.parser import parse_python_file

        f = tmp_path / "mod.py"
        f.write_text("def foo():\n    pass\n")
        graph = PropertyGraph()
        module_v = parse_python_file(f, graph)

        define_edges = graph.out_edges(module_v, "defines")
        assert len(define_edges) == 1
        assert define_edges[0].in_v.label == "function"

    def test_parses_import_statement(self, tmp_path):
        from cindex.services.indexing.graph import PropertyGraph
        from cindex.services.indexing.parser import parse_python_file

        f = tmp_path / "importer.py"
        f.write_text("import os\nimport sys\n")
        graph = PropertyGraph()
        module_v = parse_python_file(f, graph)

        imported = {e.in_v.properties["name"] for e in graph.out_edges(module_v, "imports")}
        assert imported == {"os", "sys"}

    def test_parses_from_import_statement(self, tmp_path):
        from cindex.services.indexing.graph import PropertyGraph
        from cindex.services.indexing.parser import parse_python_file

        f = tmp_path / "importer.py"
        f.write_text("from pathlib import Path\n")
        graph = PropertyGraph()
        module_v = parse_python_file(f, graph)

        import_edges = graph.out_edges(module_v, "imports")
        assert len(import_edges) == 1
        assert import_edges[0].in_v.properties["name"] == "pathlib"

    def test_duplicate_imports_share_external_vertex(self, tmp_path):
        from cindex.services.indexing.graph import PropertyGraph
        from cindex.services.indexing.parser import parse_python_file

        f1 = tmp_path / "a.py"
        f2 = tmp_path / "b.py"
        f1.write_text("import os\n")
        f2.write_text("import os\n")
        graph = PropertyGraph()
        parse_python_file(f1, graph)
        parse_python_file(f2, graph)

        os_vertices = [v for v in graph.vertices("external") if v.properties["name"] == "os"]
        assert len(os_vertices) == 1

    def test_decorated_function_is_captured(self, tmp_path):
        from cindex.services.indexing.graph import PropertyGraph
        from cindex.services.indexing.parser import parse_python_file

        f = tmp_path / "deco.py"
        f.write_text("import functools\n\n@functools.cache\ndef expensive():\n    pass\n")
        graph = PropertyGraph()
        parse_python_file(f, graph)

        names = {v.properties["name"] for v in graph.vertices("function")}
        assert "expensive" in names

    def test_decorated_method_is_captured(self, tmp_path):
        from cindex.services.indexing.graph import PropertyGraph
        from cindex.services.indexing.parser import parse_python_file

        f = tmp_path / "cls.py"
        f.write_text(
            "class Foo:\n"
            "    @staticmethod\n"
            "    def helper():\n"
            "        pass\n"
        )
        graph = PropertyGraph()
        parse_python_file(f, graph)

        classes = graph.vertices("class")
        assert len(classes) == 1
        method_edges = graph.out_edges(classes[0], "has_method")
        assert len(method_edges) == 1
        assert method_edges[0].in_v.properties["name"] == "helper"

    def test_nested_function_not_captured(self, tmp_path):
        """Functions defined inside other functions are not indexed in Phase 1."""
        from cindex.services.indexing.graph import PropertyGraph
        from cindex.services.indexing.parser import parse_python_file

        f = tmp_path / "nested.py"
        f.write_text(
            "def outer():\n"
            "    def inner():\n"
            "        pass\n"
        )
        graph = PropertyGraph()
        parse_python_file(f, graph)

        func_names = {v.properties["name"] for v in graph.vertices("function")}
        assert "outer" in func_names
        assert "inner" not in func_names


# ── Walker tests ──────────────────────────────────────────────────────────────

class TestIndexDirectory:
    def test_empty_directory(self, tmp_path):
        from cindex.services.indexing.walker import index_directory

        graph = index_directory(tmp_path)
        assert graph.vertices() == []
        assert graph.edges() == []

    def test_indexes_single_file(self, tmp_path):
        from cindex.services.indexing.walker import index_directory

        (tmp_path / "foo.py").write_text("def bar(): pass\n")
        graph = index_directory(tmp_path)
        assert len(graph.vertices("module")) == 1
        assert len(graph.vertices("function")) == 1

    def test_recurses_into_subdirectories(self, tmp_path):
        from cindex.services.indexing.walker import index_directory

        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "root.py").write_text("")
        (sub / "nested.py").write_text("")
        graph = index_directory(tmp_path)
        module_names = {v.properties["name"] for v in graph.vertices("module")}
        assert "root" in module_names
        assert "nested" in module_names

    def test_ignores_non_python_files(self, tmp_path):
        from cindex.services.indexing.walker import index_directory

        (tmp_path / "README.md").write_text("# readme")
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "script.py").write_text("")
        graph = index_directory(tmp_path)
        assert len(graph.vertices("module")) == 1

    def test_accepts_existing_graph(self, tmp_path):
        from cindex.services.indexing.graph import PropertyGraph
        from cindex.services.indexing.walker import index_directory

        (tmp_path / "a.py").write_text("")
        graph = PropertyGraph()
        result = index_directory(tmp_path, graph=graph)
        assert result is graph
        assert len(result.vertices("module")) == 1

    def test_multiple_files_accumulate_in_graph(self, tmp_path):
        from cindex.services.indexing.walker import index_directory

        (tmp_path / "a.py").write_text("class A: pass\n")
        (tmp_path / "b.py").write_text("class B: pass\n")
        graph = index_directory(tmp_path)
        assert len(graph.vertices("module")) == 2
        assert len(graph.vertices("class")) == 2


class TestSqliteStore:
    def test_persist_graph_creates_db_and_saves_rows(self, tmp_path):
        from cindex.services.indexing.graph import PropertyGraph
        from cindex.services.indexing.sqlite_store import persist_graph

        graph = PropertyGraph()
        mod = graph.add_vertex("module", name="m", path="/tmp/m.py")
        fn = graph.add_vertex("function", name="f", path="/tmp/m.py", line=1)
        graph.add_edge("defines", mod, fn)

        db = tmp_path / "graph.db"
        v_count, e_count = persist_graph(graph, db)

        assert db.exists()
        assert v_count == 2
        assert e_count == 1

    def test_load_graph_round_trip(self, tmp_path):
        from cindex.services.indexing.graph import PropertyGraph
        from cindex.services.indexing.sqlite_store import load_graph
        from cindex.services.indexing.sqlite_store import persist_graph

        graph = PropertyGraph()
        mod = graph.add_vertex("module", name="m", path="/tmp/m.py")
        cls = graph.add_vertex("class", name="C", path="/tmp/m.py", line=2)
        graph.add_edge("defines", mod, cls)

        db = tmp_path / "graph.db"
        persist_graph(graph, db)
        loaded = load_graph(db)

        assert len(loaded.vertices("module")) == 1
        assert len(loaded.vertices("class")) == 1
        assert len(loaded.edges("defines")) == 1

    def test_persist_replace_mode_clears_existing_rows(self, tmp_path):
        from cindex.services.indexing.graph import PropertyGraph
        from cindex.services.indexing.sqlite_store import persist_graph

        db = tmp_path / "graph.db"

        g1 = PropertyGraph()
        a = g1.add_vertex("module", name="a", path="/tmp/a.py")
        b = g1.add_vertex("function", name="fa", path="/tmp/a.py", line=1)
        g1.add_edge("defines", a, b)
        persist_graph(g1, db)

        g2 = PropertyGraph()
        c = g2.add_vertex("module", name="c", path="/tmp/c.py")
        d = g2.add_vertex("function", name="fc", path="/tmp/c.py", line=1)
        g2.add_edge("defines", c, d)
        v_count, e_count = persist_graph(g2, db, append=False)

        assert v_count == 2
        assert e_count == 1

    def test_persist_append_mode_keeps_existing_rows(self, tmp_path):
        from cindex.services.indexing.graph import PropertyGraph
        from cindex.services.indexing.sqlite_store import persist_graph

        db = tmp_path / "graph.db"

        g1 = PropertyGraph()
        a = g1.add_vertex("module", name="a", path="/tmp/a.py")
        b = g1.add_vertex("function", name="fa", path="/tmp/a.py", line=1)
        g1.add_edge("defines", a, b)
        persist_graph(g1, db)

        g2 = PropertyGraph()
        c = g2.add_vertex("module", name="c", path="/tmp/c.py")
        d = g2.add_vertex("function", name="fc", path="/tmp/c.py", line=1)
        g2.add_edge("defines", c, d)
        v_count, e_count = persist_graph(g2, db, append=True)

        assert v_count == 4
        assert e_count == 2

    def test_persist_vertex_embeddings_writes_rows(self, tmp_path, monkeypatch):
        from cindex.services.indexing.graph import PropertyGraph
        from cindex.services.indexing.sqlite_store import persist_graph
        from cindex.services.indexing.sqlite_store import persist_vertex_embeddings

        graph = PropertyGraph()
        mod = graph.add_vertex("module", name="m", path="/tmp/m.py")
        fn = graph.add_vertex("function", name="f", path="/tmp/m.py", line=1)
        graph.add_edge("defines", mod, fn)

        monkeypatch.setattr(
            "cindex.services.indexing.sqlite_store.generate_embeddings",
            lambda texts, model_name, cache_folder=None: [[0.1, 0.2, 0.3] for _ in texts],
        )
        monkeypatch.setattr(
            "cindex.services.indexing.sqlite_store._try_initialize_sqlite_vector",
            lambda conn, dimension: True,
        )

        db = tmp_path / "graph.db"
        persist_graph(graph, db)
        rows, dim, vector_ready = persist_vertex_embeddings(
            graph,
            db,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )

        assert rows == 2
        assert dim == 3
        assert vector_ready is True

        with sqlite3.connect(db) as conn:
            count = conn.execute("SELECT COUNT(*) FROM vertex_embeddings").fetchone()[0]
            stored_dim = conn.execute(
                "SELECT DISTINCT dimension FROM vertex_embeddings"
            ).fetchone()[0]
            assert count == 2
            assert stored_dim == 3

    def test_persist_vertex_embeddings_replace_mode_clears_previous(self, tmp_path, monkeypatch):
        from cindex.services.indexing.graph import PropertyGraph
        from cindex.services.indexing.sqlite_store import persist_graph
        from cindex.services.indexing.sqlite_store import persist_vertex_embeddings

        monkeypatch.setattr(
            "cindex.services.indexing.sqlite_store.generate_embeddings",
            lambda texts, model_name, cache_folder=None: [[0.1, 0.2] for _ in texts],
        )

        db = tmp_path / "graph.db"

        g1 = PropertyGraph()
        g1.add_vertex("module", name="a", path="/tmp/a.py")
        persist_graph(g1, db)
        persist_vertex_embeddings(g1, db, model_name="m1", append=False)

        g2 = PropertyGraph()
        g2.add_vertex("module", name="b", path="/tmp/b.py")
        persist_graph(g2, db, append=False)
        rows, _, _ = persist_vertex_embeddings(g2, db, model_name="m1", append=False)

        assert rows == 1


class TestIndexCommandSqlite:
    def test_index_command_writes_sqlite_file(self, tmp_path):
        from cindex.cli import app as cli

        src = tmp_path / "src"
        src.mkdir()
        (src / "a.py").write_text("def hello():\n    pass\n")
        db = tmp_path / "graph.db"

        rc = cli.main(["index", str(src), "--sqlite-db", str(db)])

        assert rc == 0
        assert db.exists()

    def test_index_command_embed_model_requires_sqlite_db(self, tmp_path):
        from cindex.cli import app as cli

        src = tmp_path / "src"
        src.mkdir()
        (src / "a.py").write_text("def hello():\n    pass\n")

        rc = cli.main(["index", str(src), "--embed-model", "sentence-transformers/all-MiniLM-L6-v2"])
        assert rc == 1

    def test_index_command_embed_model_persists_vectors(self, tmp_path, monkeypatch):
        from cindex.cli import app as cli

        src = tmp_path / "src"
        src.mkdir()
        (src / "a.py").write_text("def hello():\n    pass\n")
        db = tmp_path / "graph.db"

        monkeypatch.setattr(
            "cindex.cli.commands.index.persist_vertex_embeddings",
            lambda graph, db_path, **kwargs: (2, 384, True),
        )

        rc = cli.main(
            [
                "index",
                str(src),
                "--sqlite-db",
                str(db),
                "--embed-model",
                "sentence-transformers/all-MiniLM-L6-v2",
            ]
        )
        assert rc == 0
