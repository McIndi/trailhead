"""Tests for the indexing service (property graph model + tree-sitter parser)."""
from __future__ import annotations

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
