"""Tree-sitter based C++ source file adapter.

Extends C support with class/struct specifiers, namespaces, and template
declarations. The declarator unwrapping from the C adapter is reused.

Vertex labels produced
──────────────────────
    module   – one per file
    class    – class_specifier, struct_specifier, union_specifier (named)
    function – function_definition
    external – #include paths

Edge labels produced
────────────────────
    defines  – module → class / module → function
    imports  – module → external
"""
from __future__ import annotations

import logging
from pathlib import Path

from cindex.services.indexing.graph import PropertyGraph, Vertex
from cindex.services.indexing.adapters.base import (
    LanguageAdapter,
    _add_external,
    _complexity,
    _node_text,
)
from cindex.services.indexing.adapters.c import _extract_decl_name

logger = logging.getLogger(__name__)

_BRANCHING = frozenset({
    "if_statement", "else_clause", "for_statement",
    "range_based_for_statement", "while_statement", "do_statement",
    "switch_statement", "case_statement", "catch_clause",
    "conditional_expression",
})

_CLASS_LIKE_TYPES = frozenset({
    "class_specifier",
    "struct_specifier",
    "union_specifier",
})


class CppAdapter(LanguageAdapter):
    """Language adapter for C++ (.cpp, .cc, .cxx, .hpp, .hxx, .h++) files."""

    extensions = frozenset({".cpp", ".cc", ".cxx", ".hpp", ".hxx", ".h++"})

    @classmethod
    def is_available(cls) -> bool:
        try:
            import tree_sitter_cpp  # noqa: F401
            return True
        except ImportError:
            return False

    def parse(self, path: Path, graph: PropertyGraph) -> Vertex:
        try:
            import tree_sitter_cpp as tscpp
            from tree_sitter import Language, Parser
        except ImportError as exc:
            raise ImportError(
                "tree-sitter-cpp is required. "
                "Install it with: pip install tree-sitter-cpp"
            ) from exc

        source = path.read_bytes()
        language = Language(tscpp.language())
        parser = Parser(language)
        tree = parser.parse(source)

        module_v = graph.add_vertex("module", name=path.stem, path=str(path))
        _visit(tree.root_node, graph, module_v, None, source)
        return module_v


# ── AST visitors ──────────────────────────────────────────────────────────────

def _visit(node, graph: PropertyGraph, module_v: Vertex, class_v: Vertex | None, src: bytes) -> None:
    t = node.type

    if t == "function_definition":
        func_v = _handle_function(node, graph, module_v, class_v, src)
        if func_v is not None:
            body = node.child_by_field_name("body")
            if body is not None:
                for child in body.children:
                    _visit(child, graph, module_v, func_v, src)

    elif t in _CLASS_LIKE_TYPES:
        _handle_class(node, graph, module_v, src)

    elif t == "preproc_include":
        _handle_include(node, graph, module_v, src)

    elif t in ("namespace_definition", "linkage_specification"):
        # Recurse into namespace/extern "C" block
        body = node.child_by_field_name("body")
        if body is not None:
            for child in body.children:
                _visit(child, graph, module_v, class_v, src)

    elif t == "template_declaration":
        # template<...> class/function — recurse into the declaration
        for child in node.children:
            _visit(child, graph, module_v, class_v, src)

    else:
        for child in node.children:
            _visit(child, graph, module_v, class_v, src)


def _handle_function(node, graph: PropertyGraph, module_v: Vertex, class_v: Vertex | None, src: bytes) -> Vertex | None:
    decl = node.child_by_field_name("declarator")
    name = _extract_decl_name(decl, src) if decl is not None else None
    if not name:
        return None
    owner = class_v if class_v is not None else module_v
    func_v = graph.add_vertex(
        "function",
        name=name,
        path=module_v.properties["path"],
        line=node.start_point[0] + 1,
        source=_node_text(node, src),
        complexity=_complexity(node, _BRANCHING),
    )
    graph.add_edge("has_method" if owner.label == "class" else "defines", owner, func_v)
    return func_v


def _handle_class(node, graph: PropertyGraph, module_v: Vertex, src: bytes) -> None:
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    name = _node_text(name_node, src).strip()
    if not name:
        return
    class_v = graph.add_vertex(
        "class",
        name=name,
        path=module_v.properties["path"],
        line=node.start_point[0] + 1,
    )
    graph.add_edge("defines", module_v, class_v)

    body = node.child_by_field_name("body")
    if body is None:
        return
    for child in body.children:
        if child.type == "function_definition":
            func_v = _handle_function(child, graph, module_v, class_v, src)
            if func_v is not None:
                fn_body = child.child_by_field_name("body")
                if fn_body is not None:
                    for fn_child in fn_body.children:
                        _visit(fn_child, graph, module_v, func_v, src)
        elif child.type in _CLASS_LIKE_TYPES:
            _handle_class(child, graph, module_v, src)


def _handle_include(node, graph: PropertyGraph, module_v: Vertex, src: bytes) -> None:
    path_node = node.child_by_field_name("path")
    if path_node is None:
        return
    name = _node_text(path_node, src).strip("<>\"' \t")
    if name:
        _add_external(name, module_v, graph)
