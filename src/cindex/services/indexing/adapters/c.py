"""Tree-sitter based C source file adapter.

Extracts modules, struct/union types, functions, and #include directives.

C function names sit inside a nested declarator hierarchy:
    function_definition
      declarator: function_declarator
        declarator: identifier  ← the actual name
        (or pointer_declarator wrapping a function_declarator)

A helper walks that hierarchy to find the identifier.

Vertex labels produced
──────────────────────
    module   – one per file
    class    – struct_specifier, union_specifier (named)
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
    _collect_calls_ts,
    _complexity,
    _node_text,
)

logger = logging.getLogger(__name__)

_BRANCHING = frozenset({
    "if_statement", "else_clause", "for_statement", "while_statement",
    "do_statement", "switch_statement", "case_statement", "catch_clause",
})


class CAdapter(LanguageAdapter):
    """Language adapter for C (.c, .h) files."""

    extensions = frozenset({".c", ".h"})

    @classmethod
    def is_available(cls) -> bool:
        try:
            import tree_sitter_c  # noqa: F401
            return True
        except ImportError:
            return False

    def parse(self, path: Path, graph: PropertyGraph) -> Vertex:
        try:
            import tree_sitter_c as tsc
            from tree_sitter import Language, Parser
        except ImportError as exc:
            raise ImportError(
                "tree-sitter-c is required. "
                "Install it with: pip install tree-sitter-c"
            ) from exc

        source = path.read_bytes()
        language = Language(tsc.language())
        parser = Parser(language)
        tree = parser.parse(source)

        module_v = graph.add_vertex("module", name=path.stem, path=str(path))
        _visit(tree.root_node, graph, module_v, source)
        _collect_calls_ts(
            tree.root_node, graph, module_v, source,
            func_node_types=frozenset({"function_definition"}),
            call_node_types=frozenset({"call_expression"}),
            get_callee_name=_c_callee_name,
        )
        return module_v


# ── AST visitors ──────────────────────────────────────────────────────────────

def _visit(node, graph: PropertyGraph, module_v: Vertex, src: bytes, scope_v: Vertex | None = None) -> None:
    t = node.type

    if t == "function_definition":
        _handle_function(node, graph, module_v, src)

    elif t in ("struct_specifier", "union_specifier"):
        _handle_struct(node, graph, module_v, src)

    elif t == "preproc_include":
        _handle_include(node, graph, module_v, src)

    elif t == "declaration":
        # typedef struct { ... } Foo; — walk children to find struct specifiers
        for child in node.children:
            if child.type in ("struct_specifier", "union_specifier"):
                _handle_struct(child, graph, module_v, src)

    else:
        for child in node.children:
            _visit(child, graph, module_v, src, scope_v=scope_v)


def _handle_function(node, graph: PropertyGraph, module_v: Vertex, src: bytes) -> Vertex | None:
    decl = node.child_by_field_name("declarator")
    name = _extract_decl_name(decl, src) if decl is not None else None
    if not name:
        return None
    func_v = graph.add_vertex(
        "function",
        name=name,
        path=module_v.properties["path"],
        line=node.start_point[0] + 1,
        source=_node_text(node, src),
        complexity=_complexity(node, _BRANCHING),
    )
    graph.add_edge("defines", module_v, func_v)
    body = node.child_by_field_name("body")
    if body is not None:
        for child in body.children:
            _visit(child, graph, module_v, src, scope_v=func_v)
    return func_v


def _handle_struct(node, graph: PropertyGraph, module_v: Vertex, src: bytes) -> None:
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


def _handle_include(node, graph: PropertyGraph, module_v: Vertex, src: bytes) -> None:
    path_node = node.child_by_field_name("path")
    if path_node is None:
        return
    name = _node_text(path_node, src).strip("<>\"' \t")
    if name:
        _add_external(name, module_v, graph)


def _c_callee_name(call_node, src: bytes) -> str | None:
    func_node = call_node.child_by_field_name("function")
    if func_node is None:
        return None
    if func_node.type == "identifier":
        return _node_text(func_node, src)
    if func_node.type == "field_expression":
        field = func_node.child_by_field_name("field")
        return _node_text(field, src) if field else None
    return None


def _extract_decl_name(node, src: bytes) -> str | None:
    """Recursively unwrap declarator nodes to find the function identifier."""
    if node is None:
        return None
    t = node.type
    if t == "identifier":
        return _node_text(node, src)
    if t in ("function_declarator", "pointer_declarator", "array_declarator"):
        inner = node.child_by_field_name("declarator")
        return _extract_decl_name(inner, src)
    # Fallback: find first identifier child
    for child in node.children:
        if child.type == "identifier":
            return _node_text(child, src)
    return None
