"""Tree-sitter based JavaScript source file adapter.

Extracts modules, classes, functions, and imports from .js/.mjs/.cjs files.

Vertex labels produced
──────────────────────
    module    – one per file
    class     – class declarations
    function  – function declarations, arrow functions, methods
    external  – imported module specifiers

Edge labels produced
────────────────────
    defines    – module → class / module → function (top-level)
    has_method – class → function
    imports    – module → external
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

logger = logging.getLogger(__name__)

_BRANCHING = frozenset({
    "if_statement", "else_clause", "for_statement", "for_in_statement",
    "for_of_statement", "while_statement", "do_statement",
    "switch_case", "catch_clause", "ternary_expression",
})

_FUNC_DECL_TYPES = frozenset({
    "function_declaration", "generator_function_declaration",
})

_FUNC_VAL_TYPES = frozenset({
    "arrow_function", "function_expression", "generator_function",
})


class JavaScriptAdapter(LanguageAdapter):
    """Language adapter for JavaScript (.js, .mjs, .cjs) files."""

    extensions = frozenset({".js", ".mjs", ".cjs"})

    @classmethod
    def is_available(cls) -> bool:
        try:
            import tree_sitter_javascript  # noqa: F401
            return True
        except ImportError:
            return False

    def parse(self, path: Path, graph: PropertyGraph) -> Vertex:
        try:
            import tree_sitter_javascript as tsjs
            from tree_sitter import Language, Parser
        except ImportError as exc:
            raise ImportError(
                "tree-sitter-javascript is required. "
                "Install it with: pip install tree-sitter-javascript"
            ) from exc

        source = path.read_bytes()
        language = Language(tsjs.language())
        parser = Parser(language)
        tree = parser.parse(source)

        module_v = graph.add_vertex("module", name=path.stem, path=str(path))
        _visit(tree.root_node, graph, module_v, None, source)
        return module_v


# ── AST visitors ──────────────────────────────────────────────────────────────

def _visit(node, graph: PropertyGraph, module_v: Vertex, class_v: Vertex | None, src: bytes) -> None:
    t = node.type

    if t in _FUNC_DECL_TYPES:
        func_v = _handle_func_decl(node, graph, module_v, class_v, src)
        if func_v is not None:
            body = node.child_by_field_name("body")
            if body is not None:
                for child in body.children:
                    _visit(child, graph, module_v, func_v, src)

    elif t in ("lexical_declaration", "variable_declaration"):
        for child in node.children:
            if child.type == "variable_declarator":
                func_v = _handle_var_declarator(child, graph, module_v, class_v, src)
                if func_v is not None:
                    value_node = child.child_by_field_name("value")
                    if value_node is not None:
                        body = value_node.child_by_field_name("body")
                        if body is not None:
                            for bchild in body.children:
                                _visit(bchild, graph, module_v, func_v, src)

    elif t == "class_declaration":
        _handle_class(node, graph, module_v, src)

    elif t == "export_statement":
        # Unwrap: export function/class/const/default
        decl = node.child_by_field_name("declaration")
        if decl is not None:
            _visit(decl, graph, module_v, class_v, src)

    elif t == "import_statement":
        _handle_import(node, graph, module_v, src)

    else:
        for child in node.children:
            _visit(child, graph, module_v, class_v, src)


def _handle_func_decl(node, graph: PropertyGraph, module_v: Vertex, class_v: Vertex | None, src: bytes) -> Vertex | None:
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return None
    owner = class_v if class_v is not None else module_v
    return _add_function(node, _node_text(name_node, src), owner, module_v, graph, src)


def _handle_var_declarator(node, graph: PropertyGraph, module_v: Vertex, class_v: Vertex | None, src: bytes) -> Vertex | None:
    name_node = node.child_by_field_name("name")
    value_node = node.child_by_field_name("value")
    if name_node is None or value_node is None:
        return None
    if value_node.type not in _FUNC_VAL_TYPES:
        return None
    owner = class_v if class_v is not None else module_v
    return _add_function(value_node, _node_text(name_node, src), owner, module_v, graph, src)


def _handle_class(node, graph: PropertyGraph, module_v: Vertex, src: bytes) -> None:
    name_node = node.child_by_field_name("name")
    name = _node_text(name_node, src) if name_node else "<anonymous>"
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
        if child.type == "method_definition":
            _handle_method(child, graph, module_v, class_v, src)
        elif child.type in ("lexical_declaration", "variable_declaration"):
            for sub in child.children:
                if sub.type == "variable_declarator":
                    _handle_var_declarator(sub, graph, module_v, class_v, src)


def _handle_method(node, graph: PropertyGraph, module_v: Vertex, class_v: Vertex, src: bytes) -> Vertex | None:
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return None
    return _add_function(node, _node_text(name_node, src), class_v, module_v, graph, src)


def _add_function(node, name: str, owner: Vertex, module_v: Vertex, graph: PropertyGraph, src: bytes) -> Vertex:
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


def _handle_import(node, graph: PropertyGraph, module_v: Vertex, src: bytes) -> None:
    source_node = node.child_by_field_name("source")
    if source_node is None:
        return
    name = _node_text(source_node, src).strip("'\"` \t")
    if name:
        _add_external(name, module_v, graph)
