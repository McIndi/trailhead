"""Tree-sitter based C# source file adapter.

Extracts modules, classes/interfaces/structs/enums, methods, and using directives.

Vertex labels produced
──────────────────────
    module    – one per file
    class     – class, struct, interface, enum, record declarations
    function  – method and constructor declarations
    external  – using directive names

Edge labels produced
────────────────────
    defines    – module → class / module → function
    has_method – class → function
    imports    – module → external
"""
from __future__ import annotations

import logging
from pathlib import Path

from trailhead.services.indexing.graph import PropertyGraph, Vertex
from trailhead.services.indexing.adapters.base import (
    LanguageAdapter,
    _add_external,
    _collect_calls_ts,
    _complexity,
    _node_text,
    _preceding_doc_comment,
)

logger = logging.getLogger(__name__)

_BRANCHING = frozenset({
    "if_statement", "else_clause", "for_statement", "foreach_statement",
    "while_statement", "do_statement", "switch_section",
    "catch_clause", "conditional_expression",
})

_CLASS_LIKE_TYPES = frozenset({
    "class_declaration",
    "struct_declaration",
    "interface_declaration",
    "enum_declaration",
    "record_declaration",
    "record_struct_declaration",
})

_METHOD_LIKE_TYPES = frozenset({
    "method_declaration",
    "constructor_declaration",
    "operator_declaration",
    "conversion_operator_declaration",
})

_FUNC_NODE_TYPES = _METHOD_LIKE_TYPES


class CSharpAdapter(LanguageAdapter):
    """Language adapter for C# (.cs) files."""

    extensions = frozenset({".cs"})

    @classmethod
    def is_available(cls) -> bool:
        try:
            import tree_sitter_c_sharp  # noqa: F401
            return True
        except ImportError:
            return False

    def parse(self, path: Path, graph: PropertyGraph) -> Vertex:
        try:
            import tree_sitter_c_sharp as tscs
            from tree_sitter import Language, Parser
        except ImportError as exc:
            raise ImportError(
                "tree-sitter-c-sharp is required. "
                "Install it with: pip install tree-sitter-c-sharp"
            ) from exc

        source = path.read_bytes()
        language = Language(tscs.language())
        parser = Parser(language)
        tree = parser.parse(source)

        module_v = graph.add_vertex("module", name=path.stem, path=str(path))
        _visit(tree.root_node, graph, module_v, None, source)
        _collect_calls_ts(
            tree.root_node, graph, module_v, source,
            func_node_types=_FUNC_NODE_TYPES,
            call_node_types=frozenset({"invocation_expression"}),
            get_callee_name=_cs_callee_name,
        )
        return module_v


# ── AST visitors ──────────────────────────────────────────────────────────────

def _visit(node, graph: PropertyGraph, module_v: Vertex, class_v: Vertex | None, src: bytes) -> None:
    t = node.type

    if t in _CLASS_LIKE_TYPES:
        _handle_class(node, graph, module_v, src)

    elif t in _METHOD_LIKE_TYPES:
        owner = class_v if class_v is not None else module_v
        func_v = _handle_method(node, graph, module_v, owner, src)
        if func_v is not None:
            body = node.child_by_field_name("body")
            if body is not None:
                for child in body.children:
                    _visit(child, graph, module_v, func_v, src)

    elif t == "using_directive":
        _handle_using(node, graph, module_v, src)

    elif t == "namespace_declaration":
        # Recurse into namespace body
        body = node.child_by_field_name("body")
        if body is not None:
            for child in body.children:
                _visit(child, graph, module_v, class_v, src)

    else:
        for child in node.children:
            _visit(child, graph, module_v, class_v, src)


def _handle_class(node, graph: PropertyGraph, module_v: Vertex, src: bytes) -> None:
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    class_v = graph.add_vertex(
        "class",
        name=_node_text(name_node, src),
        path=module_v.properties["path"],
        line=node.start_point[0] + 1,
    )
    graph.add_edge("defines", module_v, class_v)

    body = node.child_by_field_name("body")
    if body is None:
        return
    for child in body.children:
        if child.type in _METHOD_LIKE_TYPES:
            func_v = _handle_method(child, graph, module_v, class_v, src)
            if func_v is not None:
                fn_body = child.child_by_field_name("body")
                if fn_body is not None:
                    for fn_child in fn_body.children:
                        _visit(fn_child, graph, module_v, func_v, src)
        elif child.type in _CLASS_LIKE_TYPES:
            _handle_class(child, graph, module_v, src)


def _handle_method(node, graph: PropertyGraph, module_v: Vertex, owner: Vertex, src: bytes) -> Vertex | None:
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return None
    props: dict = {
        "name": _node_text(name_node, src),
        "path": module_v.properties["path"],
        "line": node.start_point[0] + 1,
        "source": _node_text(node, src),
        "complexity": _complexity(node, _BRANCHING),
    }
    doc = _preceding_doc_comment(node, src, prefixes=("///",))
    if doc:
        props["docstring"] = doc
    func_v = graph.add_vertex("function", **props)
    graph.add_edge("has_method" if owner.label == "class" else "defines", owner, func_v)
    return func_v


def _cs_callee_name(call_node, src: bytes) -> str | None:
    func_node = call_node.child_by_field_name("function")
    if func_node is None:
        return None
    if func_node.type == "identifier":
        return _node_text(func_node, src)
    if func_node.type == "member_access_expression":
        name = func_node.child_by_field_name("name")
        return _node_text(name, src) if name else None
    return None


def _handle_using(node, graph: PropertyGraph, module_v: Vertex, src: bytes) -> None:
    # using System; / using System.Collections.Generic; / using static Foo;
    for child in node.children:
        if child.type in ("qualified_name", "identifier", "name_equals"):
            name = _node_text(child, src).strip()
            if name and name != "static":
                _add_external(name, module_v, graph)
            break
