"""Tree-sitter based Go source file adapter.

Extracts modules, struct/interface types, functions/methods, and imports.

Go methods are associated with their receiver type: if a struct vertex for the
receiver already exists, the method gets a ``has_method`` edge; otherwise it is
attached to a new class vertex created for the receiver type.

Vertex labels produced
──────────────────────
    module    – one per file
    class     – type declarations whose underlying type is struct or interface
    function  – function_declaration and method_declaration
    external  – import paths

Edge labels produced
────────────────────
    defines    – module → class / module → function
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
    _collect_calls_ts,
    _complexity,
    _node_text,
    _preceding_doc_comment,
)

logger = logging.getLogger(__name__)

_BRANCHING = frozenset({
    "if_statement", "else_clause", "for_statement", "switch_statement",
    "case_clause", "select_statement", "comm_clause", "type_switch_statement",
})

_FUNC_NODE_TYPES = frozenset({"function_declaration", "method_declaration", "func_literal"})


class GoAdapter(LanguageAdapter):
    """Language adapter for Go (.go) files."""

    extensions = frozenset({".go"})

    @classmethod
    def is_available(cls) -> bool:
        try:
            import tree_sitter_go  # noqa: F401
            return True
        except ImportError:
            return False

    def parse(self, path: Path, graph: PropertyGraph) -> Vertex:
        try:
            import tree_sitter_go as tsgo
            from tree_sitter import Language, Parser
        except ImportError as exc:
            raise ImportError(
                "tree-sitter-go is required. "
                "Install it with: pip install tree-sitter-go"
            ) from exc

        source = path.read_bytes()
        language = Language(tsgo.language())
        parser = Parser(language)
        tree = parser.parse(source)

        module_v = graph.add_vertex("module", name=path.stem, path=str(path))
        _visit(tree.root_node, graph, module_v, source)
        _collect_calls_ts(
            tree.root_node, graph, module_v, source,
            func_node_types=_FUNC_NODE_TYPES,
            call_node_types=frozenset({"call_expression"}),
            get_callee_name=_go_callee_name,
        )
        return module_v


# ── AST visitors ──────────────────────────────────────────────────────────────

def _visit(node, graph: PropertyGraph, module_v: Vertex, src: bytes, scope_v: Vertex | None = None) -> None:
    t = node.type

    if t == "function_declaration":
        _handle_func(node, graph, module_v, src)

    elif t == "method_declaration":
        _handle_method(node, graph, module_v, src)

    elif t == "type_declaration":
        _handle_type_decl(node, graph, module_v, src)

    elif t == "import_declaration":
        _handle_import(node, graph, module_v, src)

    else:
        for child in node.children:
            _visit(child, graph, module_v, src, scope_v=scope_v)


def _handle_func(node, graph: PropertyGraph, module_v: Vertex, src: bytes) -> Vertex | None:
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
    doc = _preceding_doc_comment(node, src, prefixes=("//",))
    if doc:
        props["docstring"] = doc
    func_v = graph.add_vertex("function", **props)
    graph.add_edge("defines", module_v, func_v)
    body = node.child_by_field_name("body")
    if body is not None:
        for child in body.children:
            _visit(child, graph, module_v, src, scope_v=func_v)
    return func_v


def _handle_method(node, graph: PropertyGraph, module_v: Vertex, src: bytes) -> Vertex | None:
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return None

    receiver_type = _receiver_type_name(node, src)
    if receiver_type:
        owner = _find_or_create_class(receiver_type, node, graph, module_v)
    else:
        owner = module_v

    props: dict = {
        "name": _node_text(name_node, src),
        "path": module_v.properties["path"],
        "line": node.start_point[0] + 1,
        "source": _node_text(node, src),
        "complexity": _complexity(node, _BRANCHING),
    }
    doc = _preceding_doc_comment(node, src, prefixes=("//",))
    if doc:
        props["docstring"] = doc
    func_v = graph.add_vertex("function", **props)
    graph.add_edge("has_method" if owner.label == "class" else "defines", owner, func_v)
    body = node.child_by_field_name("body")
    if body is not None:
        for child in body.children:
            _visit(child, graph, module_v, src, scope_v=func_v)
    return func_v


def _handle_type_decl(node, graph: PropertyGraph, module_v: Vertex, src: bytes) -> None:
    for child in node.children:
        if child.type == "type_spec":
            name_node = child.child_by_field_name("name")
            type_node = child.child_by_field_name("type")
            if name_node is None or type_node is None:
                continue
            if type_node.type in ("struct_type", "interface_type"):
                class_v = graph.add_vertex(
                    "class",
                    name=_node_text(name_node, src),
                    path=module_v.properties["path"],
                    line=child.start_point[0] + 1,
                )
                graph.add_edge("defines", module_v, class_v)


def _handle_import(node, graph: PropertyGraph, module_v: Vertex, src: bytes) -> None:
    for child in node.children:
        if child.type == "import_spec_list":
            for spec in child.children:
                if spec.type == "import_spec":
                    _extract_import_spec(spec, graph, module_v, src)
        elif child.type == "import_spec":
            _extract_import_spec(child, graph, module_v, src)


def _extract_import_spec(node, graph: PropertyGraph, module_v: Vertex, src: bytes) -> None:
    path_node = node.child_by_field_name("path")
    if path_node is None:
        return
    name = _node_text(path_node, src).strip("\"` \t")
    if name:
        _add_external(name, module_v, graph)


def _receiver_type_name(method_node, src: bytes) -> str | None:
    """Extract the receiver type name from a method_declaration node."""
    receiver = method_node.child_by_field_name("receiver")
    if receiver is None:
        return None
    # Walk to find the first type_identifier (skipping pointer_type wrappers)
    stack = list(receiver.children)
    while stack:
        n = stack.pop()
        if n.type == "type_identifier":
            return _node_text(n, src)
        stack.extend(n.children)
    return None


def _go_callee_name(call_node, src: bytes) -> str | None:
    func_node = call_node.child_by_field_name("function")
    if func_node is None:
        return None
    if func_node.type == "identifier":
        return _node_text(func_node, src)
    if func_node.type == "selector_expression":
        field = func_node.child_by_field_name("field")
        return _node_text(field, src) if field else None
    return None


def _find_or_create_class(name: str, ref_node, graph: PropertyGraph, module_v: Vertex) -> Vertex:
    """Return existing class vertex by name or create a new one."""
    existing = next(
        (
            v for v in graph.vertices("class")
            if v.properties.get("name") == name
            and v.properties.get("path") == module_v.properties["path"]
        ),
        None,
    )
    if existing is not None:
        return existing
    class_v = graph.add_vertex(
        "class",
        name=name,
        path=module_v.properties["path"],
        line=ref_node.start_point[0] + 1,
    )
    graph.add_edge("defines", module_v, class_v)
    return class_v
