"""Tree-sitter based Rust source file adapter.

Extracts modules, structs/enums/traits, functions, and use declarations.

Vertex labels produced
──────────────────────
    module    – one per file
    class     – struct_item, enum_item, trait_item
    function  – function_item (top-level and inside impl blocks)
    external  – use_declaration paths

Edge labels produced
────────────────────
    defines    – module → class / module → function
    has_method – class → function (impl block methods)
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
    "if_expression", "else", "match_expression", "match_arm",
    "while_expression", "loop_expression", "for_expression",
})

# Top-level item types that map to "class" vertices
_TYPE_ITEM_TYPES = frozenset({
    "struct_item", "enum_item", "trait_item",
})

_FUNC_NODE_TYPES = frozenset({"function_item"})


class RustAdapter(LanguageAdapter):
    """Language adapter for Rust (.rs) files."""

    extensions = frozenset({".rs"})

    @classmethod
    def is_available(cls) -> bool:
        try:
            import tree_sitter_rust  # noqa: F401
            return True
        except ImportError:
            return False

    def parse(self, path: Path, graph: PropertyGraph) -> Vertex:
        try:
            import tree_sitter_rust as tsrs
            from tree_sitter import Language, Parser
        except ImportError as exc:
            raise ImportError(
                "tree-sitter-rust is required. "
                "Install it with: pip install tree-sitter-rust"
            ) from exc

        source = path.read_bytes()
        language = Language(tsrs.language())
        parser = Parser(language)
        tree = parser.parse(source)

        module_v = graph.add_vertex("module", name=path.stem, path=str(path))
        _visit(tree.root_node, graph, module_v, source)
        _collect_calls_ts(
            tree.root_node, graph, module_v, source,
            func_node_types=_FUNC_NODE_TYPES,
            call_node_types=frozenset({"call_expression"}),
            get_callee_name=_rust_callee_name,
        )
        return module_v


# ── AST visitors ──────────────────────────────────────────────────────────────

def _visit(node, graph: PropertyGraph, module_v: Vertex, src: bytes, scope_v: Vertex | None = None) -> None:
    t = node.type

    if t == "function_item":
        _handle_function(node, graph, module_v, module_v, src)

    elif t in _TYPE_ITEM_TYPES:
        _handle_type_item(node, graph, module_v, src)

    elif t == "impl_item":
        _handle_impl(node, graph, module_v, src)

    elif t == "use_declaration":
        _handle_use(node, graph, module_v, src)

    elif t == "mod_item":
        # Inline module — recurse into its body
        body = node.child_by_field_name("body")
        if body is not None:
            for child in body.children:
                _visit(child, graph, module_v, src, scope_v=scope_v)

    else:
        for child in node.children:
            _visit(child, graph, module_v, src, scope_v=scope_v)


def _handle_function(node, graph: PropertyGraph, module_v: Vertex, owner: Vertex, src: bytes) -> Vertex | None:
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
    body = node.child_by_field_name("body")
    if body is not None:
        for child in body.children:
            _visit(child, graph, module_v, src, scope_v=func_v)
    return func_v


def _rust_callee_name(call_node, src: bytes) -> str | None:
    func_node = call_node.child_by_field_name("function")
    if func_node is None:
        return None
    if func_node.type == "identifier":
        return _node_text(func_node, src)
    if func_node.type in ("scoped_identifier", "field_expression"):
        # e.g. Foo::bar or self.helper — take the last name component
        for child in reversed(func_node.children):
            if child.type in ("identifier", "field_identifier"):
                return _node_text(child, src)
    return None


def _handle_type_item(node, graph: PropertyGraph, module_v: Vertex, src: bytes) -> None:
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


def _handle_impl(node, graph: PropertyGraph, module_v: Vertex, src: bytes) -> None:
    """Handle an impl block, associating its methods with the implementing type."""
    type_node = node.child_by_field_name("type")
    if type_node is None:
        return

    # Find the implementing type name
    impl_type_name = _impl_type_name(type_node, src)

    # Look for an existing class vertex with this name in this file, or create one
    existing_class = next(
        (
            v for v in graph.vertices("class")
            if v.properties.get("name") == impl_type_name
            and v.properties.get("path") == module_v.properties["path"]
        ),
        None,
    )
    if existing_class is not None:
        owner = existing_class
    else:
        owner = graph.add_vertex(
            "class",
            name=impl_type_name,
            path=module_v.properties["path"],
            line=node.start_point[0] + 1,
        )
        graph.add_edge("defines", module_v, owner)

    # Walk the declaration list for function items
    body = node.child_by_field_name("body")
    if body is None:
        return
    for child in body.children:
        if child.type == "function_item":
            _handle_function(child, graph, module_v, owner, src)


def _impl_type_name(type_node, src: bytes) -> str:
    """Extract a readable name from the type node of an impl_item."""
    t = type_node.type
    if t == "type_identifier":
        return _node_text(type_node, src)
    if t == "generic_type":
        name_node = type_node.child_by_field_name("type")
        if name_node is not None:
            return _node_text(name_node, src)
    if t == "scoped_type_identifier":
        # e.g. foo::Bar — take the last component
        for child in reversed(type_node.children):
            if child.type == "type_identifier":
                return _node_text(child, src)
    return _node_text(type_node, src)


def _handle_use(node, graph: PropertyGraph, module_v: Vertex, src: bytes) -> None:
    arg = node.child_by_field_name("argument")
    if arg is None:
        return
    name = _node_text(arg, src).strip()
    # Take just the crate/top-level module (everything before the first ::)
    top = name.split("::")[0].strip("{} \t")
    if top and top not in ("self", "super", "crate"):
        _add_external(top, module_v, graph)
