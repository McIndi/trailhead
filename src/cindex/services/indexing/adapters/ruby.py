"""Tree-sitter based Ruby source file adapter.

Extracts modules, classes/modules, methods, and require calls.

Vertex labels produced
──────────────────────
    module   – one per file
    class    – class and module declarations
    function – method and singleton_method definitions
    external – require / require_relative argument strings

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
    "if", "elsif", "else", "unless", "case", "when",
    "while", "until", "for", "rescue",
})

_REQUIRE_METHODS = frozenset({"require", "require_relative"})
_FUNC_NODE_TYPES = frozenset({"method", "singleton_method"})


class RubyAdapter(LanguageAdapter):
    """Language adapter for Ruby (.rb) files."""

    extensions = frozenset({".rb"})

    @classmethod
    def is_available(cls) -> bool:
        try:
            import tree_sitter_ruby  # noqa: F401
            return True
        except ImportError:
            return False

    def parse(self, path: Path, graph: PropertyGraph) -> Vertex:
        try:
            import tree_sitter_ruby as tsrb
            from tree_sitter import Language, Parser
        except ImportError as exc:
            raise ImportError(
                "tree-sitter-ruby is required. "
                "Install it with: pip install tree-sitter-ruby"
            ) from exc

        source = path.read_bytes()
        language = Language(tsrb.language())
        parser = Parser(language)
        tree = parser.parse(source)

        module_v = graph.add_vertex("module", name=path.stem, path=str(path))
        _visit(tree.root_node, graph, module_v, None, source)
        _collect_calls_ts(
            tree.root_node, graph, module_v, source,
            func_node_types=_FUNC_NODE_TYPES,
            call_node_types=frozenset({"call"}),
            get_callee_name=_ruby_callee_name,
        )
        return module_v


# ── AST visitors ──────────────────────────────────────────────────────────────

def _visit(node, graph: PropertyGraph, module_v: Vertex, class_v: Vertex | None, src: bytes) -> None:
    t = node.type

    if t in ("class", "module"):
        _handle_class(node, graph, module_v, src)

    elif t in ("method", "singleton_method"):
        owner = class_v if class_v is not None else module_v
        func_v = _handle_method(node, graph, module_v, owner, src)
        if func_v is not None:
            body = node.child_by_field_name("body")
            if body is not None:
                for child in body.children:
                    _visit(child, graph, module_v, func_v, src)

    elif t == "call":
        _handle_call(node, graph, module_v, src)

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
        if child.type in ("method", "singleton_method"):
            func_v = _handle_method(child, graph, module_v, class_v, src)
            if func_v is not None:
                fn_body = child.child_by_field_name("body")
                if fn_body is not None:
                    for fn_child in fn_body.children:
                        _visit(fn_child, graph, module_v, func_v, src)
        elif child.type in ("class", "module"):
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
    doc = _preceding_doc_comment(node, src, prefixes=("#",))
    if doc:
        props["docstring"] = doc
    func_v = graph.add_vertex("function", **props)
    graph.add_edge("has_method" if owner.label == "class" else "defines", owner, func_v)
    return func_v


def _ruby_callee_name(call_node, src: bytes) -> str | None:
    method_node = call_node.child_by_field_name("method")
    if method_node is None:
        return None
    name = _node_text(method_node, src)
    # Skip require/require_relative — those are import edges, not calls
    return name if name not in _REQUIRE_METHODS else None


def _handle_call(node, graph: PropertyGraph, module_v: Vertex, src: bytes) -> None:
    """Handle require / require_relative calls."""
    method_node = node.child_by_field_name("method")
    if method_node is None:
        return
    if _node_text(method_node, src) not in _REQUIRE_METHODS:
        return

    args = node.child_by_field_name("arguments")
    if args is None:
        return
    for child in args.children:
        if child.type in ("string", "simple_symbol"):
            name = _node_text(child, src).strip("'\":`() \t")
            if name:
                _add_external(name, module_v, graph)
            break
