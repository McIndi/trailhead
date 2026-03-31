"""Tree-sitter based PHP source file adapter.

Uses the ``php`` sub-grammar (files with <?php tags). Extracts classes,
functions, methods, namespaces, and use/require statements.

Vertex labels produced
──────────────────────
    module   – one per file
    class    – class, interface, trait, enum declarations
    function – function and method definitions
    external – namespace use declarations and require/include paths

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
    _complexity,
    _node_text,
)

logger = logging.getLogger(__name__)

_BRANCHING = frozenset({
    "if_statement", "else_clause", "for_statement", "foreach_statement",
    "while_statement", "do_statement", "switch_statement",
    "match_expression", "catch_clause",
})

_CLASS_LIKE_TYPES = frozenset({
    "class_declaration",
    "interface_declaration",
    "trait_declaration",
    "enum_declaration",
})

_REQUIRE_TYPES = frozenset({
    "require_expression", "require_once_expression",
    "include_expression", "include_once_expression",
})


class PHPAdapter(LanguageAdapter):
    """Language adapter for PHP (.php) files."""

    extensions = frozenset({".php"})

    @classmethod
    def is_available(cls) -> bool:
        try:
            import tree_sitter_php  # noqa: F401
            return True
        except ImportError:
            return False

    def parse(self, path: Path, graph: PropertyGraph) -> Vertex:
        try:
            import tree_sitter_php as tsphp
            from tree_sitter import Language, Parser
        except ImportError as exc:
            raise ImportError(
                "tree-sitter-php is required. "
                "Install it with: pip install tree-sitter-php"
            ) from exc

        source = path.read_bytes()
        # tree-sitter-php exposes two sub-grammars: language_php() (with <?php
        # tags) and language_php_only() (pure PHP).  If this breaks on a version
        # upgrade, check: dir(tsphp) for the right name.
        language = Language(tsphp.language_php())
        parser = Parser(language)
        tree = parser.parse(source)

        module_v = graph.add_vertex("module", name=path.stem, path=str(path))
        _visit(tree.root_node, graph, module_v, None, source)
        return module_v


# ── AST visitors ──────────────────────────────────────────────────────────────

def _visit(node, graph: PropertyGraph, module_v: Vertex, class_v: Vertex | None, src: bytes) -> None:
    t = node.type

    if t in _CLASS_LIKE_TYPES:
        _handle_class(node, graph, module_v, src)

    elif t == "function_definition":
        owner = class_v if class_v is not None else module_v
        _handle_function(node, graph, module_v, owner, src)

    elif t == "method_declaration":
        owner = class_v if class_v is not None else module_v
        _handle_function(node, graph, module_v, owner, src)

    elif t == "namespace_use_declaration":
        _handle_use(node, graph, module_v, src)

    elif t in _REQUIRE_TYPES:
        _handle_require(node, graph, module_v, src)

    elif t == "namespace_definition":
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
        if child.type == "method_declaration":
            _handle_function(child, graph, module_v, class_v, src)


def _handle_function(node, graph: PropertyGraph, module_v: Vertex, owner: Vertex, src: bytes) -> None:
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    func_v = graph.add_vertex(
        "function",
        name=_node_text(name_node, src),
        path=module_v.properties["path"],
        line=node.start_point[0] + 1,
        source=_node_text(node, src),
        complexity=_complexity(node, _BRANCHING),
    )
    graph.add_edge("has_method" if owner.label == "class" else "defines", owner, func_v)


def _handle_use(node, graph: PropertyGraph, module_v: Vertex, src: bytes) -> None:
    # use App\Models\User; — take the full qualified name text
    for child in node.children:
        if child.type in ("qualified_name", "name"):
            name = _node_text(child, src).strip("\\; \t")
            if name:
                _add_external(name, module_v, graph)
            break


def _handle_require(node, graph: PropertyGraph, module_v: Vertex, src: bytes) -> None:
    for child in node.children:
        if child.type in ("string", "encapsed_string"):
            name = _node_text(child, src).strip("'\"() \t")
            if name:
                _add_external(name, module_v, graph)
            break
