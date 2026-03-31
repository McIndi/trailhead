"""Tree-sitter based Bash/Shell source file adapter.

Bash has no classes or imports. Only function definitions are extracted.

Vertex labels produced
──────────────────────
    module   – one per file
    function – function_definition nodes

Edge labels produced
────────────────────
    defines – module → function
"""
from __future__ import annotations

import logging
from pathlib import Path

from cindex.services.indexing.graph import PropertyGraph, Vertex
from cindex.services.indexing.adapters.base import (
    LanguageAdapter,
    _complexity,
    _node_text,
)

logger = logging.getLogger(__name__)

_BRANCHING = frozenset({
    "if_statement", "case_statement", "while_statement",
    "for_statement", "until_statement", "elif_clause",
})


class BashAdapter(LanguageAdapter):
    """Language adapter for Bash/Shell (.sh, .bash) files."""

    extensions = frozenset({".sh", ".bash"})

    @classmethod
    def is_available(cls) -> bool:
        try:
            import tree_sitter_bash  # noqa: F401
            return True
        except ImportError:
            return False

    def parse(self, path: Path, graph: PropertyGraph) -> Vertex:
        try:
            import tree_sitter_bash as tsbash
            from tree_sitter import Language, Parser
        except ImportError as exc:
            raise ImportError(
                "tree-sitter-bash is required. "
                "Install it with: pip install tree-sitter-bash"
            ) from exc

        source = path.read_bytes()
        language = Language(tsbash.language())
        parser = Parser(language)
        tree = parser.parse(source)

        module_v = graph.add_vertex("module", name=path.stem, path=str(path))
        _visit(tree.root_node, graph, module_v, source)
        return module_v


# ── AST visitors ──────────────────────────────────────────────────────────────

def _visit(node, graph: PropertyGraph, module_v: Vertex, src: bytes) -> None:
    if node.type == "function_definition":
        _handle_function(node, graph, module_v, src)
    else:
        for child in node.children:
            _visit(child, graph, module_v, src)


def _handle_function(node, graph: PropertyGraph, module_v: Vertex, src: bytes) -> None:
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    name = _node_text(name_node, src).strip()
    if not name:
        return
    func_v = graph.add_vertex(
        "function",
        name=name,
        path=module_v.properties["path"],
        line=node.start_point[0] + 1,
        source=_node_text(node, src),
        complexity=_complexity(node, _BRANCHING),
    )
    graph.add_edge("defines", module_v, func_v)
