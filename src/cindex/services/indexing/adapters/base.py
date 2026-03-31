"""Abstract base class and shared utilities for language adapters."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from cindex.services.indexing.graph import PropertyGraph, Vertex


def _node_text(node, source: bytes) -> str:
    """Return the source text spanned by *node*."""
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _complexity(root_node, branching_types: frozenset[str]) -> int:
    """Compute a McCabe-style cyclomatic complexity for a function node.

    Starts at 1 and adds 1 for each descendant node whose type is in
    *branching_types*.
    """
    count = 1
    stack = [root_node]
    while stack:
        n = stack.pop()
        if n.type in branching_types:
            count += 1
        stack.extend(n.children)
    return count


def _add_external(name: str, module_v: Vertex, graph: PropertyGraph) -> None:
    """Add an import edge from *module_v* to a shared external vertex for *name*."""
    existing = next(
        (v for v in graph.vertices("external") if v.properties.get("name") == name),
        None,
    )
    target = existing or graph.add_vertex("external", name=name)
    graph.add_edge("imports", module_v, target)


class LanguageAdapter(ABC):
    """Parse source files of a specific language into a PropertyGraph.

    Subclasses must define a class-level ``extensions`` frozenset and implement
    :meth:`parse`.  Optionally override :meth:`is_available` to report whether
    the required tree-sitter language package is installed.
    """

    extensions: frozenset[str]

    @classmethod
    def is_available(cls) -> bool:
        """Return True if the required tree-sitter language package is installed."""
        return True

    @abstractmethod
    def parse(self, path: Path, graph: PropertyGraph) -> Vertex:
        """Parse *path*, populate *graph*, and return the module vertex."""
        ...
