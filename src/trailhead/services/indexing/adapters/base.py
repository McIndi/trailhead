"""Abstract base class and shared utilities for language adapters."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

from trailhead.services.indexing.graph import PropertyGraph, Vertex


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


def _preceding_doc_comment(node, source: bytes, *, prefixes: tuple[str, ...] = ("/**", "///", "//", "#")) -> str | None:
    """Return a doc comment that immediately precedes *node* in the parent's child list.

    Only comments whose stripped text starts with one of *prefixes* are returned,
    so ordinary implementation comments are excluded.  Returns None when no
    qualifying comment is found.
    """
    parent = node.parent
    if parent is None:
        return None
    siblings = parent.children
    idx = next((i for i, c in enumerate(siblings) if c.id == node.id), None)
    if idx is None or idx == 0:
        return None
    prev = siblings[idx - 1]
    if prev.type not in ("comment", "line_comment", "block_comment", "multiline_comment"):
        return None
    text = _node_text(prev, source).strip()
    if any(text.startswith(p) for p in prefixes):
        return text
    return None


def _collect_calls_ts(
    tree_root,
    graph: PropertyGraph,
    module_v: Vertex,
    source: bytes,
    *,
    func_node_types: frozenset[str],
    call_node_types: frozenset[str],
    get_callee_name: Callable,
) -> None:
    """Generic call-edge collector for tree-sitter adapters.

    Walks *tree_root* tracking the innermost enclosing function vertex
    (looked up by file path + start line).  For every node whose type is in
    *call_node_types* found inside a function body, calls
    *get_callee_name(call_node, source)* to extract a bare function/method
    name, then resolves it against all function vertices in *graph* (same-file
    preferred) and adds a ``calls`` edge.  Duplicate (caller, callee) pairs are
    deduplicated.

    Args:
        func_node_types: Tree-sitter node types that introduce a new function
            scope (e.g. ``{"function_declaration", "method_declaration"}``).
        call_node_types: Tree-sitter node types for call sites
            (e.g. ``{"call_expression"}``).
        get_callee_name: ``(call_node, source) -> str | None`` — extracts the
            bare callee name from a call node.  Return ``None`` to skip.
    """
    file_path = module_v.properties["path"]
    all_funcs = graph.vertices("function")

    local_by_name: dict[str, Vertex] = {}
    global_by_name: dict[str, Vertex] = {}
    for v in all_funcs:
        name = v.properties.get("name", "")
        if not name:
            continue
        if v.properties.get("path") == file_path:
            local_by_name[name] = v
        else:
            global_by_name.setdefault(name, v)

    caller_by_loc: dict[tuple[str, int], Vertex] = {
        (v.properties["path"], v.properties["line"]): v
        for v in all_funcs
        if "path" in v.properties and "line" in v.properties
    }

    seen_edges: set[tuple[str, str]] = set()

    def _resolve(name: str) -> Vertex | None:
        return local_by_name.get(name) or global_by_name.get(name)

    def _walk(node, caller_v: Vertex | None) -> None:
        t = node.type
        if t in func_node_types:
            line = node.start_point[0] + 1
            caller_v = caller_by_loc.get((file_path, line), caller_v)
            for child in node.children:
                _walk(child, caller_v)
        elif t in call_node_types and caller_v is not None:
            name = get_callee_name(node, source)
            if name:
                callee_v = _resolve(name)
                if callee_v and callee_v.id != caller_v.id:
                    key = (caller_v.id, callee_v.id)
                    if key not in seen_edges:
                        seen_edges.add(key)
                        graph.add_edge("calls", caller_v, callee_v)
            # Recurse so nested calls (e.g. f(g(x))) are also captured.
            for child in node.children:
                _walk(child, caller_v)
        else:
            for child in node.children:
                _walk(child, caller_v)

    _walk(tree_root, None)


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
