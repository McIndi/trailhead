"""Tree-sitter based HTML source file adapter.

HTML has no functions or classes in the traditional sense.  Instead the adapter
extracts the constructs that are most useful for code navigation and search:

* The document itself as a ``module`` vertex, with the ``<title>`` text as its
  ``docstring`` when present.
* Every element that carries an ``id`` attribute as a ``function`` vertex
  (named anchors, sections, components — the things you link to with
  ``#fragment`` and look up in an IDE).
* ``<script src="...">`` and ``<link href="...">`` declarations as ``external``
  vertices (resource dependencies, the closest HTML equivalent of imports).

Inline ``<script>`` and ``<style>`` content is **not** parsed; only the
element structure and referenced URLs are indexed.

Vertex labels produced
──────────────────────
    module    – one per file; docstring set to <title> text when present
    function  – elements with an id attribute; name = id value
    external  – <script src>, <link href> paths

Edge labels produced
────────────────────
    defines  – module → function
    imports  – module → external
"""
from __future__ import annotations

import logging
from pathlib import Path

from cindex.services.indexing.graph import PropertyGraph, Vertex
from cindex.services.indexing.adapters.base import (
    LanguageAdapter,
    _add_external,
    _node_text,
)

logger = logging.getLogger(__name__)


class HTMLAdapter(LanguageAdapter):
    """Language adapter for HTML (.html, .htm) files."""

    extensions = frozenset({".html", ".htm"})

    @classmethod
    def is_available(cls) -> bool:
        try:
            import tree_sitter_html  # noqa: F401
            return True
        except ImportError:
            return False

    def parse(self, path: Path, graph: PropertyGraph) -> Vertex:
        try:
            import tree_sitter_html as tshtml
            from tree_sitter import Language, Parser
        except ImportError as exc:
            raise ImportError(
                "tree-sitter-html is required. "
                "Install it with: pip install tree-sitter-html"
            ) from exc

        source = path.read_bytes()
        language = Language(tshtml.language())
        parser = Parser(language)
        tree = parser.parse(source)

        module_props: dict[str, object] = {"name": path.stem, "path": str(path)}
        title = _find_title(tree.root_node, source)
        if title:
            module_props["docstring"] = title

        module_v = graph.add_vertex("module", **module_props)
        _visit(tree.root_node, graph, module_v, source)
        return module_v


# ── AST visitors ──────────────────────────────────────────────────────────────

def _visit(node, graph: PropertyGraph, module_v: Vertex, src: bytes) -> None:
    t = node.type

    if t in ("element", "script_element", "style_element"):
        start = _start_tag(node)
        if start is not None:
            attrs = _attributes(start, src)
            tag = _tag_name(start, src)

            # Named element → function vertex
            elem_id = attrs.get("id")
            if elem_id:
                func_v = graph.add_vertex(
                    "function",
                    name=elem_id,
                    path=module_v.properties["path"],
                    line=node.start_point[0] + 1,
                    source=f"<{tag} id=\"{elem_id}\">",
                )
                graph.add_edge("defines", module_v, func_v)

            # Resource dependency → external vertex
            if tag == "script":
                src_attr = attrs.get("src")
                if src_attr:
                    _add_external(src_attr, module_v, graph)
            elif tag == "link":
                href = attrs.get("href")
                if href:
                    _add_external(href, module_v, graph)

        # Recurse into element children
        for child in node.children:
            _visit(child, graph, module_v, src)

    else:
        for child in node.children:
            _visit(child, graph, module_v, src)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _start_tag(element_node):
    """Return the start_tag / self_closing_tag child of an element, or None."""
    for child in element_node.children:
        if child.type in ("start_tag", "self_closing_tag"):
            return child
    return None


def _tag_name(start_tag_node, src: bytes) -> str:
    for child in start_tag_node.children:
        if child.type == "tag_name":
            return _node_text(child, src).lower()
    return ""


def _attributes(start_tag_node, src: bytes) -> dict[str, str]:
    """Return a dict of {attribute_name: attribute_value} for a start_tag node."""
    attrs: dict[str, str] = {}
    for child in start_tag_node.children:
        if child.type == "attribute":
            name: str | None = None
            value: str | None = None
            for sub in child.children:
                if sub.type == "attribute_name":
                    name = _node_text(sub, src).lower()
                elif sub.type == "attribute_value":
                    value = _node_text(sub, src).strip("\"' \t")
            if name is not None:
                attrs[name] = value or ""
    return attrs


def _find_title(root_node, src: bytes) -> str | None:
    """Walk the tree to find the first <title> element and return its text."""
    stack = [root_node]
    while stack:
        node = stack.pop()
        if node.type == "element":
            start = _start_tag(node)
            if start is not None and _tag_name(start, src) == "title":
                # The text content is a direct child of the element
                for child in node.children:
                    if child.type == "text":
                        text = _node_text(child, src).strip()
                        if text:
                            return text
        stack.extend(node.children)
    return None
