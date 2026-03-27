"""Tree-sitter based Python source file parser.

Extracts modules, classes, functions, and imports from a single .py file and
inserts them as vertices/edges into a PropertyGraph.

Vertex labels
─────────────
  module    – one per file; properties: name (stem), path
  class     – one per class definition; properties: name, path, line
  function  – one per function/method; properties: name, path, line
  external  – one per unique imported module name; properties: name

Edge labels
───────────
  defines    – module → class or module → function (top-level)
  has_method – class  → function (method inside a class body)
  imports    – module → external
"""
from __future__ import annotations

import logging
from pathlib import Path

from cindex.services.indexing.graph import PropertyGraph, Vertex

logger = logging.getLogger(__name__)

# Vertex labels
LABEL_MODULE = "module"
LABEL_CLASS = "class"
LABEL_FUNCTION = "function"
LABEL_EXTERNAL = "external"

# Edge labels
EDGE_DEFINES = "defines"
EDGE_HAS_METHOD = "has_method"
EDGE_IMPORTS = "imports"


def parse_python_file(path: Path, graph: PropertyGraph) -> Vertex:
    """Parse *path* and add all discovered entities to *graph*.

    Returns the module vertex for the file.
    """
    try:
        import tree_sitter_python as tspython
        from tree_sitter import Language, Parser
    except ImportError as exc:
        raise ImportError(
            "tree-sitter-python is required for parsing. "
            "Install it with: pip install tree-sitter tree-sitter-python"
        ) from exc

    source = path.read_bytes()
    language = Language(tspython.language())
    parser = Parser(language)
    tree = parser.parse(source)

    module_v = graph.add_vertex(LABEL_MODULE, name=path.stem, path=str(path))
    _visit_children(tree.root_node, graph, module_v, None, source)
    return module_v


# ── AST visitors ──────────────────────────────────────────────────────────────

def _text(node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _visit_children(
    node,
    graph: PropertyGraph,
    module_v: Vertex,
    class_v: Vertex | None,
    source: bytes,
) -> None:
    for child in node.children:
        _visit(child, graph, module_v, class_v, source)


def _visit(
    node,
    graph: PropertyGraph,
    module_v: Vertex,
    class_v: Vertex | None,
    source: bytes,
) -> None:
    """Dispatch to the appropriate handler or recurse into generic nodes."""
    t = node.type

    if t == "class_definition":
        _handle_class(node, graph, module_v, source)

    elif t == "function_definition":
        owner = class_v if class_v is not None else module_v
        _handle_function(node, graph, module_v, owner, source)

    elif t in ("import_statement", "import_from_statement"):
        _handle_import(node, graph, module_v, source)

    elif t == "decorated_definition":
        # Walk the decorator's children to find the actual definition.
        for child in node.children:
            if child.type in ("class_definition", "function_definition"):
                _visit(child, graph, module_v, class_v, source)

    else:
        # Generic node — recurse but don't descend into class/function bodies
        # (those are handled by their dedicated handlers).
        _visit_children(node, graph, module_v, class_v, source)


def _handle_class(
    node, graph: PropertyGraph, module_v: Vertex, source: bytes
) -> None:
    name_node = node.child_by_field_name("name")
    name = _text(name_node, source) if name_node else "<unknown>"

    class_v = graph.add_vertex(
        LABEL_CLASS,
        name=name,
        path=module_v.properties["path"],
        line=node.start_point[0] + 1,
    )
    graph.add_edge(EDGE_DEFINES, module_v, class_v)

    # Walk only the class body so we capture methods without double-visiting.
    body = node.child_by_field_name("body")
    if body:
        for child in body.children:
            if child.type == "function_definition":
                _handle_function(child, graph, module_v, class_v, source)
            elif child.type == "decorated_definition":
                for subchild in child.children:
                    if subchild.type == "function_definition":
                        _handle_function(subchild, graph, module_v, class_v, source)


def _handle_function(
    node,
    graph: PropertyGraph,
    module_v: Vertex,
    owner: Vertex,
    source: bytes,
) -> None:
    name_node = node.child_by_field_name("name")
    name = _text(name_node, source) if name_node else "<unknown>"

    func_v = graph.add_vertex(
        LABEL_FUNCTION,
        name=name,
        path=module_v.properties["path"],
        line=node.start_point[0] + 1,
    )
    edge_label = EDGE_HAS_METHOD if owner.label == LABEL_CLASS else EDGE_DEFINES
    graph.add_edge(edge_label, owner, func_v)
    # We intentionally do not recurse into the function body in Phase 1.


def _handle_import(
    node, graph: PropertyGraph, module_v: Vertex, source: bytes
) -> None:
    """Record import edges to shared external-module vertices."""
    for name in _extract_import_names(node, source):
        # Re-use a single external vertex per unique module name.
        existing = next(
            (
                v
                for v in graph.vertices(LABEL_EXTERNAL)
                if v.properties.get("name") == name
            ),
            None,
        )
        target_v = existing if existing is not None else graph.add_vertex(
            LABEL_EXTERNAL, name=name
        )
        graph.add_edge(EDGE_IMPORTS, module_v, target_v)


def _extract_import_names(node, source: bytes) -> list[str]:
    """Return the top-level module name(s) referenced by an import node."""
    names: list[str] = []

    if node.type == "import_statement":
        for child in node.children:
            if child.type == "dotted_name":
                names.append(_text(child, source))
            elif child.type == "aliased_import":
                name_node = child.child_by_field_name("name")
                if name_node:
                    names.append(_text(name_node, source))

    elif node.type == "import_from_statement":
        module_node = node.child_by_field_name("module_name")
        if module_node:
            names.append(_text(module_node, source))

    return names
