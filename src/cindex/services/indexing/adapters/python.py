"""Tree-sitter based Python source file parser.

Extracts modules, classes, functions, and imports from a single .py file and
inserts them as vertices/edges into a PropertyGraph.

Vertex labels
─────────────
    module    – one per file; properties: name (stem), path, docstring?
    class     – one per class definition; properties: name, path, line, docstring?
    function  – one per function/method; properties: name, path, line, docstring?, source
  external  – one per unique imported module name; properties: name

Edge labels
───────────
  defines    – module → class or module → function (top-level)
  has_method – class  → function (method inside a class body)
  imports    – module → external
"""
from __future__ import annotations

import ast
import logging
from pathlib import Path

from cindex.services.indexing.graph import PropertyGraph, Vertex
from cindex.services.indexing.adapters.base import LanguageAdapter

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


class PythonAdapter(LanguageAdapter):
    """Language adapter for Python (.py) files."""

    extensions = frozenset({".py"})

    def parse(self, path: Path, graph: PropertyGraph) -> Vertex:
        return parse_python_file(path, graph)


def _compute_cyclomatic_complexity(node) -> int:
    """Compute McCabe (cyclomatic) complexity for a function/method node (tree-sitter node)."""
    # Complexity starts at 1
    complexity = 1
    stack = [node]
    while stack:
        n = stack.pop()
        t = n.type
        # Branching nodes per McCabe
        if t in {
            "if_statement", "elif_clause", "else_clause", "for_statement", "while_statement",
            "except_clause", "with_statement", "case_clause", "match_statement",
            "assert_statement", "try_statement", "finally_clause",
            "comprehension", "conditional_expression"
        }:
            complexity += 1
        # Each except handler
        if t == "try_statement":
            for child in n.children:
                if child.type == "except_clause":
                    complexity += 1
        # Recurse into children
        stack.extend(n.children)
    return complexity


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

    module_props: dict[str, object] = {"name": path.stem, "path": str(path)}
    module_docstring = _extract_docstring(tree.root_node, source)
    if module_docstring is not None:
        module_props["docstring"] = module_docstring

    module_v = graph.add_vertex(LABEL_MODULE, **module_props)
    _visit_children(tree.root_node, graph, module_v, None, source)
    return module_v


# ── AST visitors ──────────────────────────────────────────────────────────────

def _text(node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _extract_docstring(node, source: bytes) -> str | None:
    body = node.child_by_field_name("body")
    if body is None:
        body = node
    if not body.children:
        return None

    for child in body.children:
        if child.type in {"comment", "\n"}:
            continue
        if child.type != "expression_statement":
            return None

        text = _text(child, source).strip()
        if not text:
            return None
        try:
            value = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return None
        return value if isinstance(value, str) else None

    return None


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

    elif t in ("function_definition", "async_function_definition"):
        owner = class_v if class_v is not None else module_v
        _handle_function(node, graph, module_v, owner, source)
        # Recursively visit the function body to find nested imports and functions
        body = node.child_by_field_name("body")
        if body:
            for child in body.children:
                _visit(child, graph, module_v, owner, source)

    elif t in ("import_statement", "import_from_statement"):
        _handle_import(node, graph, module_v, source)

    elif t == "decorated_definition":
        # Walk the decorator's children to find the actual definition.
        for child in node.children:
            if child.type in ("class_definition", "function_definition", "async_function_definition"):
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

    class_props: dict[str, object] = {
        "name": name,
        "path": module_v.properties["path"],
        "line": node.start_point[0] + 1,
    }
    class_docstring = _extract_docstring(node, source)
    if class_docstring is not None:
        class_props["docstring"] = class_docstring

    class_v = graph.add_vertex(LABEL_CLASS, **class_props)
    graph.add_edge(EDGE_DEFINES, module_v, class_v)

    # Walk only the class body so we capture methods without double-visiting.
    body = node.child_by_field_name("body")
    if body:
        for child in body.children:
            if child.type in ("function_definition", "async_function_definition"):
                _handle_function(child, graph, module_v, class_v, source)
                # Recursively visit the function body for nested imports/functions
                func_body = child.child_by_field_name("body")
                if func_body:
                    for subchild in func_body.children:
                        _visit(subchild, graph, module_v, class_v, source)
            elif child.type == "decorated_definition":
                for subchild in child.children:
                    if subchild.type in ("function_definition", "async_function_definition"):
                        _handle_function(subchild, graph, module_v, class_v, source)
                        func_body = subchild.child_by_field_name("body")
                        if func_body:
                            for subsubchild in func_body.children:
                                _visit(subsubchild, graph, module_v, class_v, source)


def _handle_function(
    node,
    graph: PropertyGraph,
    module_v: Vertex,
    owner: Vertex,
    source: bytes,
) -> None:
    name_node = node.child_by_field_name("name")
    name = _text(name_node, source) if name_node else "<unknown>"

    func_props: dict[str, object] = {
        "name": name,
        "path": module_v.properties["path"],
        "line": node.start_point[0] + 1,
        "source": _text(node, source),
        "complexity": _compute_cyclomatic_complexity(node),
    }
    function_docstring = _extract_docstring(node, source)
    if function_docstring is not None:
        func_props["docstring"] = function_docstring

    func_v = graph.add_vertex(LABEL_FUNCTION, **func_props)
    edge_label = EDGE_HAS_METHOD if owner.label == LABEL_CLASS else EDGE_DEFINES
    graph.add_edge(edge_label, owner, func_v)
    # Function body recursion is now handled in _visit to detect late imports and nested functions.


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
