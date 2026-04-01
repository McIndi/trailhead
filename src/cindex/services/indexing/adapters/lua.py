"""Tree-sitter based Lua source file adapter.

Extracts modules and functions. Lua has no formal class system — the common
patterns (function declarations, local functions, and table-assigned functions)
are all indexed as top-level function vertices.

Vertex labels produced
──────────────────────
    module   – one per file
    function – function_declaration, local_function, and table-method assignments
    external – require() arguments

Edge labels produced
────────────────────
    defines – module → function
    imports – module → external
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
    "if_statement", "elseif", "else", "while_statement",
    "repeat_statement", "for_statement", "for_in_statement",
})


class LuaAdapter(LanguageAdapter):
    """Language adapter for Lua (.lua) files."""

    extensions = frozenset({".lua"})

    @classmethod
    def is_available(cls) -> bool:
        try:
            import tree_sitter_lua  # noqa: F401
            return True
        except ImportError:
            return False

    def parse(self, path: Path, graph: PropertyGraph) -> Vertex:
        try:
            import tree_sitter_lua as tslua
            from tree_sitter import Language, Parser
        except ImportError as exc:
            raise ImportError(
                "tree-sitter-lua is required. "
                "Install it with: pip install tree-sitter-lua"
            ) from exc

        source = path.read_bytes()
        language = Language(tslua.language())
        parser = Parser(language)
        tree = parser.parse(source)

        module_v = graph.add_vertex("module", name=path.stem, path=str(path))
        _visit(tree.root_node, graph, module_v, source)
        return module_v


# ── AST visitors ──────────────────────────────────────────────────────────────

def _visit(node, graph: PropertyGraph, module_v: Vertex, src: bytes, scope_v: Vertex | None = None) -> None:
    t = node.type

    if t == "function_declaration":
        _handle_named_func(node, graph, module_v, src)

    elif t == "local_function":
        _handle_local_func(node, graph, module_v, src)

    elif t == "assignment_statement":
        # M.foo = function() ... end
        _handle_assignment(node, graph, module_v, src)

    elif t == "local_variable_declaration":
        # local foo = function() ... end
        _handle_local_assignment(node, graph, module_v, src)

    elif t == "function_call":
        _handle_call(node, graph, module_v, src)

    else:
        for child in node.children:
            _visit(child, graph, module_v, src, scope_v=scope_v)


def _handle_named_func(node, graph: PropertyGraph, module_v: Vertex, src: bytes) -> None:
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    func_v = _add_function(node, _node_text(name_node, src), graph, module_v, src)
    body = node.child_by_field_name("body")
    if body is not None:
        for child in body.children:
            _visit(child, graph, module_v, src, scope_v=func_v)


def _handle_local_func(node, graph: PropertyGraph, module_v: Vertex, src: bytes) -> None:
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    func_v = _add_function(node, _node_text(name_node, src), graph, module_v, src)
    body = node.child_by_field_name("body")
    if body is not None:
        for child in body.children:
            _visit(child, graph, module_v, src, scope_v=func_v)


def _handle_assignment(node, graph: PropertyGraph, module_v: Vertex, src: bytes) -> None:
    # Check if the value is a function
    varlist = node.child_by_field_name("variable_list")
    explist = node.child_by_field_name("expression_list")
    if varlist is None or explist is None:
        return
    for exp in explist.children:
        if exp.type == "function_definition":
            # Use the first variable name as the function name
            for var in varlist.children:
                if var.type not in (",",):
                    name = _node_text(var, src).strip()
                    if name:
                        func_v = _add_function(exp, name, graph, module_v, src)
                        body = exp.child_by_field_name("body")
                        if body is not None:
                            for child in body.children:
                                _visit(child, graph, module_v, src, scope_v=func_v)
                    break
            break


def _handle_local_assignment(node, graph: PropertyGraph, module_v: Vertex, src: bytes) -> None:
    # Handles: local foo = function() ... end
    # Walk direct children: identifiers before "=" are names, function_definitions
    # after "=" are the values to index.  Also handles a "variable_list" wrapper
    # that some grammar versions emit.
    names: list[str] = []
    func_nodes: list = []
    seen_equals = False

    for child in node.children:
        t = child.type
        if t == "=":
            seen_equals = True
        elif not seen_equals:
            if t == "identifier":
                names.append(_node_text(child, src))
            elif t == "variable_list":
                for sub in child.children:
                    if sub.type == "identifier":
                        names.append(_node_text(sub, src))
        else:
            if t == "function_definition":
                func_nodes.append(child)
            elif t == "expression_list":
                for sub in child.children:
                    if sub.type == "function_definition":
                        func_nodes.append(sub)

    for i, func in enumerate(func_nodes):
        if i < len(names):
            func_v = _add_function(func, names[i], graph, module_v, src)
            body = func.child_by_field_name("body")
            if body is not None:
                for child in body.children:
                    _visit(child, graph, module_v, src, scope_v=func_v)


def _add_function(node, name: str, graph: PropertyGraph, module_v: Vertex, src: bytes) -> Vertex:
    func_v = graph.add_vertex(
        "function",
        name=name,
        path=module_v.properties["path"],
        line=node.start_point[0] + 1,
        source=_node_text(node, src),
        complexity=_complexity(node, _BRANCHING),
    )
    graph.add_edge("defines", module_v, func_v)
    return func_v


def _handle_call(node, graph: PropertyGraph, module_v: Vertex, src: bytes) -> None:
    """Handle require('module') calls."""
    name_node = node.child_by_field_name("name")
    if name_node is None or _node_text(name_node, src) != "require":
        return
    args = node.child_by_field_name("args")
    if args is None:
        return
    for child in args.children:
        if child.type == "string":
            name = _node_text(child, src).strip("'\"() \t")
            if name:
                _add_external(name, module_v, graph)
            break
