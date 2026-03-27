"""In-memory property graph following the Apache TinkerPop property graph model.

Vertices have a label and arbitrary key/value properties.
Edges connect two vertices, carry a label, and may also have properties.
PropertyGraph is the top-level container.

This implementation uses in-memory Python structures.  A later phase will
expose traversal via gremlinpython connected to a real TinkerPop/Gremlin server.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Vertex:
    id: str
    label: str
    properties: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        name = self.properties.get("name", self.id)
        return f"v[{self.label}:{name}]"


@dataclass
class Edge:
    id: str
    label: str
    out_v: Vertex
    in_v: Vertex
    properties: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"e[{self.out_v!r}-{self.label}->{self.in_v!r}]"


class PropertyGraph:
    """In-memory TinkerPop-compatible property graph."""

    def __init__(self) -> None:
        self._vertices: dict[str, Vertex] = {}
        self._edges: list[Edge] = []

    # ── mutation ──────────────────────────────────────────────────────────────

    def add_vertex(self, label: str, **properties: Any) -> Vertex:
        v = Vertex(id=str(uuid.uuid4()), label=label, properties=dict(properties))
        self._vertices[v.id] = v
        return v

    def add_edge(
        self, label: str, out_v: Vertex, in_v: Vertex, **properties: Any
    ) -> Edge:
        e = Edge(
            id=str(uuid.uuid4()),
            label=label,
            out_v=out_v,
            in_v=in_v,
            properties=dict(properties),
        )
        self._edges.append(e)
        return e

    # ── read ──────────────────────────────────────────────────────────────────

    def get_vertex(self, vertex_id: str) -> Vertex | None:
        return self._vertices.get(vertex_id)

    def vertices(self, label: str | None = None) -> list[Vertex]:
        if label is None:
            return list(self._vertices.values())
        return [v for v in self._vertices.values() if v.label == label]

    def edges(self, label: str | None = None) -> list[Edge]:
        if label is None:
            return list(self._edges)
        return [e for e in self._edges if e.label == label]

    def out_edges(self, vertex: Vertex, label: str | None = None) -> list[Edge]:
        return [
            e
            for e in self._edges
            if e.out_v.id == vertex.id and (label is None or e.label == label)
        ]

    def in_edges(self, vertex: Vertex, label: str | None = None) -> list[Edge]:
        return [
            e
            for e in self._edges
            if e.in_v.id == vertex.id and (label is None or e.label == label)
        ]
