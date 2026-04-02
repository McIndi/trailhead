"""Code indexing service — property graph construction from source trees."""

from trailhead.services.indexing.graph import PropertyGraph
from trailhead.services.indexing.graph_query import search_vertices
from trailhead.services.indexing.graph_query import traverse_graph
from trailhead.services.indexing.live_indexer import LiveIndexer
from trailhead.services.indexing.query_templates import get_query_template
from trailhead.services.indexing.query_templates import list_query_templates
from trailhead.services.indexing.sqlite_store import get_indexed_files
from trailhead.services.indexing.sqlite_store import load_graph
from trailhead.services.indexing.sqlite_store import persist_graph
from trailhead.services.indexing.sqlite_store import persist_indexed_files
from trailhead.services.indexing.sqlite_store import persist_vertex_embeddings
from trailhead.services.indexing.sqlite_store import reindex_file
from trailhead.services.indexing.sqlite_store import vector_to_blob
from trailhead.services.indexing.walker import index_directory

__all__ = [
    "PropertyGraph",
    "LiveIndexer",
    "index_directory",
    "persist_graph",
    "persist_indexed_files",
    "persist_vertex_embeddings",
    "reindex_file",
    "vector_to_blob",
    "get_indexed_files",
    "load_graph",
    "search_vertices",
    "traverse_graph",
    "list_query_templates",
    "get_query_template",
]
