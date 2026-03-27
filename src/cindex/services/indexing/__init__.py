"""Code indexing service — property graph construction from source trees."""

from cindex.services.indexing.graph import PropertyGraph
from cindex.services.indexing.graph_query import search_vertices
from cindex.services.indexing.graph_query import traverse_graph
from cindex.services.indexing.live_indexer import LiveIndexer
from cindex.services.indexing.sqlite_store import get_indexed_files
from cindex.services.indexing.sqlite_store import load_graph
from cindex.services.indexing.sqlite_store import persist_graph
from cindex.services.indexing.sqlite_store import persist_indexed_files
from cindex.services.indexing.sqlite_store import persist_vertex_embeddings
from cindex.services.indexing.sqlite_store import reindex_file
from cindex.services.indexing.sqlite_store import vector_to_blob
from cindex.services.indexing.walker import index_directory

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
]
