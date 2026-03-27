"""Code indexing service — property graph construction from source trees."""

from cindex.services.indexing.graph import PropertyGraph
from cindex.services.indexing.sqlite_store import load_graph
from cindex.services.indexing.sqlite_store import persist_graph
from cindex.services.indexing.sqlite_store import persist_vertex_embeddings
from cindex.services.indexing.walker import index_directory

__all__ = [
	"PropertyGraph",
	"index_directory",
	"persist_graph",
	"persist_vertex_embeddings",
	"load_graph",
]
