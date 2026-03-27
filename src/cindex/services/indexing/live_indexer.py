"""Background indexing service for keeping a SQLite code index fresh."""

from __future__ import annotations

import logging
from pathlib import Path
from threading import Event
from threading import Lock
from threading import Thread

from watchfiles import Change
from watchfiles import watch

from cindex.services.indexing.sqlite_store import get_indexed_files
from cindex.services.indexing.sqlite_store import persist_graph
from cindex.services.indexing.sqlite_store import persist_indexed_files
from cindex.services.indexing.sqlite_store import persist_vertex_embeddings
from cindex.services.indexing.sqlite_store import reindex_file
from cindex.services.indexing.walker import index_directory

logger = logging.getLogger(__name__)


class LiveIndexer:
    """Maintain a SQLite graph index in the background for one source tree."""

    def __init__(
        self,
        *,
        root: Path,
        db_path: Path,
        model_name: str | None = None,
        cache_folder: str | None = None,
    ) -> None:
        self.root = root.resolve()
        self.db_path = db_path.resolve()
        self.model_name = model_name
        self.cache_folder = cache_folder
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._write_lock = Lock()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = Thread(target=self._watch_loop, name="cindex-live-indexer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def synchronize(self) -> None:
        current_files = {
            str(path.resolve()): path.stat().st_mtime_ns
            for path in self.root.rglob("*")
            if path.is_file() and path.suffix == ".py"
        }
        indexed_files = get_indexed_files(self.db_path)

        if not indexed_files:
            self.rebuild_full_index()
            return

        pending_paths = {
            Path(path)
            for path, mtime_ns in current_files.items()
            if indexed_files.get(path) != mtime_ns
        }
        pending_paths.update(Path(path) for path in indexed_files.keys() - current_files.keys())

        if pending_paths:
            self.reindex_paths(pending_paths)

    def rebuild_full_index(self) -> None:
        with self._write_lock:
            logger.info("Building full index for %s", self.root)
            graph = index_directory(self.root)
            persist_graph(graph, self.db_path, append=False)
            if self.model_name:
                persist_vertex_embeddings(
                    graph,
                    self.db_path,
                    model_name=self.model_name,
                    cache_folder=self.cache_folder,
                    append=False,
                    initialize_vector_extension=True,
                )
            persist_indexed_files(self.root, self.db_path, append=False)

    def reindex_paths(self, paths: set[Path]) -> None:
        with self._write_lock:
            for path in sorted({path.resolve() for path in paths}):
                if path.suffix != ".py":
                    continue
                logger.info("Incrementally reindexing %s", path)
                reindex_file(
                    self.db_path,
                    path,
                    model_name=self.model_name,
                    cache_folder=self.cache_folder,
                    initialize_vector_extension=True,
                )

    def _watch_loop(self) -> None:
        for changes in watch(self.root, stop_event=self._stop_event, recursive=True, debounce=1000):
            pending_paths = {
                Path(path)
                for change, path in changes
                if _include_watch_change(Change(change), path)
            }
            if pending_paths:
                self.reindex_paths(pending_paths)



def _include_watch_change(change: Change, path: str) -> bool:
    file_path = Path(path)
    return file_path.suffix == ".py" or (change == Change.deleted and file_path.suffix == ".py")
