"""Background indexing service for keeping a SQLite code index fresh."""

from __future__ import annotations

import logging
import queue
from pathlib import Path
from threading import Event
from threading import Thread

from watchfiles import Change
from watchfiles import watch

from cindex.services.indexing.adapters import supported_suffixes
from cindex.services.indexing.sqlite_store import get_indexed_files
from cindex.services.indexing.sqlite_store import persist_graph
from cindex.services.indexing.sqlite_store import persist_indexed_files
from cindex.services.indexing.sqlite_store import persist_vertex_embeddings
from cindex.services.indexing.sqlite_store import reindex_file
from cindex.services.indexing.walker import index_directory

logger = logging.getLogger(__name__)


class LiveIndexer:
    """Maintain a SQLite graph index in the background for one source tree.

    Two threads are used after :meth:`start` is called:

    * **watcher** — runs :func:`watchfiles.watch` and enqueues sets of changed
      paths.  It is intentionally kept free of heavy work so that file-system
      events are never dropped.
    * **worker** — drains the queue and calls :meth:`reindex_paths` for each
      batch.  Embedding generation happens here, keeping it off the watcher
      thread.

    :meth:`synchronize` is always called *before* :meth:`start`, so it runs on
    the main thread with no contention.
    """

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
        self._change_queue: queue.Queue[set[Path]] = queue.Queue()
        self._watch_thread: Thread | None = None
        self._worker_thread: Thread | None = None

    def start(self) -> None:
        if self._watch_thread is not None and self._watch_thread.is_alive():
            return

        self._stop_event.clear()
        self._worker_thread = Thread(
            target=self._worker_loop,
            name="cindex-indexer-worker",
            daemon=True,
        )
        self._watch_thread = Thread(
            target=self._watch_loop,
            name="cindex-indexer-watcher",
            daemon=True,
        )
        self._worker_thread.start()
        self._watch_thread.start()
        logger.info("Live indexer started, watching %s", self.root)

    def stop(self) -> None:
        logger.info("Live indexer stopping")
        self._stop_event.set()
        if self._watch_thread is not None:
            self._watch_thread.join(timeout=5)
        if self._worker_thread is not None:
            # Give the worker time to finish the current reindex (may include
            # embedding generation, which can be slow).
            self._worker_thread.join(timeout=30)
        logger.info("Live indexer stopped")

    def synchronize(self) -> None:
        current_files = {
            str(path.resolve()): path.stat().st_mtime_ns
            for path in self.root.rglob("*")
            if path.is_file() and path.suffix in supported_suffixes()
        }
        indexed_files = get_indexed_files(self.db_path)
        logger.info(
            "Synchronizing index: %d current source files, %d previously indexed",
            len(current_files),
            len(indexed_files),
        )

        if not indexed_files:
            logger.info("No indexed files found — performing full index build")
            self.rebuild_full_index()
            return

        changed = {
            Path(path)
            for path, mtime_ns in current_files.items()
            if indexed_files.get(path) != mtime_ns
        }
        deleted = {Path(path) for path in indexed_files.keys() - current_files.keys()}
        pending_paths = changed | deleted

        if deleted:
            logger.info("Detected %d deleted file(s): %s", len(deleted), [p.name for p in deleted])
        if changed:
            logger.info("Detected %d changed/new file(s): %s", len(changed), [p.name for p in changed])

        if pending_paths:
            self.reindex_paths(pending_paths)
        else:
            logger.info("Index is up to date, no reindexing needed")

    def rebuild_full_index(self) -> None:
        logger.info("Building full index for %s", self.root)
        graph = index_directory(self.root)
        vertex_count = sum(1 for _ in graph.vertices())
        logger.info("Parsed %d vertices, persisting graph", vertex_count)
        persist_graph(graph, self.db_path, append=False)
        if self.model_name:
            logger.info("Generating embeddings (model=%s)", self.model_name)
            persist_vertex_embeddings(
                graph,
                self.db_path,
                model_name=self.model_name,
                cache_folder=self.cache_folder,
                append=False,
                initialize_vector_extension=True,
            )
        persist_indexed_files(self.root, self.db_path, append=False)
        logger.info("Full index build complete")

    def reindex_paths(self, paths: set[Path]) -> None:
        py_paths = sorted(p.resolve() for p in paths if p.suffix in supported_suffixes())
        if not py_paths:
            return
        for path in py_paths:
            exists = path.exists()
            action = "reindexing" if exists else "removing deleted"
            logger.info("Incrementally %s %s", action, path.name)
            vertex_count, embedding_rows = reindex_file(
                self.db_path,
                path,
                model_name=self.model_name,
                cache_folder=self.cache_folder,
                initialize_vector_extension=True,
            )
            if exists:
                logger.debug("  -> %d vertices, %d embeddings", vertex_count, embedding_rows)
        logger.info("Incremental reindex complete (%d file(s))", len(py_paths))

    def _watch_loop(self) -> None:
        logger.info("File watcher active on %s", self.root)
        for changes in watch(self.root, stop_event=self._stop_event, recursive=True, debounce=1000):
            pending_paths = {
                Path(path)
                for change, path in changes
                if _include_watch_change(Change(change), path)
            }
            if pending_paths:
                logger.info(
                    "Watcher detected changes in: %s",
                    ", ".join(p.name for p in sorted(pending_paths)),
                )
                self._change_queue.put(pending_paths)
        logger.info("File watcher exited")

    def _worker_loop(self) -> None:
        logger.info("Indexer worker started")
        while True:
            try:
                paths = self._change_queue.get(timeout=0.5)
            except queue.Empty:
                if self._stop_event.is_set():
                    break
                continue
            try:
                self.reindex_paths(paths)
            finally:
                self._change_queue.task_done()
        logger.info("Indexer worker exited")


def _include_watch_change(change: Change, path: str) -> bool:
    file_path = Path(path)
    return file_path.suffix in supported_suffixes() or (
        change == Change.deleted and file_path.suffix in supported_suffixes()
    )
