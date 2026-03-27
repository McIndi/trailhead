"""FastAPI app for warm embedding and query workflows."""

from __future__ import annotations

from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pydantic import Field

from cindex.services.embeddings import generate_embedding
from cindex.services.embeddings import generate_embeddings
from cindex.services.embeddings import preload_embedding_model
from cindex.services.indexing import LiveIndexer
from cindex.services.indexing import search_vertices
from cindex.services.indexing import traverse_graph
from cindex.services.indexing.query import execute_sql_query
from cindex.services.indexing.query import find_similar_vertices

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbedRequest(BaseModel):
    text: str
    model: str = DEFAULT_MODEL
    cache_dir: str | None = None


class BatchEmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1)
    model: str = DEFAULT_MODEL
    cache_dir: str | None = None


class SimilarRequest(BaseModel):
    text: str
    model: str | None = None
    cache_dir: str | None = None
    k: int = Field(default=10, ge=1)
    label: str | None = None


class SqlRequest(BaseModel):
    sql: str


class GraphSearchRequest(BaseModel):
    name: str | None = None
    label: str | None = None
    path_contains: str | None = None
    limit: int = Field(default=20, ge=1, le=200)


class GraphTraverseRequest(BaseModel):
    vertex_id: str
    direction: str = Field(default="both")
    depth: int = Field(default=1, ge=1, le=5)
    edge_labels: list[str] | None = None
    max_vertices: int = Field(default=100, ge=1, le=500)


def create_app(
    *,
    default_model: str = DEFAULT_MODEL,
    cache_dir: str | None = None,
    sqlite_db: str | None = None,
    watch_directory: str | None = None,
    preload_default_model: bool = True,
    run_indexer: bool = False,
) -> FastAPI:
    configured_sqlite_db = str(Path(sqlite_db).resolve()) if sqlite_db else None
    configured_watch_directory = str(Path(watch_directory).resolve()) if watch_directory else None
    live_indexer: LiveIndexer | None = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal live_indexer

        if preload_default_model:
            preload_embedding_model(default_model, cache_folder=cache_dir)

        if run_indexer:
            if not configured_sqlite_db or not configured_watch_directory:
                raise RuntimeError("run_indexer requires both sqlite_db and watch_directory")
            live_indexer = LiveIndexer(
                root=Path(configured_watch_directory),
                db_path=Path(configured_sqlite_db),
                model_name=default_model,
                cache_folder=cache_dir,
            )
            live_indexer.synchronize()
            live_indexer.start()
            app.state.live_indexer = live_indexer

        try:
            yield
        finally:
            if live_indexer is not None:
                live_indexer.stop()

    app = FastAPI(
        title="cindex API",
        version="0.1.0",
        description="Warm-model API for embeddings and SQLite-backed code queries.",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "default_model": default_model,
            "sqlite_db": configured_sqlite_db,
            "watch_directory": configured_watch_directory,
            "preloaded": preload_default_model,
            "indexer_enabled": run_indexer,
        }

    @app.post("/embed")
    def embed(request: EmbedRequest) -> dict[str, object]:
        embedding = generate_embedding(
            request.text,
            request.model,
            cache_folder=request.cache_dir,
        )
        return {
            "model": request.model,
            "dimensions": len(embedding),
            "embedding": embedding,
        }

    @app.post("/embed/batch")
    def embed_batch(request: BatchEmbedRequest) -> dict[str, object]:
        embeddings = generate_embeddings(
            request.texts,
            request.model,
            cache_folder=request.cache_dir,
        )
        return {
            "model": request.model,
            "count": len(embeddings),
            "dimensions": len(embeddings[0]) if embeddings else 0,
            "embeddings": embeddings,
        }

    @app.post("/query/sql")
    def query_sql(request: SqlRequest) -> dict[str, object]:
        try:
            columns, rows = execute_sql_query(_require_sqlite_db(configured_sqlite_db), request.sql)
        except (ValueError, RuntimeError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"columns": columns, "rows": rows}

    @app.post("/query/similar")
    def query_similar(request: SimilarRequest) -> dict[str, object]:
        try:
            rows = find_similar_vertices(
                _require_sqlite_db(configured_sqlite_db),
                request.text,
                model_name=request.model or default_model,
                cache_folder=request.cache_dir,
                k=request.k,
                label=request.label,
            )
        except (ValueError, RuntimeError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"count": len(rows), "rows": rows}

    @app.post("/graph/vertices")
    def graph_vertices(request: GraphSearchRequest) -> dict[str, object]:
        try:
            rows = search_vertices(
                _require_sqlite_db(configured_sqlite_db),
                name=request.name,
                label=request.label,
                path_contains=request.path_contains,
                limit=request.limit,
            )
        except (ValueError, RuntimeError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"count": len(rows), "rows": rows}

    @app.post("/graph/traverse")
    def graph_traverse(request: GraphTraverseRequest) -> dict[str, object]:
        try:
            return traverse_graph(
                _require_sqlite_db(configured_sqlite_db),
                vertex_id=request.vertex_id,
                direction=request.direction,
                depth=request.depth,
                edge_labels=request.edge_labels,
                max_vertices=request.max_vertices,
            )
        except (ValueError, RuntimeError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
    def ui() -> str:
        return _load_ui_template().replace("{{DEFAULT_MODEL}}", default_model).replace(
            "{{SQLITE_DB}}", configured_sqlite_db or "Not configured"
        )

    return app


def _require_sqlite_db(configured_sqlite_db: str | None) -> Path:
    if not configured_sqlite_db:
        raise HTTPException(status_code=503, detail="SQLite database is not configured.")
    return Path(configured_sqlite_db)


@lru_cache(maxsize=1)
def _load_ui_template() -> str:
    template_path = Path(__file__).resolve().parent / "templates" / "query_ui.html"
    return template_path.read_text(encoding="utf-8")
