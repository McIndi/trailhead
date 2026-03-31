"""FastAPI app for warm embedding and query workflows."""

from __future__ import annotations

import sqlite3
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
from cindex.services.indexing.query_templates import get_query_template
from cindex.services.indexing.query_templates import list_query_templates

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

    @app.get("/api/health")
    def health() -> dict[str, object]:
        """
        Health check endpoint.
        Returns basic status and configuration information for the API server.

        Returns:
            dict: Status, model, database, and indexer configuration.
        """
        return {
            "status": "ok",
            "default_model": default_model,
            "sqlite_db": configured_sqlite_db,
            "watch_directory": configured_watch_directory,
            "preloaded": preload_default_model,
            "indexer_enabled": run_indexer,
        }

    @app.post("/api/embed")
    def embed(request: EmbedRequest) -> dict[str, object]:
        """
        Generate an embedding vector for a single text string.

        Args:
            request (EmbedRequest):
                text: The input string to embed.
                model: The embedding model to use.
                cache_dir: Optional cache directory for model files.

        Returns:
            dict: Model name, embedding vector, and dimension count.
        """
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

    @app.post("/api/embed/batch")
    def embed_batch(request: BatchEmbedRequest) -> dict[str, object]:
        """
        Generate embedding vectors for a batch of text strings.

        Args:
            request (BatchEmbedRequest):
                texts: List of input strings to embed.
                model: The embedding model to use.
                cache_dir: Optional cache directory for model files.

        Returns:
            dict: Model name, embedding vectors, count, and dimension.
        """
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

    @app.post("/api/query/sql")
    def query_sql(request: SqlRequest) -> dict[str, object]:
        """
        Execute an arbitrary SQL query against the indexed code database.

        Args:
            request (SqlRequest):
                sql: The SQL query string to execute.

        Returns:
            dict: Columns and rows from the query result.
        """
        try:
            columns, rows = execute_sql_query(_require_sqlite_db(configured_sqlite_db), request.sql)
        except (ValueError, RuntimeError, OSError, sqlite3.Error) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"columns": columns, "rows": rows}

    @app.get("/api/query/templates")
    def query_templates() -> dict[str, object]:
        """
        List all available starter SQL query templates.

        Returns:
            dict: Count and metadata for each query template.
        """
        templates = list_query_templates()
        return {"count": len(templates), "templates": templates}

    @app.get("/api/query/templates/{name}")
    def query_template(name: str) -> dict[str, str]:
        """
        Get metadata and SQL for a specific starter query template.

        Args:
            name (str): Name of the query template.

        Returns:
            dict: Template metadata and SQL string.
        """
        try:
            template = get_query_template(name)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {
            "name": template.name,
            "category": template.category,
            "title": template.title,
            "description": template.description,
            "sql": template.sql,
        }

    @app.post("/api/query/templates/{name}/run")
    def run_query_template(name: str) -> dict[str, object]:
        """
        Execute a starter query template by name and return the results.

        Args:
            name (str): Name of the query template to run.

        Returns:
            dict: Template metadata, columns, and rows from the query result.
        """
        try:
            template = get_query_template(name)
            columns, rows = execute_sql_query(_require_sqlite_db(configured_sqlite_db), template.sql)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (RuntimeError, OSError, sqlite3.Error) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "template": {
                "name": template.name,
                "category": template.category,
                "title": template.title,
                "description": template.description,
            },
            "columns": columns,
            "rows": rows,
        }

    @app.get("/api/query/similar")
    def query_similar(
        text: str,
        model: str = DEFAULT_MODEL,
        cache_dir: str = None,
        k: int = 10,
        label: str = None
    ) -> dict[str, object]:
        """
        Perform a semantic similarity search for code symbols.

        Args (as query parameters):
            text (str): Query text to search for similar code symbols.
            model (str, optional): Embedding model to use. Default is the configured model.
            cache_dir (str, optional): Optional cache directory for model files.
            k (int, optional): Number of top results to return. Default is 10.
            label (str, optional): Filter by symbol label (e.g., function, class).

        Returns:
            dict: Count and list of matching code symbols.
        """
        try:
            rows = find_similar_vertices(
                _require_sqlite_db(configured_sqlite_db),
                text,
                model_name=model or default_model,
                cache_folder=cache_dir,
                k=k,
                label=label,
            )
        except (ValueError, RuntimeError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"count": len(rows), "rows": rows}



    @app.get("/api/graph/vertices")
    def graph_vertices_get(
        name: str = None,
        label: str = None,
        path_contains: str = None,
        limit: int = 20
    ) -> dict[str, object]:
        """
        Search for code vertices (modules, classes, functions, etc.) in the code index.

        Args (as query parameters):
            name (str, optional): Filter by symbol name (substring match).
            label (str, optional): Filter by symbol label (e.g., function, class).
            path_contains (str, optional): Filter by file path substring.
            limit (int, optional): Maximum number of results to return. Default is 20.

        Returns:
            dict: Count and list of matching vertices.
        """
        try:
            rows = search_vertices(
                _require_sqlite_db(configured_sqlite_db),
                name=name,
                label=label,
                path_contains=path_contains,
                limit=limit,
            )
        except (ValueError, RuntimeError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"count": len(rows), "rows": rows}



    @app.get("/api/graph/traverse")
    def graph_traverse_get(
        vertex_id: str,
        direction: str = "both",
        depth: int = 1,
        edge_labels: str = None,
        max_vertices: int = 100
    ) -> dict[str, object]:
        """
        Traverse the code property graph from a given vertex.

        Args (as query parameters):
            vertex_id (str): The ID of the starting vertex.
            direction (str, optional): 'in', 'out', or 'both'. Default is 'both'.
            depth (int, optional): Number of hops to traverse. Default is 1.
            edge_labels (str, optional): Comma-separated edge labels to follow.
            max_vertices (int, optional): Maximum vertices to return. Default is 100.

        Returns:
            dict: Graph traversal result (vertices, edges, etc.).
        """
        try:
            edge_labels_list = [e.strip() for e in edge_labels.split(",") if e.strip()] if edge_labels else None
            return traverse_graph(
                _require_sqlite_db(configured_sqlite_db),
                vertex_id=vertex_id,
                direction=direction,
                depth=depth,
                edge_labels=edge_labels_list,
                max_vertices=max_vertices,
            )
        except (ValueError, RuntimeError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/ui/", response_class=HTMLResponse, include_in_schema=False)
    def ui_dashboard() -> str:
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
