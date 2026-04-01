"""FastAPI app for warm embedding and query workflows."""

from __future__ import annotations

import re
import sqlite3
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Query
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pydantic import Field

from cindex.server.rate_limit import RateLimitMiddleware

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
    text: str = Field(max_length=10_000)


class BatchEmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=100)


class SqlRequest(BaseModel):
    sql: str = Field(max_length=10_000)


@dataclass(frozen=True)
class AppConfig:
    default_model: str
    cache_dir: str | None
    configured_sqlite_db: str | None
    configured_watch_directory: str | None
    preload_default_model: bool
    run_indexer: bool


def _build_lifespan(config: AppConfig):
    live_indexer: LiveIndexer | None = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal live_indexer

        if config.preload_default_model:
            preload_embedding_model(config.default_model, cache_folder=config.cache_dir)

        if config.run_indexer:
            if not config.configured_sqlite_db or not config.configured_watch_directory:
                raise RuntimeError("run_indexer requires both sqlite_db and watch_directory")
            live_indexer = LiveIndexer(
                root=Path(config.configured_watch_directory),
                db_path=Path(config.configured_sqlite_db),
                model_name=config.default_model,
                cache_folder=config.cache_dir,
            )
            live_indexer.synchronize()
            live_indexer.start()
            app.state.live_indexer = live_indexer

        try:
            yield
        finally:
            if live_indexer is not None:
                live_indexer.stop()

    return lifespan


def _get_config(request: Request) -> AppConfig:
    return request.app.state.config


def health(request: Request) -> dict[str, object]:
    config = _get_config(request)
    return {
        "status": "ok",
        "default_model": config.default_model,
        "sqlite_db": config.configured_sqlite_db,
        "watch_directory": config.configured_watch_directory,
        "preloaded": config.preload_default_model,
        "indexer_enabled": config.run_indexer,
    }


def embed(request: Request, payload: EmbedRequest) -> dict[str, object]:
    config = _get_config(request)
    embedding = generate_embedding(
        payload.text,
        config.default_model,
        cache_folder=config.cache_dir,
    )
    return {
        "model": config.default_model,
        "dimensions": len(embedding),
        "embedding": embedding,
    }


def embed_batch(request: Request, payload: BatchEmbedRequest) -> dict[str, object]:
    config = _get_config(request)
    embeddings = generate_embeddings(
        payload.texts,
        config.default_model,
        cache_folder=config.cache_dir,
    )
    return {
        "model": config.default_model,
        "count": len(embeddings),
        "dimensions": len(embeddings[0]) if embeddings else 0,
        "embeddings": embeddings,
    }


def query_sql(request: Request, payload: SqlRequest) -> dict[str, object]:
    config = _get_config(request)
    try:
        columns, rows = execute_sql_query(_require_sqlite_db(config.configured_sqlite_db), payload.sql)
    except (ValueError, RuntimeError, OSError, sqlite3.Error) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"columns": columns, "rows": rows}


def query_templates() -> dict[str, object]:
    templates = list_query_templates()
    return {"count": len(templates), "templates": templates}


def query_template(name: str) -> dict[str, str]:
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


def run_query_template(request: Request, name: str) -> dict[str, object]:
    config = _get_config(request)
    try:
        template = get_query_template(name)
        columns, rows = execute_sql_query(_require_sqlite_db(config.configured_sqlite_db), template.sql)
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


def query_similar(
    request: Request,
    text: str = Query(max_length=2_000),
    k: int = Query(default=10, ge=1, le=100),
    label: str = Query(default=None, max_length=100),
    include_external: bool = Query(default=False),
) -> dict[str, object]:
    config = _get_config(request)
    try:
        rows = find_similar_vertices(
            _require_sqlite_db(config.configured_sqlite_db),
            text,
            model_name=config.default_model,
            cache_folder=config.cache_dir,
            k=k,
            label=label,
            exclude_external=not include_external,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"count": len(rows), "rows": rows}


def graph_vertices_get(
    request: Request,
    name: str = Query(default=None, max_length=200),
    label: str = Query(default=None, max_length=100),
    path_contains: str = Query(default=None, max_length=500),
    limit: int = Query(default=20, ge=1, le=200),
    include_external: bool = Query(default=False),
) -> dict[str, object]:
    config = _get_config(request)
    try:
        rows = search_vertices(
            _require_sqlite_db(config.configured_sqlite_db),
            name=name,
            label=label,
            path_contains=path_contains,
            limit=limit,
            exclude_external=not include_external,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"count": len(rows), "rows": rows}


def graph_traverse_get(
    request: Request,
    vertex_id: str = Query(max_length=200),
    direction: str = Query(default="both", max_length=10),
    depth: int = Query(default=1, ge=1, le=5),
    edge_labels: str = Query(default=None, max_length=500),
    max_vertices: int = Query(default=100, ge=1, le=500),
) -> dict[str, object]:
    config = _get_config(request)
    try:
        edge_labels_list = [e.strip() for e in edge_labels.split(",") if e.strip()] if edge_labels else None
        return traverse_graph(
            _require_sqlite_db(config.configured_sqlite_db),
            vertex_id=vertex_id,
            direction=direction,
            depth=depth,
            edge_labels=edge_labels_list,
            max_vertices=max_vertices,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def ui_dashboard(request: Request) -> str:
    config = _get_config(request)
    return _render_ui_template(
        _load_ui_template(),
        {
            "DEFAULT_MODEL": config.default_model,
            "SQLITE_DB": config.configured_sqlite_db or "Not configured",
        },
    )


def _register_routes(app: FastAPI) -> None:
    app.add_api_route("/api/health", health, methods=["GET"])
    app.add_api_route("/api/embed", embed, methods=["POST"])
    app.add_api_route("/api/embed/batch", embed_batch, methods=["POST"])
    app.add_api_route("/api/query/sql", query_sql, methods=["POST"])
    app.add_api_route("/api/query/templates", query_templates, methods=["GET"])
    app.add_api_route("/api/query/templates/{name}", query_template, methods=["GET"])
    app.add_api_route("/api/query/templates/{name}/run", run_query_template, methods=["POST"])
    app.add_api_route("/api/query/similar", query_similar, methods=["GET"])
    app.add_api_route("/api/graph/vertices", graph_vertices_get, methods=["GET"])
    app.add_api_route("/api/graph/traverse", graph_traverse_get, methods=["GET"])
    app.add_api_route(
        "/ui/",
        ui_dashboard,
        methods=["GET"],
        response_class=HTMLResponse,
        include_in_schema=False,
    )


def create_app(
    *,
    default_model: str = DEFAULT_MODEL,
    cache_dir: str | None = None,
    sqlite_db: str | None = None,
    watch_directory: str | None = None,
    preload_default_model: bool = True,
    run_indexer: bool = False,
    cors_origins: list[str] | None = None,
    rate_limit: int = 120,
) -> FastAPI:
    config = AppConfig(
        default_model=default_model,
        cache_dir=cache_dir,
        configured_sqlite_db=str(Path(sqlite_db).resolve()) if sqlite_db else None,
        configured_watch_directory=str(Path(watch_directory).resolve()) if watch_directory else None,
        preload_default_model=preload_default_model,
        run_indexer=run_indexer,
    )

    app = FastAPI(
        title="cindex API",
        version="0.1.0",
        description="Warm-model API for embeddings and SQLite-backed code queries.",
        lifespan=_build_lifespan(config),
    )
    app.state.config = config

    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )

    if rate_limit > 0:
        app.add_middleware(RateLimitMiddleware, requests_per_minute=rate_limit)

    _register_routes(app)

    return app


def _require_sqlite_db(configured_sqlite_db: str | None) -> Path:
    if not configured_sqlite_db:
        raise HTTPException(status_code=503, detail="SQLite database is not configured.")
    return Path(configured_sqlite_db)


@lru_cache(maxsize=1)
def _load_ui_template() -> str:
    template_path = Path(__file__).resolve().parent / "templates" / "query_ui.html"
    return template_path.read_text(encoding="utf-8")


def _render_ui_template(template: str, replacements: dict[str, str]) -> str:
    """Substitute {{KEY}} placeholders in a single regex pass.

    All replacements happen simultaneously, so no substituted value can
    accidentally match and trigger a second substitution.
    """
    pattern = re.compile("|".join(re.escape(f"{{{{{k}}}}}") for k in replacements))
    return pattern.sub(lambda m: replacements[m.group(0)[2:-2]], template)
