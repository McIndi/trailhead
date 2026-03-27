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


def create_app(
    *,
    default_model: str = DEFAULT_MODEL,
    cache_dir: str | None = None,
    sqlite_db: str | None = None,
    preload_default_model: bool = True,
) -> FastAPI:
    configured_sqlite_db = str(Path(sqlite_db).resolve()) if sqlite_db else None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if preload_default_model:
            preload_embedding_model(default_model, cache_folder=cache_dir)
        yield

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
            "preloaded": preload_default_model,
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
