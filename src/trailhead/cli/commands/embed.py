from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from trailhead.services.config import get_cache_dir
from trailhead.services.embeddings import generate_embedding

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RECOMMENDED_MODELS: tuple[str, ...] = (
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
)

logger = logging.getLogger(__name__)


def configure_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "embed",
        help="Generate a sentence embedding.",
        description="Generate text embeddings with a sentence-transformers model.",
    )
    parser.add_argument(
        "text",
        help="The text to embed.",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Hugging Face model name to load. Recommended models: "
            + ", ".join(RECOMMENDED_MODELS)
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help=(
            "Directory to cache downloaded models. Overrides TRAILHEAD_CACHE_DIR env var. "
            "If not set, uses Hugging Face default cache."
        ),
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    cache_dir = args.cache_dir or get_cache_dir()
    if cache_dir:
        cache_dir = str(Path(cache_dir).resolve())
    logger.info(f"Using cache directory: {cache_dir or 'Hugging Face default'}")

    embedding = generate_embedding(args.text, args.model, cache_folder=cache_dir)
    print(json.dumps(embedding))
    return 0
