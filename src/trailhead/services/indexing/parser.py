"""Backwards-compatible re-export; implementation lives in adapters/python.py."""
from trailhead.services.indexing.adapters.python import parse_python_file

__all__ = ["parse_python_file"]
