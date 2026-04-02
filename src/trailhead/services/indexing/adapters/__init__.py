"""Language adapter registry for trailhead.

Built-in adapters are auto-registered on import *only when their required
tree-sitter language package is installed*.  Third-party adapters can be
added at runtime via :func:`register`.

Example — registering a custom adapter::

    from trailhead.services.indexing.adapters import register
    from my_package import KotlinAdapter

    register(KotlinAdapter())
"""
from __future__ import annotations

import logging

from trailhead.services.indexing.adapters.base import LanguageAdapter
from trailhead.services.indexing.adapters.registry import (
    get_adapter,
    parse_file,
    register,
    supported_suffixes,
)
from trailhead.services.indexing.adapters.python import PythonAdapter
from trailhead.services.indexing.adapters.javascript import JavaScriptAdapter
from trailhead.services.indexing.adapters.typescript import TypeScriptAdapter, TSXAdapter
from trailhead.services.indexing.adapters.rust import RustAdapter
from trailhead.services.indexing.adapters.go import GoAdapter
from trailhead.services.indexing.adapters.java import JavaAdapter
from trailhead.services.indexing.adapters.csharp import CSharpAdapter
from trailhead.services.indexing.adapters.c import CAdapter
from trailhead.services.indexing.adapters.cpp import CppAdapter
from trailhead.services.indexing.adapters.ruby import RubyAdapter
from trailhead.services.indexing.adapters.php import PHPAdapter
from trailhead.services.indexing.adapters.bash import BashAdapter
from trailhead.services.indexing.adapters.html import HTMLAdapter

logger = logging.getLogger(__name__)

# Python is always registered (it's a hard dependency).
register(PythonAdapter())

# All other adapters are optional — only registered when their tree-sitter
# language package is actually installed.
_OPTIONAL_ADAPTERS = [
    JavaScriptAdapter,
    TypeScriptAdapter,
    TSXAdapter,
    RustAdapter,
    GoAdapter,
    JavaAdapter,
    CSharpAdapter,
    CAdapter,
    CppAdapter,
    RubyAdapter,
    PHPAdapter,
    BashAdapter,
    HTMLAdapter,
]

for _cls in _OPTIONAL_ADAPTERS:
    if _cls.is_available():
        register(_cls())
        logger.debug("Registered language adapter: %s", _cls.__name__)

__all__ = [
    "LanguageAdapter",
    "PythonAdapter",
    "JavaScriptAdapter",
    "TypeScriptAdapter",
    "TSXAdapter",
    "RustAdapter",
    "GoAdapter",
    "JavaAdapter",
    "CSharpAdapter",
    "CAdapter",
    "CppAdapter",
    "RubyAdapter",
    "PHPAdapter",
    "BashAdapter",
    "HTMLAdapter",
    "get_adapter",
    "parse_file",
    "register",
    "supported_suffixes",
]
