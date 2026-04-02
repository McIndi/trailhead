# trailhead

Command-line code indexing and semantic search tool. It parses source files into a **property graph** (modules, classes, functions, and their relationships) stored in SQLite, generates text embeddings with sentence-transformers, and exposes everything through a CLI and HTTP API.

- Single CLI command: `th`
- Text embeddings powered by sentence-transformers (models cached locally)
- Polyglot code indexing via tree-sitter (Python built-in; 12 additional languages optional)
- Property graph persisted in a single SQLite file with optional vector search
- Warm-model FastAPI server keeps the embedding model loaded in memory
- Background file watcher incrementally re-indexes on change
- Interactive browser UI for querying and visualizing the code graph

## Requirements

- Python 3.10+

## Install

Create and activate a virtual environment, then install the project in editable mode with dev dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

### Language support

Python is supported out of the box. Additional languages are installed as optional extras. Only the packages you install will be active; missing ones are silently skipped at startup.

Install individual languages:

```powershell
pip install -e .[javascript]
pip install -e .[typescript]
pip install -e .[rust]
pip install -e .[go]
pip install -e .[java]
pip install -e .[csharp]
pip install -e .[c]
pip install -e .[cpp]
pip install -e .[ruby]
pip install -e .[php]
pip install -e .[bash]
pip install -e .[html]
```

Or install everything at once:

```powershell
pip install -e .[all-languages]
```

| Extra | Language | File extensions |
|-------|----------|-----------------|
| `python` *(built-in)* | Python | `.py` |
| `javascript` | JavaScript | `.js` `.mjs` `.cjs` |
| `typescript` | TypeScript / TSX | `.ts` `.tsx` |
| `rust` | Rust | `.rs` |
| `go` | Go | `.go` |
| `java` | Java | `.java` |
| `csharp` | C# | `.cs` |
| `c` | C | `.c` `.h` |
| `cpp` | C++ | `.cpp` `.cc` `.cxx` `.hpp` `.hxx` `.h++` |
| `ruby` | Ruby | `.rb` |
| `php` | PHP | `.php` |
| `bash` | Bash / Shell | `.sh` `.bash` |
| `html` | HTML | `.html` `.htm` |

## Quick start

The typical workflow is: index your source tree once, then serve and query it.

```powershell
# 1. Index a project (writes .cindex/db.sqlite by default)
th index . --sqlite-db ./.cindex/graph.db --embed-model sentence-transformers/all-MiniLM-L6-v2

# 2. Start the server (watches for changes, keeps embeddings warm)
th serve . --sqlite-db ./.cindex/graph.db --model sentence-transformers/all-MiniLM-L6-v2

# 3. Open the browser UI
start http://localhost:8000

# 4. Or query from the CLI or another terminal
th query similar "HTTP route registration"
curl "http://localhost:8000/api/query/similar?text=HTTP+route+registration"
```

The server re-indexes changed files automatically in the background. You do not need to re-run `th index` while the server is running.

## Usage

### embed

Generate an embedding for a piece of text:

```powershell
th embed "A short sentence to embed"
th embed "A short sentence to embed" --model sentence-transformers/all-mpnet-base-v2
```

The command prints the embedding as a JSON array of floats.

Optional cache override:

```powershell
$env:CINDEX_CACHE_DIR = "C:\models\cache"
th embed "A short sentence to embed"
th embed "A short sentence to embed" --cache-dir "C:\another\cache"
```

### index

Index a directory of source files. The graph is persisted to `.cindex/db.sqlite` by default (smart sync: full build on first run, incremental on subsequent runs):

```powershell
th index .
```

Use `--in-memory` to build the graph without writing to disk and print a summary:

```powershell
th index . --in-memory
th index . --in-memory --output json
```

Watch for file changes and reindex incrementally (Ctrl-C to stop):

```powershell
th index . --watch
th index . --sqlite-db ./.cindex/graph.db --embed-model sentence-transformers/all-MiniLM-L6-v2 --watch
```

Use a custom database path or add embeddings:

```powershell
th index . --sqlite-db ./.cindex/graph.db
th index . --sqlite-db ./.cindex/graph.db --embed-model sentence-transformers/all-MiniLM-L6-v2
th index . --sqlite-db ./.cindex/graph.db --embed-model sentence-transformers/all-MiniLM-L6-v2 --embed-cache-dir C:\models\cache
```

When `sqlite-vector` can be loaded, trailhead also initializes vector search for the `vertex_embeddings.embedding` column. If extension loading is unavailable on your platform build, embeddings are still stored as Float32 BLOBs in SQLite.

### serve

Run the warm-model API server with a background indexer. The server watches the source tree, keeps the SQLite graph fresh, and reuses the loaded embedding model across index updates. The database defaults to `.cindex/db.sqlite` under the watched directory:

```powershell
th serve .
th serve . --model sentence-transformers/all-MiniLM-L6-v2
th serve . --sqlite-db ./.cindex/graph.db --model sentence-transformers/all-MiniLM-L6-v2
```

The browser UI is available at `http://localhost:8000` once the server starts.

### query

Run a read-only SQL query against the SQLite database (defaults to `./.cindex/db.sqlite`):

```powershell
th query sql --sql "SELECT label, COUNT(*) AS n FROM vertices GROUP BY label ORDER BY label"
th query sql --sqlite-db ./.cindex/graph.db --sql "SELECT label, COUNT(*) AS n FROM vertices GROUP BY label ORDER BY label"
```

Run a semantic similarity query against stored vertex embeddings:

```powershell
th query similar "find sqlite vector initialization code"
th query similar "find sqlite vector initialization code" --sqlite-db ./.cindex/graph.db
th query similar "graph persistence" --sqlite-db ./.cindex/graph.db --label function --k 5 --output json
```

## HTTP API

When the server is running, the full API schema is available at:

```
http://localhost:8000/openapi.json
http://localhost:8000/docs
```

The schema documents every endpoint, parameter name, type, default, and constraint. **Check it first** before probing endpoints manually.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Browser UI |
| `GET` | `/api/health` | Server status and configuration |
| `POST` | `/api/embed` | Embed a single text string |
| `POST` | `/api/embed/batch` | Embed multiple texts |
| `POST` | `/api/query/sql` | Run a read-only SQL query |
| `GET` | `/api/query/templates` | List built-in query templates |
| `GET` | `/api/query/templates/{name}` | Get a template's SQL |
| `POST` | `/api/query/templates/{name}/run` | Run a template against the database |
| `GET` | `/api/query/similar` | Semantic similarity search (parameter: `text`, `k`) |
| `GET` | `/api/graph/vertices` | Search vertices by name, label, or path |
| `GET` | `/api/graph/traverse` | Traverse the graph from a vertex |

### SQL schema

The two core tables are:

**`vertices`** — one row per code symbol:

| Column | Type | Notes |
|--------|------|-------|
| `id` | TEXT | UUID, used for graph traversal |
| `label` | TEXT | `module`, `class`, `function`, `external` |
| `name` | TEXT | Symbol name |
| `path` | TEXT | Absolute file path |
| `line` | INTEGER | Line number (null for modules) |
| `complexity` | INTEGER | McCabe complexity (functions only) |
| `properties_json` | TEXT | JSON blob with `source`, `docstring`, and all other properties |

**`edges`** — relationships between vertices:

| Column | Type | Notes |
|--------|------|-------|
| `id` | TEXT | UUID |
| `label` | TEXT | `defines`, `has_method`, `imports`, `calls` |
| `out_v_id` | TEXT | Source vertex id |
| `in_v_id` | TEXT | Target vertex id |
| `properties_json` | TEXT | Always `{}` currently |

Edge labels and their meaning:

| Label | Meaning |
|-------|---------|
| `defines` | Module → class or function it defines |
| `has_method` | Class → method |
| `imports` | Module → external symbol it imports |
| `calls` | Function → function it calls |

`source` and `docstring` live inside `properties_json` rather than as top-level columns. To filter on them in SQL, use `json_extract`:

```sql
-- Functions whose source mentions "HTTPException"
SELECT name, path, line
FROM vertices
WHERE label = 'function'
  AND json_extract(properties_json, '$.source') LIKE '%HTTPException%'

-- Functions with a docstring
SELECT name, path
FROM vertices
WHERE label = 'function'
  AND json_extract(properties_json, '$.docstring') IS NOT NULL
```

### HTTP query examples

```powershell
# Health check
curl http://localhost:8000/api/health

# Embed text
curl -X POST http://localhost:8000/api/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "hello world"}'

# Semantic search — note the parameter is "k", not "limit"
curl "http://localhost:8000/api/query/similar?text=route+registration&k=5"

# Filter semantic search to functions only
curl "http://localhost:8000/api/query/similar?text=route+registration&k=5&label=function"

# SQL query
curl -X POST http://localhost:8000/api/query/sql \
  -H "Content-Type: application/json" \
  -d '{"sql": "SELECT label, COUNT(*) AS n FROM vertices GROUP BY label"}'

# Find a vertex by name, then get its id for traversal
curl "http://localhost:8000/api/graph/vertices?name=ui_dashboard&label=function"

# Traverse outward along call edges only (shows what a function calls)
curl "http://localhost:8000/api/graph/traverse?vertex_id=<id>&direction=out&depth=2&edge_labels=calls"

# Traverse inward along call edges only (shows what calls a function)
curl "http://localhost:8000/api/graph/traverse?vertex_id=<id>&direction=in&depth=2&edge_labels=calls"

# Run a built-in query template
curl http://localhost:8000/api/query/templates
curl -X POST http://localhost:8000/api/query/templates/function_complexity/run
```

### Built-in query templates

Templates are pre-built SQL queries runnable without writing any SQL. Categories:

| Category | Templates |
|----------|-----------|
| `quality` | `function_complexity`, `missing_docstrings`, `undocumented_public_api`, `todo_fixme_inventory` |
| `testing` | `symbols_not_represented_by_tests`, `test_coverage_ratio_by_file`, `largest_untested_symbols` |
| `architecture` | `duplicate_symbol_names`, `dependency_hotspots`, `external_dependency_pressure` |
| `calls` | `most_called_functions`, `call_graph_hubs` |
| `data_health` | `missing_source_for_functions`, `orphan_edges` |

### Typical workflow for code exploration

1. **Find a starting point** — use semantic search or `/api/graph/vertices?name=...` to locate a vertex and grab its `id`.
2. **Understand its call chain** — traverse outward with `edge_labels=calls` to see what it calls; inward to see its callers.
3. **Understand its structure** — traverse with `edge_labels=defines,has_method` to see what a module or class contains.
4. **Run quality checks** — use the built-in templates for complexity, missing docs, or dependency hotspots without writing SQL.
5. **Ad-hoc queries** — use `/api/query/sql` with `json_extract` to filter on source content, docstrings, or any property.

## Tests

```powershell
pytest
```

## Project Layout

```text
.
|-- pyproject.toml
|-- README.md
|-- src/
|   `-- cindex/
|       |-- __init__.py
|       |-- __main__.py
|       |-- cli/
|       |   |-- __init__.py
|       |   |-- __main__.py
|       |   |-- app.py
|       |   `-- commands/
|       |       |-- __init__.py
|       |       |-- embed.py
|       |       |-- index.py
|       |       |-- query.py
|       |       `-- serve.py
|       |-- server/
|       |   |-- __init__.py
|       |   |-- __main__.py
|       |   |-- app.py
|       |   `-- templates/
|       |       `-- query_ui.html
|       `-- services/
|           |-- config/
|           |   `-- cache.py
|           |-- indexing/
|           |   |-- __init__.py
|           |   |-- graph.py
|           |   |-- graph_query.py
|           |   |-- live_indexer.py
|           |   |-- parser.py          # re-exports parse_python_file (backwards compat)
|           |   |-- query.py
|           |   |-- sqlite_store.py
|           |   |-- walker.py
|           |   `-- adapters/          # language adapter registry
|           |       |-- __init__.py    # auto-registers available adapters
|           |       |-- base.py        # LanguageAdapter ABC + shared utilities
|           |       |-- registry.py    # extension → adapter map, parse_file()
|           |       |-- python.py      # Python (built-in)
|           |       |-- javascript.py  # JavaScript (optional)
|           |       |-- typescript.py  # TypeScript / TSX (optional)
|           |       |-- rust.py        # Rust (optional)
|           |       |-- go.py          # Go (optional)
|           |       |-- java.py        # Java (optional)
|           |       |-- csharp.py      # C# (optional)
|           |       |-- c.py           # C (optional)
|           |       |-- cpp.py         # C++ (optional)
|           |       |-- ruby.py        # Ruby (optional)
|           |       |-- php.py         # PHP (optional)
|           |       |-- bash.py        # Bash / Shell (optional)
|           |       `-- html.py        # HTML (optional)
|           `-- embeddings/
|               |-- generator.py
|               `-- model_store.py
`-- tests/
    |-- conftest.py
    |-- test_indexing.py
    |-- test_query.py
    |-- test_server.py
    `-- test_smoke.py
```

## Adding a custom language adapter

Any language with a tree-sitter Python binding can be supported in three steps:

```python
# 1. Create your adapter (e.g. my_adapters/kotlin.py)
from cindex.services.indexing.adapters.base import LanguageAdapter, _node_text, _complexity
from cindex.services.indexing.graph import PropertyGraph, Vertex
from pathlib import Path

class KotlinAdapter(LanguageAdapter):
    extensions = frozenset({".kt", ".kts"})

    @classmethod
    def is_available(cls) -> bool:
        try:
            import tree_sitter_kotlin  # noqa: F401
            return True
        except ImportError:
            return False

    def parse(self, path: Path, graph: PropertyGraph) -> Vertex:
        import tree_sitter_kotlin as tskotlin
        from tree_sitter import Language, Parser
        source = path.read_bytes()
        language = Language(tskotlin.language())
        parser = Parser(language)
        tree = parser.parse(source)
        module_v = graph.add_vertex("module", name=path.stem, path=str(path))
        # ... walk tree and add vertices/edges ...
        return module_v

# 2. Register it at startup (e.g. in your app's __init__ or conftest)
from cindex.services.indexing.adapters import register
register(KotlinAdapter())

# 3. Done — th index, serve, and query all pick it up automatically.
```

What each adapter should produce:

| Vertex label | Meaning | Required properties |
|---|---|---|
| `module` | one per source file | `name`, `path` |
| `class` | class / struct / interface / trait | `name`, `path`, `line` |
| `function` | function / method | `name`, `path`, `line`, `source`, `complexity` |
| `external` | imported module name | `name` |

Edges: `defines` (module→class, module→function), `has_method` (class→function), `imports` (module→external), `calls` (function→function).
