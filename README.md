# trailhead

Command-line code indexing and text embedding tool with:

- a single CLI command: th
- text embeddings powered by sentence-transformers
- polyglot code indexing powered by tree-sitter (Python built-in; 12 additional languages optional)
- optional graph persistence in a single SQLite file
- test suite using pytest

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

## Usage

Generate an embedding with the embed subcommand:

```powershell
th embed "A short sentence to embed"
```

Specify a model explicitly:

```powershell
th embed "A short sentence to embed" --model sentence-transformers/all-mpnet-base-v2
```

Optional cache override:

```powershell
$env:CINDEX_CACHE_DIR = "C:\models\cache"
th embed "A short sentence to embed"
th embed "A short sentence to embed" --cache-dir "C:\another\cache"
```

The command prints the embedding as a JSON array of floats.

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

When `sqlite-vector` can be loaded, `trailhead` also initializes vector search for
the `vertex_embeddings.embedding` column. If extension loading is unavailable on
your platform build, embeddings are still stored as Float32 BLOBs in SQLite.

Run the warm-model API server with a background indexer. The server watches the source tree, keeps the SQLite graph fresh, and reuses the loaded embedding model across index updates. The database defaults to `.cindex/db.sqlite` under the watched directory:

```powershell
th serve .
th serve . --model sentence-transformers/all-MiniLM-L6-v2
th serve . --sqlite-db ./.cindex/graph.db --model sentence-transformers/all-MiniLM-L6-v2
```

Then call the API from another terminal:

```powershell
curl http://127.0.0.1:8000/api/health
curl -X POST http://127.0.0.1:8000/api/embed -H "Content-Type: application/json" -d '{"text":"hello world"}'
```

Run a read-only SQL query against the SQLite database (defaults to `./.cindex/db.sqlite`):

```powershell
th query sql --sql "SELECT label, COUNT(*) AS n FROM vertices GROUP BY label ORDER BY label"
th query sql --sqlite-db ./.cindex/graph.db --sql "SELECT label, COUNT(*) AS n FROM vertices GROUP BY label ORDER BY label"
```

Run a semantic similarity query against stored vertex embeddings:

```powershell
th query similar "find sqlite vector initialization code"
th query similar "find sqlite vector initialization code" --sqlite-db ./.cindex/graph.db
```

The same semantic query is also available over HTTP once the server is running. The server owns the configured SQLite database path, so the browser or API client only sends the query payload:

```powershell
curl "http://127.0.0.1:8000/api/query/similar?text=find%20sqlite%20vector%20initialization%20code"
```

Limit the search to a specific vertex label and format the output as JSON:

```powershell
th query similar "graph persistence" --sqlite-db ./.cindex/graph.db --label function --k 5 --output json
```

Search graph vertices over HTTP:

```powershell
curl "http://127.0.0.1:8000/api/graph/vertices?name=persist&label=function"
```

Traverse a local subgraph from a known vertex id:

```powershell
curl "http://127.0.0.1:8000/api/graph/traverse?vertex_id=<vertex-id>&direction=both&depth=1"
```

Run the installed command directly:

```powershell
th embed "A short sentence to embed"
```

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

### Adding a custom language adapter

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

Edges: `defines` (module→class, module→function), `has_method` (class→function), `imports` (module→external).
