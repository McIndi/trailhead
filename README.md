# cindex

Command-line code indexing and text embedding tool with:

- a single CLI command: cindex
- text embeddings powered by sentence-transformers
- code indexing powered by tree-sitter
- optional graph persistence in a single SQLite file
- a smoke test suite using pytest
- modern packaging via pyproject.toml

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

## Usage

Generate an embedding with the embed subcommand:

```powershell
cindex embed "A short sentence to embed"
```

Specify a model explicitly:

```powershell
cindex embed "A short sentence to embed" --model sentence-transformers/all-mpnet-base-v2
```

Optional cache override:

```powershell
$env:CINDEX_CACHE_DIR = "C:\models\cache"
cindex embed "A short sentence to embed"
cindex embed "A short sentence to embed" --cache-dir "C:\another\cache"
```

The command prints the embedding as a JSON array of floats.

Index a directory of Python source files:

```powershell
cindex index .
```

Persist the graph into a single SQLite file:

```powershell
cindex index . --sqlite-db ./.cindex/graph.db
```

Persist graph + embeddings into the same SQLite file:

```powershell
cindex index . \
    --sqlite-db ./.cindex/graph.db \
    --embed-model sentence-transformers/all-MiniLM-L6-v2
```

Optional embedding cache override:

```powershell
cindex index . --sqlite-db ./.cindex/graph.db --embed-model sentence-transformers/all-MiniLM-L6-v2 --embed-cache-dir C:\models\cache
```

Append to an existing SQLite graph instead of replacing it:

```powershell
cindex index . --sqlite-db ./.cindex/graph.db --append
```

When `sqlite-vector` can be loaded, `cindex` also initializes vector search for
the `vertex_embeddings.embedding` column. If extension loading is unavailable on
your platform build, embeddings are still stored as Float32 BLOBs in SQLite.

Run a read-only SQL query against the SQLite database:

```powershell
cindex query sql --sqlite-db ./.cindex/graph.db --sql "SELECT label, COUNT(*) AS n FROM vertices GROUP BY label ORDER BY label"
```

Run a semantic similarity query against stored vertex embeddings:

```powershell
cindex query similar "find sqlite vector initialization code" --sqlite-db ./.cindex/graph.db
```

Limit the search to a specific vertex label and format the output as JSON:

```powershell
cindex query similar "graph persistence" --sqlite-db ./.cindex/graph.db --label function --k 5 --output json
```

You can also run the module directly:

```powershell
python -m cindex embed "A short sentence to embed"
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
|       |       `-- query.py
|       `-- services/
|           |-- config/
|           |   `-- cache.py
|           |-- indexing/
|           |   |-- __init__.py
|           |   |-- graph.py
|           |   |-- parser.py
|           |   |-- query.py
|           |   |-- sqlite_store.py
|           |   `-- walker.py
|           `-- embeddings/
|               `-- generator.py
`-- tests/
    |-- test_indexing.py
    |-- test_query.py
    `-- test_smoke.py
```
