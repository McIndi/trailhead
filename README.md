# cindex

Command-line text embedding tool with:

- a single CLI command: cindex
- text embeddings powered by sentence-transformers
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
|       |       `-- embed.py
|       `-- services/
|           |-- config/
|           |   `-- cache.py
|           `-- embeddings/
|               `-- generator.py
`-- tests/
    `-- test_smoke.py
```
