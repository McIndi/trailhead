# cindex

Minimal Python project scaffold with:

- a single CLI command: `cindex`
- a smoke test suite using `pytest`
- modern packaging via `pyproject.toml`

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

Run the installed command:

```powershell
cindex
```

You can also run the module directly:

```powershell
python -m cindex
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
|       `-- cli.py
`-- tests/
    `-- test_smoke.py
```
