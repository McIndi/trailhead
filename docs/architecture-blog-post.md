# Trailhead: One SQLite File That Knows Your Entire Codebase

*A technical deep-dive into how Trailhead indexes, stores, and queries code using vectors, graphs, and relational data — all in a single portable file.*

---

Most tools that help AI agents understand code work the same way: read a bunch of files, shove them into a context window, and hope the model figures the rest out. That approach has a fundamental problem. It's slow, expensive, and the quality of answers degrades as the codebase grows. What you actually want is something that has already *understood* the code and lets you ask precise questions about it.

That's the idea behind Trailhead. Build a complete, queryable map of your code once. Store it in one file. Query it directly instead of reading everything blind.

This post walks through how that actually works — the architecture, the data model, and what becomes possible once everything is in place. The best part: I'm writing this with Trailhead running against its own index. Every number you see here was queried live from `localhost:8000`.

---

## The Three-Layer Store

The central design decision in Trailhead is that vectors, graphs, and relational data all live in **one SQLite file**. No separate vector database, no graph server to run, no synchronization headaches between stores.

The schema has five tables:

```sql
-- Symbols extracted from source files
CREATE TABLE vertices (
  id TEXT PRIMARY KEY,
  label TEXT NOT NULL,          -- 'module', 'class', 'function', 'external'
  properties_json TEXT NOT NULL,

  -- Generated columns — no storage cost, always consistent
  name       TEXT    GENERATED ALWAYS AS (json_extract(properties_json, '$.name'))       VIRTUAL,
  path       TEXT    GENERATED ALWAYS AS (COALESCE(json_extract(properties_json, '$.path'), '')) VIRTUAL,
  line       INTEGER GENERATED ALWAYS AS (json_extract(properties_json, '$.line'))       VIRTUAL,
  complexity INTEGER GENERATED ALWAYS AS (json_extract(properties_json, '$.complexity')) VIRTUAL
);

-- Relationships between symbols
CREATE TABLE edges (
  id TEXT PRIMARY KEY,
  label TEXT NOT NULL,          -- 'defines', 'calls', 'imports', 'has_method'
  out_v_id TEXT REFERENCES vertices(id) ON DELETE CASCADE,
  in_v_id  TEXT REFERENCES vertices(id) ON DELETE CASCADE,
  properties_json TEXT NOT NULL
);

-- Semantic embeddings stored as raw float32 BLOBs
CREATE TABLE vertex_embeddings (
  vertex_id  TEXT PRIMARY KEY REFERENCES vertices(id) ON DELETE CASCADE,
  model_name TEXT NOT NULL,
  source_text TEXT NOT NULL,
  dimension  INTEGER NOT NULL,
  embedding  BLOB NOT NULL      -- packed IEEE 754 float32 array
);

-- Tracks what has been indexed (mtime-based sync)
CREATE TABLE indexed_files (
  path     TEXT PRIMARY KEY,
  mtime_ns INTEGER NOT NULL
);
```

The `properties_json` column stores everything flexible about a symbol — name, file path, source text, docstring, line number, complexity score — as a JSON blob. The generated columns (`name`, `path`, `line`, `complexity`) pull those values out into real SQL columns without duplicating storage. This gives you clean `WHERE complexity > 10` queries without any application-level JSON parsing.

### The index on itself

Right now, running against Trailhead's own source tree:

| Label | Count |
|-------|-------|
| `function` | 350 |
| `external` | 91 |
| `class` | 58 |
| `module` | 55 |
| **Total vertices** | **554** |

| Edge type | Count |
|-----------|-------|
| `imports` | 417 |
| `calls` | 300 |
| `defines` | 247 |
| `has_method` | 161 |
| **Total edges** | **1,125** |

---

## Parsing: Tree-Sitter All the Way Down

Trailhead understands code structurally, not textually. Every file is parsed with **[tree-sitter](https://tree-sitter.github.io/tree-sitter/)**, a battle-tested incremental parser that produces a concrete syntax tree for 13 languages.

Python support is built in. The other languages (JavaScript, TypeScript, Rust, Go, Java, C#, C, C++, Ruby, PHP, Bash, and HTML) are optional pip extras:

```bash
# Optional Typescript support
pip install "trailhead[typescript]"

# Install all optional languages
pip install "trailhead[all-languages]"
```

Each language has its own adapter that implements a single interface:

```python
class LanguageAdapter(ABC):
    @abstractmethod
    def parse(self, path: Path, graph: PropertyGraph) -> Vertex:
        ...
```

The adapter walks the tree-sitter CST and populates an in-memory `PropertyGraph`. The graph model follows the Apache TinkerPop property graph spec: vertices with labels and key-value properties, edges connecting vertices with their own labels and properties.

Every function vertex gets a **McCabe cyclomatic complexity score** computed during the parse pass. The algorithm starts at 1 and adds 1 for each branching node (`if`, `for`, `while`, `match`, etc.):

```python
def _complexity(root_node, branching_types: frozenset[str]) -> int:
    count = 1
    stack = [root_node]
    while stack:
        n = stack.pop()
        if n.type in branching_types:
            count += 1
        stack.extend(n.children)
    return count
```

There's also a second pass for most languages that resolves **call edges**: for every call site inside a function body, it looks up the callee by name (preferring same-file definitions) and adds a `calls` edge. The result is a navigable call graph, queryable in either direction.

### Shared adapter utilities

The `base.py` adapter module contains helpers shared across all language adapters. The call graph data confirms this: `_node_text` (extracts text from a tree-sitter node) has **55 inbound call edges** called from every single language adapter. It is the most-called function in the entire codebase.

```
_node_text         55 callers   base.py:11
_add_external      12 callers   base.py:134
_collect_calls_ts  11 callers   base.py:55
_complexity        12 callers   base.py:16
```

`_collect_calls_ts` is particularly interesting: it's a generic call-edge collector parameterized by node types and a callee name extractor. Instead of each language adapter reinventing call graph construction, they pass in the language-specific node type sets and a single `get_callee_name` function:

```python
_collect_calls_ts(
    tree.root_node, graph, module_v, source,
    func_node_types=frozenset({"function_declaration", "method_declaration"}),
    call_node_types=frozenset({"call_expression"}),
    get_callee_name=_java_callee_name,
)
```

---

## The Indexing Pipeline

The full flow from source file to queryable SQLite:

```
source files
     │
     ▼ tree-sitter parse (per-language adapter)
PropertyGraph (in-memory)
     │
     ▼ persist_graph()
SQLite: vertices + edges
     │
     ▼ persist_vertex_embeddings()  [optional]
SQLite: vertex_embeddings (float32 BLOBs)
     │
     ▼ persist_indexed_files()
SQLite: indexed_files (path + mtime_ns)
```

On subsequent runs, the `LiveIndexer` compares current file mtimes against `indexed_files`. Only changed or deleted files are reprocessed full rebuilds are rare after the first index.

### Live indexing: two threads, clean separation

`LiveIndexer` runs two background threads once `start()` is called:

- **watcher thread** — runs `watchfiles.watch()` and enqueues sets of changed paths. It does no heavy work; file-system events are never dropped.
- **worker thread** — drains the queue and calls `reindex_file()` per path. Embedding generation happens here, off the watcher thread.

```python
class LiveIndexer:
    def start(self) -> None:
        self._worker_thread = Thread(target=self._worker_loop, daemon=True)
        self._watch_thread  = Thread(target=self._watch_loop,  daemon=True)
        ...
```

`reindex_file()` does a per-file atomic replace: delete all existing vertices and edges for the path, re-parse, re-embed, and write. If the file was deleted, the cleanup alone runs.

The `synchronize()` method always runs on the main thread before `start()`, so there's no contention during the initial catch-up scan.

---

## Semantic Search via sqlite-vector

Embeddings are generated with [sentence-transformers](https://www.sbert.net/). The default model is `sentence-transformers/all-MiniLM-L6-v2` — small, fast, and locally cached. Embeddings are stored as packed IEEE 754 float32 BLOBs in the `vertex_embeddings` table.

Similarity search uses the **sqlite-vector** extension and cosine distance:

```sql
SELECT v.id, v.label, v.name, nn.distance
FROM vector_full_scan('vertex_embeddings', 'embedding', ?, 10) AS nn
JOIN vertex_embeddings ve ON ve.rowid = nn.rowid
JOIN vertices v ON v.id = ve.vertex_id
ORDER BY nn.distance
LIMIT 10
```

The `vector_full_scan` virtual table function is provided by the extension. The embedding column is initialized as `type=FLOAT32,distance=COSINE` before each query.

### Semantic search in action

Running a live query against the self-index — searching for "index codebase parse files":

| Rank | Name | File | Distance |
|------|------|------|----------|
| 1 | `parse_file` | `adapters/registry.py` | 0.374 |
| 2 | `parse` | `adapters/java.py` | 0.396 |
| 3 | `parse` | `adapters/c.py` | 0.400 |
| 4 | `parse` | `adapters/bash.py` | 0.408 |
| 5 | `parser` (module) | `indexing/parser.py` | 0.411 |

The model has never seen this codebase. It returns the right functions based on semantic meaning alone — no keyword matches required.

---

## The Query API

The server is a FastAPI app that keeps the embedding model loaded in memory across requests (the "warm model" pattern). Cold-loading a sentence-transformers model takes a few seconds; the server eliminates that latency for every query after the first.

### Endpoints

| Endpoint | What it does |
|----------|--------------|
| `POST /api/query/sql` | Run any SQL against the read-only index |
| `GET /api/query/similar` | Semantic nearest-neighbor search |
| `GET /api/query/templates` | List built-in query templates |
| `POST /api/query/templates/{name}/run` | Run a named template |
| `GET /api/graph/vertices` | Filter vertices by name/label/path |
| `GET /api/graph/traverse` | Traverse the call/import graph |
| `POST /api/embed` | Embed a single text (warm model) |
| `POST /api/embed/batch` | Embed up to 100 texts in one call |

The SQL endpoint opens a read-only connection (`mode=ro` URI flag). No string-prefix heuristics, no allowlists — the engine itself enforces the constraint.

### Built-in query templates

Fourteen templates ship out of the box, organized by category:

**quality** — `function_complexity`, `missing_docstrings`, `undocumented_public_api`, `todo_fixme_inventory`

**testing** — `symbols_not_represented_by_tests`, `test_coverage_ratio_by_file`, `largest_untested_symbols`

**architecture** — `duplicate_symbol_names`, `dependency_hotspots`, `external_dependency_pressure`

**calls** — `most_called_functions`, `call_graph_hubs`

**data_health** — `missing_source_for_functions`, `orphan_edges`

Each template is a named SQL query stored in `query_templates.py`. They run against the relational + graph layers — no embeddings required.

### Complexity analysis: live results

Running `function_complexity` against the self-index, the top results:

| Function | File | Complexity |
|----------|------|------------|
| `run` | `cli/commands/index.py:102` | **20** |
| `_visit` | `adapters/php.py:106` | 19 |
| `_collect_calls` | `adapters/python.py:189` | 19 |
| `_visit` | `adapters/javascript.py:97` | 18 |
| `_collect_calls_ts` | `adapters/base.py:55` | 14 |

`run` at complexity 20 is the CLI entry point for `th index` — it handles all the argument validation, mode selection (in-memory vs. SQLite, watch vs. one-shot), and model conflict detection. It's a candidate for refactoring into smaller, focused functions.

The `_visit` functions across PHP and JavaScript adapters score 18–19, reflecting the inherent complexity of dispatching over a full language's AST node types in a single recursive function.

### Dependency hotspots: live results

The `dependency_hotspots` template counts inbound import edges per file:

| File | Import edges | Unique targets |
|------|-------------|----------------|
| `tests/test_indexing.py` | 72 | 8 |
| `tests/test_server.py` | 27 | 7 |
| `server/app.py` | 25 | 15 |
| `adapters/__init__.py` | 16 | 16 |
| `services/indexing/live_indexer.py` | 14 | 8 |

The server app imports from 15 unique targets — it's the integration point for embeddings, graph queries, SQL execution, templates, and the live indexer.

---

## Use Cases

### For AI agents

An agent pointed at a Trailhead server can ask precise questions instead of reading files:

- *"Which functions call `generate_embedding`?"* — graph traversal, inbound `calls` edges
- *"Find code related to embedding cache invalidation"* — semantic similarity search
- *"What are the most complex functions I should look at first?"* — `function_complexity` template
- *"Which external libraries does this module depend on?"* — `external_dependency_pressure`

Every query returns structured data with source text, file paths, and line numbers. The agent gets exactly what it needs without reading a single file.

### For code review and architecture

Before reviewing a pull request, run `function_complexity` to identify high-risk functions. Run `dependency_hotspots` to find files that many things depend on — changes there have wide blast radii. Run `symbols_not_represented_by_tests` for a heuristic coverage gap report.

These aren't static analysis tools that need to be wired into CI. They're ad-hoc queries against a local index.

### For refactoring

The call graph is bidirectional. Given a function you want to refactor:

1. Traverse **outbound** `calls` edges to see what it depends on
2. Traverse **inbound** `calls` edges to see who will break if the signature changes
3. Check its complexity score to understand how tangled the internals are

The original README describes exactly this workflow: complexity 33 → identify callers → break apart → complexity 5. That's a real query, not a demo.

---

## The "One File" Property

Everything — vertices, edges, embeddings, indexed file state — lives in a single `.db` file. This has practical consequences:

- **Portable**: copy the file, carry the entire index. No server to restart.
- **Private**: nothing ever leaves your machine unless you choose to move the file.
- **Cheap to rebuild**: if something goes wrong, delete the file and re-run `th index`.
- **Standard tooling**: any SQLite client can open it. Browse the schema, run ad-hoc queries, inspect the data — no proprietary format.

The CLI workflow reflects this:

```bash
# Index a project (creates ./trailhead/graph.db by default)
th index /path/to/project --sqlite-db ./index.db --embed-model sentence-transformers/all-MiniLM-L6-v2

# Serve queries against the index (warm model, live file watch)
th serve --sqlite-db ./index.db --watch-directory /path/to/project

# Query from the CLI
th query similar "authentication middleware"
th query sql "SELECT name, complexity FROM vertices WHERE label='function' ORDER BY complexity DESC LIMIT 10"
```

---

## What's Next

Trailhead has been thoroughly tested on its own Python codebase. The 12 additional language adapters (JavaScript, TypeScript, Rust, Go, Java, C#, C, C++, Ruby, PHP, Bash, and HTML) are structurally complete but need real-world stress testing on large, production codebases.

There's also an open question about token efficiency: how much context does a Trailhead query actually save compared to feeding raw files to an AI agent? The hypothesis is that it's significant — a single `function_complexity` query returns actionable results that would otherwise require reading and summarizing hundreds of source files. But that hypothesis needs a repeatable benchmark.

If either of those sounds like your kind of problem (stress-testing the language adapters on a large Go or Rust project, or designing a token-savings benchmark) open an issue on [Github](https://github.com/McIndi/trailhead/issues).

The index is the easy part. What you do with it is the interesting question.
