"""Reusable starter SQL query templates for common trailhead workflows."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QueryTemplate:
    name: str
    category: str
    title: str
    description: str
    sql: str


# NOTE: test representation is heuristic-based.
# A symbol is considered "represented by tests" if either:
# 1) a test file name appears to target the module (test_<module>.py or <module>_test.py), or
# 2) a test function source string references the symbol name.
_QUERY_TEMPLATES: tuple[QueryTemplate, ...] = (
    QueryTemplate(
        name="function_complexity",
        category="quality",
        title="Function Complexity",
        description="List functions and methods with their McCabe (cyclomatic) complexity.",
        sql="""
WITH symbols AS (
  SELECT
    v.id AS vertex_id,
    CASE
      WHEN EXISTS (
        SELECT 1 FROM edges e
        WHERE e.label = 'has_method' AND e.in_v_id = v.id
      ) THEN 'method'
      ELSE 'function'
    END AS symbol_kind,
    json_extract(v.properties_json, '$.name') AS name,
    json_extract(v.properties_json, '$.path') AS path,
    json_extract(v.properties_json, '$.line') AS line,
    json_extract(v.properties_json, '$.complexity') AS complexity,
    json_extract(v.properties_json, '$.source') AS source
  FROM vertices v
  WHERE v.label = 'function'
)
SELECT symbol_kind, path, line, name, complexity, source, vertex_id
FROM symbols
ORDER BY complexity DESC, path, line, name;
""".strip(),
    ),
    QueryTemplate(
        name="missing_docstrings",
        category="quality",
        title="Missing Docstrings",
        description="List modules, functions, and methods missing docstrings.",
        sql="""
WITH symbols AS (
  SELECT
    v.id AS vertex_id,
    CASE
      WHEN v.label = 'module' THEN 'module'
      WHEN EXISTS (
        SELECT 1 FROM edges e
        WHERE e.label = 'has_method' AND e.in_v_id = v.id
      ) THEN 'method'
      ELSE 'function'
    END AS symbol_kind,
    json_extract(v.properties_json, '$.name') AS name,
    json_extract(v.properties_json, '$.path') AS path,
    json_extract(v.properties_json, '$.line') AS line,
    json_extract(v.properties_json, '$.docstring') AS docstring,
    json_extract(v.properties_json, '$.source') AS source
  FROM vertices v
  WHERE v.label IN ('module', 'function')
)
SELECT symbol_kind, path, line, name, docstring, source, vertex_id
FROM symbols
WHERE docstring IS NULL OR TRIM(docstring) = ''
ORDER BY symbol_kind, path, line, name;
""".strip(),
    ),
    QueryTemplate(
        name="undocumented_public_api",
        category="quality",
        title="Undocumented Public API",
        description="Public modules/functions/methods without docstrings (name not prefixed by underscore).",
        sql="""
WITH symbols AS (
  SELECT
    v.id AS vertex_id,
    CASE
      WHEN v.label = 'module' THEN 'module'
      WHEN EXISTS (
        SELECT 1 FROM edges e
        WHERE e.label = 'has_method' AND e.in_v_id = v.id
      ) THEN 'method'
      ELSE 'function'
    END AS symbol_kind,
    json_extract(v.properties_json, '$.name') AS name,
    json_extract(v.properties_json, '$.path') AS path,
    json_extract(v.properties_json, '$.line') AS line,
    json_extract(v.properties_json, '$.docstring') AS docstring,
    json_extract(v.properties_json, '$.source') AS source
  FROM vertices v
  WHERE v.label IN ('module', 'function')
)
SELECT symbol_kind, path, line, name, docstring, source, vertex_id
FROM symbols
WHERE name IS NOT NULL
  AND name NOT LIKE '\\_%' ESCAPE '\\'
  AND (docstring IS NULL OR TRIM(docstring) = '')
ORDER BY symbol_kind, path, line, name;
""".strip(),
    ),
    QueryTemplate(
        name="symbols_not_represented_by_tests",
        category="testing",
        title="Symbols Not Represented by Tests",
        description=(
            "Heuristic report for modules/functions/methods not represented by tests. "
            "Matches by expected test file path or test function source containing symbol name."
        ),
        sql="""
WITH symbols AS (
  SELECT
    v.id AS vertex_id,
    CASE
      WHEN v.label = 'module' THEN 'module'
      WHEN EXISTS (
        SELECT 1 FROM edges e
        WHERE e.label = 'has_method' AND e.in_v_id = v.id
      ) THEN 'method'
      ELSE 'function'
    END AS symbol_kind,
    json_extract(v.properties_json, '$.name') AS name,
    json_extract(v.properties_json, '$.path') AS path,
    json_extract(v.properties_json, '$.line') AS line,
    json_extract(v.properties_json, '$.docstring') AS docstring,
    json_extract(v.properties_json, '$.source') AS source,
    lower(json_extract(v.properties_json, '$.name')) AS lower_name
  FROM vertices v
  WHERE v.label IN ('module', 'function')
    AND json_extract(v.properties_json, '$.path') NOT LIKE '%/tests/%'
    AND json_extract(v.properties_json, '$.path') NOT LIKE '%\\tests\\%'
),
tests AS (
  SELECT
    v.id AS vertex_id,
    v.label,
    json_extract(v.properties_json, '$.name') AS name,
    replace(json_extract(v.properties_json, '$.path'), '\\', '/') AS path,
    lower(COALESCE(json_extract(v.properties_json, '$.source'), '')) AS source
  FROM vertices v
  WHERE json_extract(v.properties_json, '$.path') LIKE '%/tests/%'
     OR json_extract(v.properties_json, '$.path') LIKE '%\\tests\\%'
),
matched AS (
  SELECT DISTINCT s.vertex_id
  FROM symbols s
  JOIN tests t ON (
    (
      s.symbol_kind = 'module'
      AND s.lower_name IS NOT NULL
      AND s.lower_name != ''
      AND (
        t.path LIKE '%/test_' || s.lower_name || '.py'
        OR t.path LIKE '%/' || s.lower_name || '_test.py'
      )
    )
    OR (
      t.label = 'function'
      AND s.lower_name IS NOT NULL
      AND s.lower_name != ''
      AND t.source LIKE '%' || s.lower_name || '%'
    )
  )
)
SELECT s.symbol_kind, s.path, s.line, s.name, s.docstring, s.source, s.vertex_id
FROM symbols s
LEFT JOIN matched m ON m.vertex_id = s.vertex_id
WHERE m.vertex_id IS NULL
ORDER BY s.symbol_kind, s.path, s.line, s.name;
""".strip(),
    ),
    QueryTemplate(
        name="test_coverage_ratio_by_file",
        category="testing",
        title="Test Coverage Ratio by File",
        description="Heuristic representation ratio by source file based on naming and symbol mentions in tests.",
        sql="""
WITH symbols AS (
  SELECT
    v.id AS vertex_id,
    json_extract(v.properties_json, '$.path') AS path,
    lower(json_extract(v.properties_json, '$.name')) AS lower_name
  FROM vertices v
  WHERE v.label IN ('module', 'function')
    AND json_extract(v.properties_json, '$.path') NOT LIKE '%/tests/%'
    AND json_extract(v.properties_json, '$.path') NOT LIKE '%\\tests\\%'
),
tests AS (
  SELECT
    v.label,
    replace(json_extract(v.properties_json, '$.path'), '\\', '/') AS path,
    lower(COALESCE(json_extract(v.properties_json, '$.source'), '')) AS source
  FROM vertices v
  WHERE json_extract(v.properties_json, '$.path') LIKE '%/tests/%'
     OR json_extract(v.properties_json, '$.path') LIKE '%\\tests\\%'
),
matched AS (
  SELECT DISTINCT s.vertex_id
  FROM symbols s
  JOIN tests t ON (
    (
      t.path LIKE '%/test_' || s.lower_name || '.py'
      OR t.path LIKE '%/' || s.lower_name || '_test.py'
    )
    OR (
      t.label = 'function'
      AND s.lower_name IS NOT NULL
      AND s.lower_name != ''
      AND t.source LIKE '%' || s.lower_name || '%'
    )
  )
),
by_file AS (
  SELECT
    s.path,
    COUNT(*) AS total_symbols,
    SUM(CASE WHEN m.vertex_id IS NULL THEN 0 ELSE 1 END) AS represented_symbols
  FROM symbols s
  LEFT JOIN matched m ON m.vertex_id = s.vertex_id
  GROUP BY s.path
)
SELECT
  path,
  total_symbols,
  represented_symbols,
  (total_symbols - represented_symbols) AS missing_symbols,
  ROUND((represented_symbols * 100.0) / NULLIF(total_symbols, 0), 1) AS represented_pct
FROM by_file
ORDER BY represented_pct ASC, missing_symbols DESC, path;
""".strip(),
    ),
    QueryTemplate(
        name="largest_untested_symbols",
        category="testing",
        title="Largest Untested Symbols",
        description="Heuristic list of unrepresented symbols ranked by source length.",
        sql="""
WITH symbols AS (
  SELECT
    v.id AS vertex_id,
    CASE
      WHEN v.label = 'module' THEN 'module'
      WHEN EXISTS (
        SELECT 1 FROM edges e
        WHERE e.label = 'has_method' AND e.in_v_id = v.id
      ) THEN 'method'
      ELSE 'function'
    END AS symbol_kind,
    json_extract(v.properties_json, '$.name') AS name,
    json_extract(v.properties_json, '$.path') AS path,
    json_extract(v.properties_json, '$.line') AS line,
    json_extract(v.properties_json, '$.source') AS source,
    lower(json_extract(v.properties_json, '$.name')) AS lower_name
  FROM vertices v
  WHERE v.label IN ('module', 'function')
    AND json_extract(v.properties_json, '$.path') NOT LIKE '%/tests/%'
    AND json_extract(v.properties_json, '$.path') NOT LIKE '%\\tests\\%'
),
tests AS (
  SELECT
    v.label,
    replace(json_extract(v.properties_json, '$.path'), '\\', '/') AS path,
    lower(COALESCE(json_extract(v.properties_json, '$.source'), '')) AS source
  FROM vertices v
  WHERE json_extract(v.properties_json, '$.path') LIKE '%/tests/%'
     OR json_extract(v.properties_json, '$.path') LIKE '%\\tests\\%'
),
matched AS (
  SELECT DISTINCT s.vertex_id
  FROM symbols s
  JOIN tests t ON (
    (
      t.path LIKE '%/test_' || s.lower_name || '.py'
      OR t.path LIKE '%/' || s.lower_name || '_test.py'
    )
    OR (
      t.label = 'function'
      AND s.lower_name IS NOT NULL
      AND s.lower_name != ''
      AND t.source LIKE '%' || s.lower_name || '%'
    )
  )
)
SELECT
  s.symbol_kind,
  s.path,
  s.line,
  s.name,
  LENGTH(COALESCE(s.source, '')) AS source_chars,
  s.source,
  s.vertex_id
FROM symbols s
LEFT JOIN matched m ON m.vertex_id = s.vertex_id
WHERE m.vertex_id IS NULL
ORDER BY source_chars DESC, s.symbol_kind, s.path, s.line, s.name;
""".strip(),
    ),
    QueryTemplate(
        name="duplicate_symbol_names",
        category="architecture",
        title="Duplicate Symbol Names",
        description="Names reused across multiple files and/or symbol kinds.",
        sql="""
WITH symbols AS (
  SELECT
    CASE
      WHEN v.label = 'module' THEN 'module'
      WHEN EXISTS (
        SELECT 1 FROM edges e
        WHERE e.label = 'has_method' AND e.in_v_id = v.id
      ) THEN 'method'
      ELSE 'function'
    END AS symbol_kind,
    json_extract(v.properties_json, '$.name') AS name,
    json_extract(v.properties_json, '$.path') AS path,
    json_extract(v.properties_json, '$.line') AS line,
    v.id AS vertex_id
  FROM vertices v
  WHERE v.label IN ('module', 'function')
),
dupe_names AS (
  SELECT name
  FROM symbols
  WHERE name IS NOT NULL AND TRIM(name) != ''
  GROUP BY name
  HAVING COUNT(*) > 1
)
SELECT s.name, s.symbol_kind, s.path, s.line, s.vertex_id
FROM symbols s
JOIN dupe_names d ON d.name = s.name
ORDER BY s.name, s.symbol_kind, s.path, s.line;
""".strip(),
    ),
    QueryTemplate(
        name="dependency_hotspots",
        category="architecture",
        title="Dependency Hotspots",
        description="Files with the largest number of import edges.",
        sql="""
WITH imports AS (
  SELECT
    json_extract(out_v.properties_json, '$.path') AS source_path,
    out_v.id AS source_vertex_id,
    in_v.id AS target_vertex_id,
    in_v.label AS target_label,
    json_extract(in_v.properties_json, '$.name') AS target_name
  FROM edges e
  JOIN vertices out_v ON out_v.id = e.out_v_id
  JOIN vertices in_v ON in_v.id = e.in_v_id
  WHERE e.label = 'imports'
),
per_file AS (
  SELECT
    source_path,
    COUNT(*) AS import_edges,
    COUNT(DISTINCT target_vertex_id) AS unique_import_targets,
    SUM(CASE WHEN target_label = 'external' THEN 1 ELSE 0 END) AS external_import_edges
  FROM imports
  GROUP BY source_path
)
SELECT source_path, import_edges, unique_import_targets, external_import_edges
FROM per_file
ORDER BY import_edges DESC, unique_import_targets DESC, source_path;
""".strip(),
    ),
    QueryTemplate(
        name="external_dependency_pressure",
        category="architecture",
        title="External Dependency Pressure",
        description="Most-referenced external symbols via import relationships.",
        sql="""
SELECT
  json_extract(ext.properties_json, '$.name') AS external_name,
  COUNT(*) AS import_edges,
  COUNT(DISTINCT json_extract(src.properties_json, '$.path')) AS importing_files
FROM edges e
JOIN vertices src ON src.id = e.out_v_id
JOIN vertices ext ON ext.id = e.in_v_id
WHERE e.label = 'imports'
  AND ext.label = 'external'
GROUP BY external_name
ORDER BY import_edges DESC, importing_files DESC, external_name;
""".strip(),
    ),
    QueryTemplate(
        name="missing_source_for_functions",
        category="data_health",
        title="Functions Missing Source",
        description="Function/method vertices that have empty or missing source text in index metadata.",
        sql="""
WITH symbols AS (
  SELECT
    v.id AS vertex_id,
    CASE
      WHEN EXISTS (
        SELECT 1 FROM edges e
        WHERE e.label = 'has_method' AND e.in_v_id = v.id
      ) THEN 'method'
      ELSE 'function'
    END AS symbol_kind,
    json_extract(v.properties_json, '$.name') AS name,
    json_extract(v.properties_json, '$.path') AS path,
    json_extract(v.properties_json, '$.line') AS line,
    json_extract(v.properties_json, '$.docstring') AS docstring,
    json_extract(v.properties_json, '$.source') AS source
  FROM vertices v
  WHERE v.label = 'function'
)
SELECT symbol_kind, path, line, name, docstring, source, vertex_id
FROM symbols
WHERE source IS NULL OR TRIM(source) = ''
ORDER BY path, line, name;
""".strip(),
    ),
    QueryTemplate(
        name="orphan_edges",
        category="data_health",
        title="Orphan Edges",
        description="Edges whose in/out vertices are missing.",
        sql="""
SELECT
  e.id AS edge_id,
  e.label,
  e.out_v_id,
  e.in_v_id,
  CASE WHEN out_v.id IS NULL THEN 1 ELSE 0 END AS missing_out_vertex,
  CASE WHEN in_v.id IS NULL THEN 1 ELSE 0 END AS missing_in_vertex
FROM edges e
LEFT JOIN vertices out_v ON out_v.id = e.out_v_id
LEFT JOIN vertices in_v ON in_v.id = e.in_v_id
WHERE out_v.id IS NULL OR in_v.id IS NULL
ORDER BY e.label, e.id;
""".strip(),
    ),
    QueryTemplate(
        name="most_called_functions",
        category="calls",
        title="Most Called Functions",
        description="Functions sorted by how many distinct callers call them (inbound call degree).",
        sql="""
SELECT
  json_extract(v.properties_json, '$.name') AS name,
  json_extract(v.properties_json, '$.path') AS path,
  json_extract(v.properties_json, '$.line') AS line,
  COUNT(DISTINCT e.out_v_id) AS caller_count,
  v.id AS vertex_id
FROM vertices v
JOIN edges e ON e.in_v_id = v.id AND e.label = 'calls'
WHERE v.label = 'function'
GROUP BY v.id
ORDER BY caller_count DESC, name;
""".strip(),
    ),
    QueryTemplate(
        name="call_graph_hubs",
        category="calls",
        title="Call Graph Hubs",
        description="Functions with high combined inbound + outbound call degree — likely orchestrators or utilities.",
        sql="""
WITH outbound AS (
  SELECT out_v_id AS v_id, COUNT(DISTINCT in_v_id) AS calls_out
  FROM edges WHERE label = 'calls'
  GROUP BY out_v_id
),
inbound AS (
  SELECT in_v_id AS v_id, COUNT(DISTINCT out_v_id) AS calls_in
  FROM edges WHERE label = 'calls'
  GROUP BY in_v_id
)
SELECT
  json_extract(v.properties_json, '$.name') AS name,
  json_extract(v.properties_json, '$.path') AS path,
  json_extract(v.properties_json, '$.line') AS line,
  COALESCE(o.calls_out, 0) AS calls_out,
  COALESCE(i.calls_in, 0) AS calls_in,
  COALESCE(o.calls_out, 0) + COALESCE(i.calls_in, 0) AS degree,
  v.id AS vertex_id
FROM vertices v
LEFT JOIN outbound o ON o.v_id = v.id
LEFT JOIN inbound i ON i.v_id = v.id
WHERE v.label = 'function'
  AND (COALESCE(o.calls_out, 0) + COALESCE(i.calls_in, 0)) > 0
ORDER BY degree DESC, name;
""".strip(),
    ),
    QueryTemplate(
        name="todo_fixme_inventory",
        category="quality",
        title="TODO/FIXME Inventory",
        description="Symbols whose indexed source contains TODO/FIXME/HACK markers.",
        sql="""
WITH symbols AS (
  SELECT
    v.id AS vertex_id,
    CASE
      WHEN v.label = 'module' THEN 'module'
      WHEN EXISTS (
        SELECT 1 FROM edges e
        WHERE e.label = 'has_method' AND e.in_v_id = v.id
      ) THEN 'method'
      ELSE 'function'
    END AS symbol_kind,
    json_extract(v.properties_json, '$.name') AS name,
    json_extract(v.properties_json, '$.path') AS path,
    json_extract(v.properties_json, '$.line') AS line,
    json_extract(v.properties_json, '$.docstring') AS docstring,
    json_extract(v.properties_json, '$.source') AS source,
    lower(COALESCE(json_extract(v.properties_json, '$.source'), '')) AS lower_source
  FROM vertices v
  WHERE v.label IN ('module', 'function')
)
SELECT symbol_kind, path, line, name, docstring, source, vertex_id
FROM symbols
WHERE lower_source LIKE '%todo%'
   OR lower_source LIKE '%fixme%'
   OR lower_source LIKE '%hack%'
ORDER BY symbol_kind, path, line, name;
""".strip(),
    ),
)


def list_query_templates() -> list[dict[str, str]]:
    """Return starter query metadata for UI/API listing."""
    return [
        {
            "name": template.name,
          "category": template.category,
            "title": template.title,
            "description": template.description,
        }
        for template in _QUERY_TEMPLATES
    ]


def get_query_template(name: str) -> QueryTemplate:
    """Return a starter query template by *name*."""
    for template in _QUERY_TEMPLATES:
        if template.name == name:
            return template
    raise ValueError(f"Unknown query template: {name}")
