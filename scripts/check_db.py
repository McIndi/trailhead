"""Quick DB integrity snapshot — run before/after edits to verify no duplicates."""
import sqlite3
import sys

db = r"C:\Users\cliff\projects\mcindi\cindex\.cindex\db.sqlite"
label_filter = sys.argv[1] if len(sys.argv) > 1 else None

with sqlite3.connect(db) as conn:
    conn.row_factory = sqlite3.Row

    print("=== Vertex counts by label ===")
    for row in conn.execute("SELECT label, COUNT(*) c FROM vertices GROUP BY label ORDER BY c DESC"):
        print(f"  {row['label']}: {row['c']}")

    print()
    print("=== Edge counts by label ===")
    for row in conn.execute("SELECT label, COUNT(*) c FROM edges GROUP BY label ORDER BY c DESC"):
        print(f"  {row['label']}: {row['c']}")

    print()
    n = conn.execute("SELECT COUNT(*) FROM vertex_embeddings").fetchone()[0]
    print(f"=== Embeddings: {n} ===")

    print()
    print("=== Orphan edge check (edges pointing to missing vertices) ===")
    orphan_out = conn.execute(
        "SELECT COUNT(*) FROM edges WHERE out_v_id NOT IN (SELECT id FROM vertices)"
    ).fetchone()[0]
    orphan_in = conn.execute(
        "SELECT COUNT(*) FROM edges WHERE in_v_id NOT IN (SELECT id FROM vertices)"
    ).fetchone()[0]
    print(f"  Orphan out_v_id: {orphan_out}")
    print(f"  Orphan in_v_id:  {orphan_in}")

    print()
    print("=== Duplicate name+label+path check ===")
    dups = conn.execute("""
        SELECT label,
               json_extract(properties_json, '$.name') AS name,
               json_extract(properties_json, '$.path') AS path,
               COUNT(*) c
        FROM vertices
        WHERE label IN ('module', 'class', 'function')
        GROUP BY label, name, path, json_extract(properties_json, '$.line')
        HAVING c > 1
    """).fetchall()
    if dups:
        for d in dups:
            print(f"  DUPLICATE: {d['label']} {d['name']} @ {d['path']}  ({d['c']}x)")
    else:
        print("  None — all clear")

    if label_filter:
        print()
        print(f"=== Symbols in *{label_filter}* ===")
        rows = conn.execute("""
            SELECT label,
                   json_extract(properties_json, '$.name') AS name,
                   json_extract(properties_json, '$.line') AS line
            FROM vertices
            WHERE label NOT IN ('external', 'module')
              AND json_extract(properties_json, '$.path') LIKE ?
            ORDER BY CAST(line AS INT)
        """, (f"%{label_filter}",)).fetchall()
        for r in rows:
            print(f"  {r['label']:<10} L{str(r['line']).rjust(3)}  {r['name']}")
