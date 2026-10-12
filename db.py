from __future__ import annotations

import os
import re
import uuid
from pathlib import Path

import pandas as pd

LAPTOP_COLUMNS = [
    "name", "price", "price_numeric", "processor", "memory", "ram_gb", "storage", "storage_gb",
    "graphics", "cpu_cores", "gpu_cores", "url", "processor_normalized", "graphics_normalized",
    "cpu_passmark_mark", "cpu_passmark_name", "cpu_mapping_status", "gpu_passmark_g3d_mark",
    "gpu_passmark_name", "gpu_mapping_status", "benchmark_date",
]
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


# ----------------------------------------------------------------------------
# connection
# ----------------------------------------------------------------------------
def read_config_from_env():
    """MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE -> dict or None."""
    host, user, database = (os.environ.get("MYSQL_HOST"), os.environ.get("MYSQL_USER"),
                            os.environ.get("MYSQL_DATABASE"))
    if not (host and user and database):
        return None
    return dict(host=host, port=int(os.environ.get("MYSQL_PORT", "3306")), user=user,
                password=os.environ.get("MYSQL_PASSWORD", ""), database=database)


def connect(cfg: dict):
    try:
        import pymysql
        import pymysql.cursors
    except ImportError as exc:
        raise RuntimeError("Paket PyMySQL belum terpasang. Jalankan: pip install pymysql") from exc
    return pymysql.connect(
        host=cfg["host"], port=int(cfg.get("port", 3306)), user=cfg["user"],
        password=cfg.get("password", ""), database=cfg["database"], charset="utf8mb4",
        connect_timeout=5, cursorclass=pymysql.cursors.DictCursor, autocommit=False)


def _fetchall(conn, sql, params=()):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return list(cur.fetchall())


def _fetchone(conn, sql, params=()):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()


def _execute(conn, sql, params=()):
    with conn.cursor() as cur:
        cur.execute(sql, params)


# ----------------------------------------------------------------------------
# schema + catalog
# ----------------------------------------------------------------------------
def schema_statements(schema_text: str | None = None) -> list:
    text = schema_text if schema_text is not None else SCHEMA_PATH.read_text(encoding="utf-8")
    text = "\n".join(l for l in text.splitlines() if not l.strip().startswith("--"))
    return [s.strip() for s in text.split(";") if s.strip()]


def create_schema(conn, schema_text: str | None = None) -> int:
    stmts = schema_statements(schema_text)
    for s in stmts:
        _execute(conn, s)
    conn.commit()
    return len(stmts)


def load_catalog(conn) -> pd.DataFrame:
    rows = _fetchall(conn, "SELECT * FROM laptops ORDER BY id")
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.rename(columns={"id": "laptop_id"})
    if "benchmark_date" in df.columns:
        df["benchmark_date"] = df["benchmark_date"].astype(str)
    return df


def catalog_stats(conn) -> dict:
    row = _fetchone(conn, """
        SELECT COUNT(*) AS n,
               SUM(CASE WHEN cpu_passmark_mark IS NULL THEN 1 ELSE 0 END) AS cpu_null,
               SUM(CASE WHEN gpu_passmark_g3d_mark IS NULL THEN 1 ELSE 0 END) AS gpu_null,
               SUM(CASE WHEN cpu_passmark_mark = 0 THEN 1 ELSE 0 END) AS cpu_zero,
               SUM(CASE WHEN gpu_passmark_g3d_mark = 0 THEN 1 ELSE 0 END) AS gpu_zero
        FROM laptops""")
    return {k: int(v or 0) for k, v in row.items()}


def _same(a, b) -> bool:
    """Compare a database value with a CSV value (None-safe, float tolerant, dates as text)."""
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-9
    return str(a).strip() == str(b).strip()


def upsert_laptops(conn, records: list) -> dict:
    """Insert new laptops, update changed ones (matched by url), skip unchanged ones.

    Returns counts: inserted, updated (rows whose values really changed), unchanged,
    and changed_columns (how many rows changed per column), so a data correction in the
    CSV is visible after the import.
    """
    cols = LAPTOP_COLUMNS
    ins_sql = ("INSERT INTO laptops (" + ", ".join(f"`{c}`" for c in cols) + ") VALUES ("
               + ", ".join(["%s"] * len(cols)) + ")")
    upd_cols = [c for c in cols if c != "url"]
    upd_sql = ("UPDATE laptops SET " + ", ".join(f"`{c}` = %s" for c in upd_cols) + " WHERE id = %s")
    inserted = updated = unchanged = 0
    changed_columns: dict = {}
    try:
        for rec in records:
            existing = _fetchone(conn, "SELECT * FROM laptops WHERE url = %s", (rec["url"],))
            if existing:
                diff = [c for c in upd_cols if not _same(existing.get(c), rec[c])]
                if diff:
                    _execute(conn, upd_sql, [rec[c] for c in upd_cols] + [existing["id"]])
                    updated += 1
                    for c in diff:
                        changed_columns[c] = changed_columns.get(c, 0) + 1
                else:
                    unchanged += 1
            else:
                _execute(conn, ins_sql, [rec[c] for c in cols])
                inserted += 1
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return dict(inserted=inserted, updated=updated, unchanged=unchanged, changed_columns=changed_columns)


# ----------------------------------------------------------------------------
# sessions + feedback
# ----------------------------------------------------------------------------
def insert_session(conn, record: dict, items: list) -> str:
    """Log one query anonymously. `items`: dicts with rank, laptop_id, distance."""
    session_id = str(uuid.uuid4())
    try:
        _execute(conn, """
            INSERT INTO sessions (id, profile, budget_idr, min_ram_gb, min_storage_gb, cpu_level,
                gpu_level, require_dedicated, lambda_surplus, catalog_source, n_after_filter,
                n_rankable, n_recommendations)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                 (session_id, record["profile"], record["budget_idr"], record["min_ram_gb"],
                  record["min_storage_gb"], record["cpu_level"], record["gpu_level"],
                  1 if record["require_dedicated"] else 0, record["lambda_surplus"],
                  record["catalog_source"], record["n_after_filter"], record["n_rankable"],
                  record["n_recommendations"]))
        for it in items:
            _execute(conn, "INSERT INTO session_items (session_id, rank_position, laptop_id, knn_distance) "
                           "VALUES (%s, %s, %s, %s)",
                     (session_id, int(it["rank"]), int(it["laptop_id"]), float(it["distance"])))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return session_id


def upsert_rating(conn, session_id: str, laptop_id: int, rating: int) -> None:
    if not isinstance(rating, int) or not (1 <= rating <= 5):
        raise ValueError("Rating harus bilangan bulat 1 sampai 5.")
    if not _fetchone(conn, "SELECT 1 AS ok FROM session_items WHERE session_id = %s AND laptop_id = %s",
                     (session_id, laptop_id)):
        raise ValueError("Item tidak ditemukan pada sesi ini.")
    try:
        if _fetchone(conn, "SELECT 1 AS ok FROM feedback WHERE session_id = %s AND laptop_id = %s",
                     (session_id, laptop_id)):
            _execute(conn, "UPDATE feedback SET rating = %s WHERE session_id = %s AND laptop_id = %s",
                     (rating, session_id, laptop_id))
        else:
            _execute(conn, "INSERT INTO feedback (session_id, laptop_id, rating) VALUES (%s, %s, %s)",
                     (session_id, laptop_id, rating))
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def save_comment(conn, session_id: str, text: str) -> None:
    text = (text or "").strip()[:2000]
    try:
        _execute(conn, "UPDATE sessions SET comment = %s WHERE id = %s", (text or None, session_id))
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def feedback_overview(conn) -> dict:
    one = lambda sql: int((_fetchone(conn, sql) or {}).get("n") or 0)  # noqa: E731
    totals = dict(
        sessions=one("SELECT COUNT(*) AS n FROM sessions"),
        sessions_with_rating=one("SELECT COUNT(DISTINCT session_id) AS n FROM feedback"),
        ratings=one("SELECT COUNT(*) AS n FROM feedback"))
    by_profile = pd.DataFrame(_fetchall(conn, """
        SELECT s.profile AS profil, COUNT(*) AS jumlah_rating, AVG(f.rating) AS rata2_rating,
               AVG(CASE WHEN f.rating >= 4 THEN 1 ELSE 0 END) AS proporsi_rating_4_atau_lebih
        FROM feedback f JOIN sessions s ON s.id = f.session_id
        GROUP BY s.profile ORDER BY s.profile"""))
    by_rank = pd.DataFrame(_fetchall(conn, """
        SELECT si.rank_position AS peringkat, COUNT(*) AS jumlah_rating, AVG(f.rating) AS rata2_rating
        FROM feedback f JOIN session_items si
          ON si.session_id = f.session_id AND si.laptop_id = f.laptop_id
        GROUP BY si.rank_position ORDER BY si.rank_position"""))
    dist = pd.DataFrame(_fetchall(conn, """
        SELECT rating, COUNT(*) AS jumlah FROM feedback GROUP BY rating ORDER BY rating"""))
    return dict(totals=totals, by_profile=by_profile, by_rank=by_rank, distribution=dist)


def export_feedback(conn) -> pd.DataFrame:
    return pd.DataFrame(_fetchall(conn, """
        SELECT f.session_id, s.created_at, s.profile, s.budget_idr, s.min_ram_gb, s.min_storage_gb,
               s.cpu_level, s.gpu_level, s.require_dedicated, s.lambda_surplus,
               si.rank_position, si.knn_distance, l.name AS laptop, l.url, f.rating
        FROM feedback f
        JOIN session_items si ON si.session_id = f.session_id AND si.laptop_id = f.laptop_id
        JOIN sessions s ON s.id = f.session_id
        JOIN laptops l ON l.id = f.laptop_id
        ORDER BY s.created_at, si.rank_position"""))
