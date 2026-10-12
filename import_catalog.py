from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import db

ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "dataset" / "laptops_benchmark_enriched_resolved.csv"
INT_COLUMNS = {"price_numeric", "ram_gb", "storage_gb", "cpu_cores", "gpu_cores"}
FLOAT_COLUMNS = {"cpu_passmark_mark", "gpu_passmark_g3d_mark"}
MAX_LEN = {"name": 255, "url": 255, "processor": 255, "processor_normalized": 255,
           "graphics": 255, "graphics_normalized": 255, "memory": 128, "storage": 128,
           "price": 32, "cpu_passmark_name": 128, "gpu_passmark_name": 128,
           "cpu_mapping_status": 32, "gpu_mapping_status": 32}


def clean(col: str, v):
    if v is None or (isinstance(v, float) and np.isnan(v)) or (not isinstance(v, str) and pd.isna(v)):
        return None
    if col in INT_COLUMNS:
        return int(v)
    if col in FLOAT_COLUMNS:
        return float(v)
    s = str(v).strip()
    return s if s != "" else None


def read_csv_records(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV tidak ditemukan: {csv_path}")
    df = pd.read_csv(csv_path)
    missing = [c for c in db.LAPTOP_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom CSV tidak ada: {missing}")
    if df["url"].isna().any() or df["url"].duplicated().any():
        raise ValueError("Kolom url harus terisi dan unik (dipakai sebagai kunci).")
    records = []
    for _, row in df.iterrows():
        rec = {c: clean(c, row[c]) for c in db.LAPTOP_COLUMNS}
        for c, limit in MAX_LEN.items():
            if rec[c] is not None and len(rec[c]) > limit:
                raise ValueError(f"Nilai kolom {c} melebihi {limit} karakter: {rec[c][:40]}...")
        for c in ("name", "url", "price_numeric", "ram_gb", "storage_gb"):
            if rec[c] is None:
                raise ValueError(f"Kolom wajib {c} kosong pada: {rec['url']}")
        records.append(rec)
    expected = dict(
        n=len(df), cpu_null=int(df["cpu_passmark_mark"].isna().sum()),
        gpu_null=int(df["gpu_passmark_g3d_mark"].isna().sum()),
        cpu_zero=int((df["cpu_passmark_mark"] == 0).sum()),
        gpu_zero=int((df["gpu_passmark_g3d_mark"] == 0).sum()))
    return records, expected


def run_import(conn, csv_path: Path, create_schema: bool = True) -> dict:
    records, expected = read_csv_records(csv_path)
    if create_schema:
        db.create_schema(conn)
    counts = db.upsert_laptops(conn, records)
    actual = db.catalog_stats(conn)
    ok = (actual["n"] >= expected["n"] and
          actual["cpu_null"] == expected["cpu_null"] and actual["gpu_null"] == expected["gpu_null"] and
          actual["cpu_zero"] == expected["cpu_zero"] and actual["gpu_zero"] == expected["gpu_zero"])
    return dict(counts=counts, expected=expected, actual=actual, ok=ok)


def load_config():
    cfg = db.read_config_from_env()
    if cfg:
        return cfg
    secrets = ROOT / ".streamlit" / "secrets.toml"
    if secrets.exists():
        try:
            import tomllib  # Python 3.11+
        except ImportError:
            raise SystemExit(
                "Python < 3.11 tidak dapat membaca secrets.toml dari skrip ini.\n"
                "Gunakan variabel lingkungan (PowerShell):\n"
                '  $env:MYSQL_HOST="127.0.0.1"; $env:MYSQL_USER="..."; $env:MYSQL_PASSWORD="..."; $env:MYSQL_DATABASE="laptop_recommender"')
        with open(secrets, "rb") as f:
            sec = tomllib.load(f).get("mysql")
        if sec:
            return dict(host=sec["host"], port=int(sec.get("port", 3306)), user=sec["user"],
                        password=sec.get("password", ""), database=sec["database"])
    raise SystemExit("Konfigurasi MySQL tidak ditemukan. Isi .streamlit/secrets.toml (lihat secrets.toml.example) "
                     "atau set variabel lingkungan MYSQL_HOST/USER/PASSWORD/DATABASE.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Impor katalog laptop CSV -> MySQL.")
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--dry-run", action="store_true", help="hanya validasi CSV")
    args = ap.parse_args(argv)
    try:
        if args.dry_run:
            records, expected = read_csv_records(args.csv)
            print(f"CSV valid: {expected['n']} baris; cpu_passmark_mark NULL={expected['cpu_null']}; "
                  f"gpu_passmark_g3d_mark NULL={expected['gpu_null']}.")
            print("Dry-run selesai; database tidak disentuh.")
            return 0
        conn = db.connect(load_config())
        try:
            r = run_import(conn, args.csv)
        finally:
            conn.close()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # koneksi/SQL
        print(f"ERROR database: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    c = r["counts"]
    print(f"Dimasukkan: {c['inserted']}, diperbarui (nilai berubah): {c['updated']}, tidak berubah: {c['unchanged']}")
    if c["changed_columns"]:
        print(f"Kolom yang berubah (jumlah baris): {c['changed_columns']}")
    print(f"Harapan (CSV): {r['expected']}")
    print(f"Aktual (DB)  : {r['actual']}")
    print("VERIFIKASI:", "OK" if r["ok"] else "GAGAL (jumlah baris/NULL tidak cocok dengan CSV)")
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
