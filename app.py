"""
RoboClean Predictive Maintenance Dashboard — single-file edition.

Everything (FastAPI backend + HTML + CSS + JS) lives in this one file.

Run:
    pip install -r requirements.txt
    python app.py
Then open:
    http://127.0.0.1:8000
"""
from __future__ import annotations

import json
import logging
import os
import threading
from contextlib import asynccontextmanager, contextmanager
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
log = logging.getLogger("dashboard")

# ============================================================================
# Data source
# ============================================================================
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "Lightcap/pudu-robot-operation-logs-bau-capstone-2026")
HF_DATASET_REVISION = os.getenv("HF_DATASET_REVISION", "main")
DUCKDB_PATH = os.getenv("DASHBOARD_DUCKDB_PATH", ":memory:")
DATASET_TABLES = {
    "public.robot_logs_error": "data/public_robot_logs_error.parquet",
    "public.robot_logs_error_training": "data/public_robot_logs_error_training.parquet",
    "public.robot_logs_error_validation": "data/public_robot_logs_error_validation.parquet",
    "public.robot_logs_error_test": "data/public_robot_logs_error_test.parquet",
    "public.robot_logs_info": "data/public_robot_logs_info.parquet",
    "model_training.training_runs": "data/model_training_training_runs.parquet",
    "model_training.training_artifacts": "data/model_training_training_artifacts.parquet",
    "model_training.training_source_tables": "data/model_training_training_source_tables.parquet",
}

DATA_CONN: duckdb.DuckDBPyConnection | None = None
DATA_LOCK = threading.RLock()


def _sql_path(path: str | Path) -> str:
    return str(path).replace("'", "''")


def init_pool() -> None:
    """Initialize DuckDB over the published Hugging Face Parquet snapshot."""
    global DATA_CONN
    if DATA_CONN is not None:
        return

    with DATA_LOCK:
        if DATA_CONN is not None:
            return
        conn = duckdb.connect(DUCKDB_PATH)
        conn.execute("CREATE SCHEMA IF NOT EXISTS public;")
        conn.execute("CREATE SCHEMA IF NOT EXISTS model_training;")

        for table_name, filename in DATASET_TABLES.items():
            local_path = hf_hub_download(
                repo_id=HF_DATASET_REPO,
                repo_type="dataset",
                filename=filename,
                revision=HF_DATASET_REVISION,
            )
            conn.execute(
                f"CREATE OR REPLACE VIEW {table_name} AS "
                f"SELECT * FROM read_parquet('{_sql_path(local_path)}');"
            )

        DATA_CONN = conn
        log.info("Dataset ready from Hugging Face: %s@%s", HF_DATASET_REPO, HF_DATASET_REVISION)


class DuckDictCursor:
    """Small adapter for the existing RealDictCursor-style endpoint code."""

    def __init__(self, conn: duckdb.DuckDBPyConnection):
        self.conn = conn
        self.result: duckdb.DuckDBPyConnection | None = None

    def execute(self, sql: str, params: tuple[Any, ...] | list[Any] | None = None) -> None:
        self.result = self.conn.execute(sql.replace("%s", "?"), params or [])

    def _as_dict(self, row: tuple[Any, ...] | None) -> dict[str, Any] | None:
        if row is None or self.result is None or self.result.description is None:
            return None
        names = [d[0] for d in self.result.description]
        return dict(zip(names, row))

    def fetchone(self) -> dict[str, Any] | None:
        if self.result is None:
            return None
        return self._as_dict(self.result.fetchone())

    def fetchall(self) -> list[dict[str, Any]]:
        if self.result is None:
            return []
        return [self._as_dict(row) for row in self.result.fetchall()]

    def close(self) -> None:
        self.result = None


@contextmanager
def get_cursor():
    if DATA_CONN is None:
        init_pool()
    with DATA_LOCK:
        cur = DuckDictCursor(DATA_CONN)
        try:
            yield cur
        finally:
            cur.close()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_pool()
    yield


# ============================================================================
# FastAPI
# ============================================================================
app = FastAPI(title="RoboClean Dashboard", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------- helpers ----------
CRITICAL_RATIO_THRESHOLD = 0.6
WARNING_RATIO_THRESHOLD = 0.3
HIGH_SEVERITY_LEVELS = {"critical", "error", "fatal"}
TIME_WINDOW_DAYS: dict[str, int | None] = {
    "all": None,
    "last_7_days": 7,
    "last_30_days": 30,
}
LOG_SEARCH_COLUMNS = (
    "robot_id",
    "product_code",
    "record_id",
    "error_id",
    "error_type",
    "error_level",
    "error_detail",
    "source_group",
    "source_file",
)


def _as_db_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _json_value(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _data_extent(cur) -> tuple[datetime, datetime]:
    """Full range: from earliest to latest log in the database."""
    cur.execute("SELECT MIN(task_time) AS min_t, MAX(task_time) AS max_t FROM public.robot_logs_error;")
    row = cur.fetchone()
    start = row["min_t"]
    end = row["max_t"] or datetime.now()
    if start is None:
        start = end - timedelta(days=7)
    return _as_db_datetime(start), _as_db_datetime(end)


def _resolve_time_window(
    cur,
    window: str | None = "all",
    start_date: date | None = None,
    end_date: date | None = None,
) -> tuple[datetime, datetime, dict[str, Any]]:
    """Resolve a requested window against the actual data extent.

    Relative windows are anchored to the latest timestamp in the source table,
    not to wall-clock today. That keeps historical snapshots inspectable.
    """
    extent_start, extent_end = _data_extent(cur)
    key = window or "all"

    if key == "custom":
        start = datetime.combine(start_date, time.min) if start_date else extent_start
        end = datetime.combine(end_date, time.max) if end_date else extent_end
    elif key in TIME_WINDOW_DAYS:
        days = TIME_WINDOW_DAYS[key]
        end = extent_end
        start = extent_start if days is None else max(extent_start, end - timedelta(days=days))
    else:
        raise HTTPException(status_code=422, detail=f"Unsupported time window: {key}")

    if start > end:
        raise HTTPException(status_code=422, detail="start_date must be before end_date")

    return start, end, {
        "key": key,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "extent_start": extent_start.isoformat(),
        "extent_end": extent_end.isoformat(),
    }


def _trend_bucket(start: datetime, end: datetime) -> str:
    """Pick a sensible date_trunc unit so the trend chart doesn't blow up."""
    days = (end - start).days
    if days <= 60:
        return "day"
    if days <= 365 * 2:
        return "week"
    return "month"


def _classify_status(error_level: str | None, hourly_ratio: float | None) -> str:
    if (error_level or "").lower() in HIGH_SEVERITY_LEVELS:
        return "Critical"
    r = hourly_ratio or 0.0
    if r >= CRITICAL_RATIO_THRESHOLD:
        return "Critical"
    if r >= WARNING_RATIO_THRESHOLD:
        return "Warning"
    return "Normal"


def _append_log_filters(
    where: list[str],
    params: list[Any],
    *,
    search: str | None = None,
    robot_id: str | None = None,
    fault_type: str | None = None,
    error_level: str | None = None,
) -> None:
    if search:
        like = f"%{search}%"
        where.append("(" + " OR ".join(f"COALESCE({col}, '') ILIKE %s" for col in LOG_SEARCH_COLUMNS) + ")")
        params.extend([like] * len(LOG_SEARCH_COLUMNS))
    if robot_id and robot_id.lower() != "all":
        where.append("robot_id = %s")
        params.append(robot_id)
    if fault_type and fault_type.lower() != "all":
        where.append("error_type = %s")
        params.append(fault_type)
    if error_level and error_level.lower() != "all":
        where.append("error_level = %s")
        params.append(error_level)


# ---------- API ----------
@app.get("/api/health")
def api_health() -> dict[str, Any]:
    try:
        with get_cursor() as cur:
            cur.execute("SELECT 1;")
            cur.fetchone()
        return {
            "ok": True,
            "source": "huggingface",
            "dataset": HF_DATASET_REPO,
            "revision": HF_DATASET_REVISION,
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/api/stats")
def api_stats(
    window: str = Query("all"),
    start_date: date | None = Query(default=None),
    end_date: date | None = Query(default=None),
) -> dict[str, Any]:
    with get_cursor() as cur:
        start, end, window_meta = _resolve_time_window(cur, window, start_date, end_date)
        comparable = window_meta["key"] != "all"
        prev_start = start - (end - start)
        prev_end = start

        cur.execute("SELECT COUNT(DISTINCT robot_id) AS n FROM public.robot_logs_error WHERE robot_id IS NOT NULL;")
        total = cur.fetchone()["n"] or 0
        cur.execute("SELECT COUNT(*) AS n FROM public.robot_logs_error WHERE task_time BETWEEN %s AND %s;", (start, end))
        log_records = cur.fetchone()["n"] or 0
        cur.execute(
            """
            WITH latest AS (
                SELECT DISTINCT ON (robot_id) robot_id, error_level, hourly_ratio
                FROM public.robot_logs_error
                WHERE task_time BETWEEN %s AND %s AND robot_id IS NOT NULL
                ORDER BY robot_id, task_time DESC
            )
            SELECT COUNT(*) AS active,
                   COUNT(*) FILTER (
                     WHERE LOWER(COALESCE(error_level,'')) IN ('critical','error','fatal') OR hourly_ratio >= %s
                   ) AS critical
            FROM latest;
            """,
            (start, end, CRITICAL_RATIO_THRESHOLD),
        )
        latest_counts = cur.fetchone()
        active = latest_counts["active"] or 0
        critical = latest_counts["critical"] or 0
        active_prev = None
        critical_prev = None
        if comparable:
            cur.execute(
                """
                WITH latest AS (
                    SELECT DISTINCT ON (robot_id) robot_id, error_level, hourly_ratio
                    FROM public.robot_logs_error
                    WHERE task_time BETWEEN %s AND %s AND robot_id IS NOT NULL
                    ORDER BY robot_id, task_time DESC
                )
                SELECT COUNT(*) AS active,
                       COUNT(*) FILTER (
                         WHERE LOWER(COALESCE(error_level,'')) IN ('critical','error','fatal') OR hourly_ratio >= %s
                       ) AS critical
                FROM latest;
                """,
                (prev_start, prev_end, CRITICAL_RATIO_THRESHOLD),
            )
            prev_counts = cur.fetchone()
            active_prev = prev_counts["active"] or 0
            critical_prev = prev_counts["critical"] or 0

        fleet = (1 - critical / active) * 100 if active else 0
        fleet_prev = (1 - critical_prev / active_prev) * 100 if active_prev else None

        def pct(now_v: float, prev_v: float | None) -> float | None:
            return None if prev_v in (None, 0) else round((now_v - prev_v) / prev_v * 100, 1)

        return {
            "range": window_meta,
            "comparison_label": "vs previous range" if comparable else "full data",
            "active_robots":   {"value": active,   "total": total, "delta_pct": pct(active, active_prev)},
            "critical_alerts": {"value": critical, "delta_pct": pct(critical, critical_prev)},
            "fleet_health":    {"value": round(fleet, 1), "delta_pct": None if fleet_prev is None else round(fleet - fleet_prev, 1)},
            "log_records":     {"value": int(log_records), "delta_pct": None},
        }


@app.get("/api/anomaly-trend")
def api_anomaly_trend(
    robot_id: str | None = Query(default=None),
    window: str = Query("all"),
    start_date: date | None = Query(default=None),
    end_date: date | None = Query(default=None),
) -> dict[str, Any]:
    with get_cursor() as cur:
        start, end, window_meta = _resolve_time_window(cur, window, start_date, end_date)
        bucket = _trend_bucket(start, end)
        params: list[Any] = [bucket, start, end]
        sql = """
            SELECT date_trunc(%s, task_time) AS bucket_start,
                   AVG(hourly_ratio) * 100 AS score,
                   COUNT(*) AS sample_n
            FROM public.robot_logs_error
            WHERE task_time BETWEEN %s AND %s
        """
        if robot_id:
            sql += " AND robot_id = %s"
            params.append(robot_id)
        sql += " GROUP BY 1 ORDER BY 1;"
        cur.execute(sql, params)
        rows = cur.fetchall()
        return {
            "range": window_meta,
            "bucket": bucket,
            "points": [
                {"date": r["bucket_start"].isoformat(), "score": round(float(r["score"] or 0), 1), "samples": int(r["sample_n"] or 0)}
                for r in rows
            ],
        }


@app.get("/api/fault-distribution")
def api_fault_distribution(
    window: str = Query("all"),
    start_date: date | None = Query(default=None),
    end_date: date | None = Query(default=None),
    limit: int = Query(6, ge=3, le=12),
) -> dict[str, Any]:
    with get_cursor() as cur:
        start, end, window_meta = _resolve_time_window(cur, window, start_date, end_date)
        cur.execute(
            """
            SELECT COALESCE(error_type,'Unknown') AS category, COUNT(*) AS cnt
            FROM public.robot_logs_error
            WHERE task_time BETWEEN %s AND %s
            GROUP BY 1 ORDER BY cnt DESC;
            """,
            (start, end),
        )
        rows = cur.fetchall()
        if not rows:
            return {"range": window_meta, "total": 0, "items": []}
        total = sum(r["cnt"] for r in rows)
        top, other = rows[:limit], rows[limit:]
        items = [{"label": r["category"], "count": int(r["cnt"]), "pct": round(r["cnt"] * 100 / total, 1)} for r in top]
        if other:
            on = sum(r["cnt"] for r in other)
            items.append({"label": "Other", "count": int(on), "pct": round(on * 100 / total, 1)})
        return {"range": window_meta, "total": int(total), "items": items}


@app.get("/api/robots")
def api_robots(
    page: int = Query(1, ge=1),
    page_size: int = Query(5, ge=1, le=100),
    search: str | None = None,
    status: str | None = None,
    fault_type: str | None = None,
    window: str = Query("all"),
    start_date: date | None = Query(default=None),
    end_date: date | None = Query(default=None),
) -> dict[str, Any]:
    with get_cursor() as cur:
        start, end, window_meta = _resolve_time_window(cur, window, start_date, end_date)
        sql = """
            WITH latest AS (
                SELECT DISTINCT ON (robot_id)
                       robot_id, product_code, error_type, error_detail, error_level,
                       hourly_ratio, task_time
                FROM public.robot_logs_error
                WHERE task_time BETWEEN %s AND %s AND robot_id IS NOT NULL
                ORDER BY robot_id, task_time DESC
            )
            SELECT * FROM latest WHERE TRUE
        """
        params: list[Any] = [start, end]
        if search:
            sql += " AND (robot_id ILIKE %s OR product_code ILIKE %s)"
            like = f"%{search}%"
            params += [like, like]
        if fault_type and fault_type.lower() != "all":
            sql += " AND error_type = %s"
            params.append(fault_type)
        cur.execute(sql, params)
        rows = cur.fetchall()

        out = []
        for r in rows:
            st = _classify_status(r["error_level"], r["hourly_ratio"])
            if status and status.lower() != "all" and st.lower() != status.lower():
                continue
            out.append({
                "robot_id": r["robot_id"],
                "area": r["product_code"] or "Unknown",
                "status": st,
                "predicted_fault": r["error_type"] or "No Fault Detected",
                "predicted_detail": r["error_detail"] or "All Systems Normal",
                "confidence": round((r["hourly_ratio"] or 0) * 100, 0),
                "last_updated": r["task_time"].isoformat() if r["task_time"] else None,
            })
        order = {"Critical": 0, "Warning": 1, "Normal": 2}
        out.sort(key=lambda x: (order.get(x["status"], 3), x["robot_id"]))
        start_idx = (page - 1) * page_size
        return {"range": window_meta, "total": len(out), "page": page, "page_size": page_size, "items": out[start_idx:start_idx + page_size]}


@app.get("/api/filter-options")
def api_filter_options() -> dict[str, Any]:
    with get_cursor() as cur:
        extent_start, extent_end = _data_extent(cur)
        cur.execute("SELECT COUNT(*) AS n, COUNT(DISTINCT robot_id) AS robots FROM public.robot_logs_error WHERE robot_id IS NOT NULL;")
        counts = cur.fetchone()
        cur.execute("SELECT DISTINCT robot_id FROM public.robot_logs_error WHERE robot_id IS NOT NULL ORDER BY robot_id;")
        robot_ids = [r["robot_id"] for r in cur.fetchall()]
        cur.execute("SELECT DISTINCT error_type FROM public.robot_logs_error WHERE error_type IS NOT NULL ORDER BY error_type;")
        fault_types = [r["error_type"] for r in cur.fetchall()]
        cur.execute("SELECT DISTINCT error_level FROM public.robot_logs_error WHERE error_level IS NOT NULL ORDER BY error_level;")
        error_levels = [r["error_level"] for r in cur.fetchall()]
        return {
            "range": {
                "extent_start": extent_start.isoformat(),
                "extent_end": extent_end.isoformat(),
            },
            "counts": {
                "logs": int(counts["n"] or 0),
                "robots": int(counts["robots"] or 0),
            },
            "robot_ids": robot_ids,
            "statuses": ["All Statuses", "Critical", "Warning", "Normal"],
            "fault_types": ["All Fault Types"] + fault_types,
            "error_levels": ["All Levels"] + error_levels,
        }


@app.get("/api/logs")
def api_logs(
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
    search: str | None = None,
    robot_id: str | None = None,
    fault_type: str | None = None,
    error_level: str | None = None,
    window: str = Query("all"),
    start_date: date | None = Query(default=None),
    end_date: date | None = Query(default=None),
) -> dict[str, Any]:
    with get_cursor() as cur:
        start, end, window_meta = _resolve_time_window(cur, window, start_date, end_date)
        where = ["task_time BETWEEN %s AND %s"]
        params: list[Any] = [start, end]
        _append_log_filters(
            where,
            params,
            search=search,
            robot_id=robot_id,
            fault_type=fault_type,
            error_level=error_level,
        )
        where_sql = " AND ".join(where)

        cur.execute(f"SELECT COUNT(*) AS n FROM public.robot_logs_error WHERE {where_sql};", params)
        total = int(cur.fetchone()["n"] or 0)

        offset = (page - 1) * page_size
        cur.execute(
            f"""
            SELECT ingest_id, source_group, source_file, robot_id, product_code, task_time,
                   record_id, error_id, error_type, error_level, error_detail,
                   hourly_error_count, pair_max_hourly_count, noise_threshold, hourly_ratio
            FROM public.robot_logs_error
            WHERE {where_sql}
            ORDER BY task_time DESC NULLS LAST, ingest_id DESC
            LIMIT %s OFFSET %s;
            """,
            params + [page_size, offset],
        )
        rows = cur.fetchall()
        return {
            "range": window_meta,
            "total": total,
            "page": page,
            "page_size": page_size,
            "items": [
                {
                    "ingest_id": r["ingest_id"],
                    "source_group": r["source_group"],
                    "source_file": r["source_file"],
                    "robot_id": r["robot_id"],
                    "product_code": r["product_code"],
                    "task_time": r["task_time"].isoformat() if r["task_time"] else None,
                    "record_id": r["record_id"],
                    "error_id": r["error_id"],
                    "error_type": r["error_type"],
                    "error_level": r["error_level"],
                    "error_detail": r["error_detail"],
                    "hourly_error_count": r["hourly_error_count"],
                    "pair_max_hourly_count": r["pair_max_hourly_count"],
                    "noise_threshold": r["noise_threshold"],
                    "hourly_ratio": float(r["hourly_ratio"] or 0),
                }
                for r in rows
            ],
        }


@app.get("/api/robot/{robot_id}")
def api_robot_detail(robot_id: str) -> dict[str, Any]:
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT robot_id, product_code, sn, mac, soft_version, hard_version, os_version,
                   error_type, error_detail, error_level, hourly_ratio, task_time
            FROM public.robot_logs_error
            WHERE robot_id = %s
            ORDER BY task_time DESC
            LIMIT 20;
            """,
            (robot_id,),
        )
        rows = cur.fetchall()
        if not rows:
            raise HTTPException(status_code=404, detail="Robot not found")
        h = rows[0]
        return {
            "robot_id": h["robot_id"], "product_code": h["product_code"], "sn": h["sn"], "mac": h["mac"],
            "soft_version": h["soft_version"], "hard_version": h["hard_version"], "os_version": h["os_version"],
            "recent_logs": [
                {
                    "task_time": r["task_time"].isoformat() if r["task_time"] else None,
                    "error_type": r["error_type"], "error_detail": r["error_detail"],
                    "error_level": r["error_level"], "hourly_ratio": float(r["hourly_ratio"] or 0),
                } for r in rows
            ],
        }


@app.get("/api/model-info")
def api_model_info() -> dict[str, Any]:
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT model_name, status, retrained, dataset_row_count, metrics, created_at
            FROM model_training.training_runs ORDER BY created_at DESC LIMIT 1;
            """
        )
        r = cur.fetchone()
        if not r:
            return {}
        return {
            "model_name": r["model_name"], "status": r["status"], "retrained": r["retrained"],
            "dataset_row_count": r["dataset_row_count"], "metrics": _json_value(r["metrics"]),
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
        }


# ============================================================================
# Frontend  (HTML + CSS + JS — all inline)
# ============================================================================
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>RoboClean - Predictive Maintenance Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root{
  --sidebar-bg:#1f2a44;--sidebar-bg-2:#1a2540;--sidebar-fg:#cfd6e6;--sidebar-fg-mute:#8a96b3;
  --sidebar-active:#3b82f6;--bg:#f5f7fb;--card:#fff;--border:#e5e9f2;--text:#1f2937;--text-mute:#6b7280;
  --primary:#3b82f6;--primary-2:#2563eb;--green:#10b981;--green-soft:#d1fae5;
  --red:#ef4444;--red-soft:#fee2e2;--amber:#f59e0b;--amber-soft:#fef3c7;--blue-soft:#dbeafe;
  --shadow:0 1px 2px rgba(15,23,42,.04),0 1px 3px rgba(15,23,42,.06);--radius:12px;
}
*{box-sizing:border-box}html,body{margin:0;padding:0;overflow-x:hidden}
body{font-family:'Inter',-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;
  background:var(--bg);color:var(--text);font-size:14px;-webkit-font-smoothing:antialiased}
a{color:inherit;text-decoration:none}button{font-family:inherit;cursor:pointer}

.app{display:flex;min-height:100vh;max-width:100vw;overflow-x:hidden}
.sidebar{width:248px;background:linear-gradient(180deg,var(--sidebar-bg),var(--sidebar-bg-2));
  color:var(--sidebar-fg);display:flex;flex-direction:column;justify-content:space-between;
  padding:18px 14px;position:sticky;top:0;height:100vh;flex-shrink:0}
.brand{display:flex;align-items:center;gap:10px;padding:6px 8px 18px;
  border-bottom:1px solid rgba(255,255,255,.06);margin-bottom:12px}
.brand-logo{width:38px;height:38px;border-radius:10px;
  background:linear-gradient(135deg,#3b82f6,#60a5fa);display:grid;place-items:center;color:#fff}
.brand-title{font-weight:700;font-size:16px;color:#fff}
.brand-sub{font-size:11px;color:var(--sidebar-fg-mute)}
.nav{display:flex;flex-direction:column;gap:2px}
.nav-item{display:flex;align-items:center;gap:12px;padding:10px 12px;border-radius:8px;
  color:var(--sidebar-fg);font-size:13.5px;transition:background .15s}
.nav-item:hover{background:rgba(255,255,255,.05)}
.nav-item.active{background:var(--sidebar-active);color:#fff;box-shadow:0 4px 14px rgba(59,130,246,.35)}
.nav-icon{width:18px;text-align:center;opacity:.9}
.nav-label{flex:1}.nav-arrow{color:var(--sidebar-fg-mute)}

.sidebar-bottom{display:flex;flex-direction:column;gap:10px;padding:8px}
.status-card,.user-card{display:flex;align-items:center;gap:10px;background:rgba(255,255,255,.04);
  padding:10px 12px;border-radius:10px}
.status-dot{width:9px;height:9px;border-radius:50%;background:#10b981;
  box-shadow:0 0 0 4px rgba(16,185,129,.18)}
.status-title{font-size:12.5px;font-weight:600;color:#fff}
.status-sub{font-size:11px;color:var(--sidebar-fg-mute)}
.user-avatar{width:36px;height:36px;border-radius:50%;
  background:linear-gradient(135deg,#6366f1,#3b82f6);color:#fff;display:grid;place-items:center;
  font-size:12px;font-weight:700}
.user-meta{flex:1;min-width:0}
.user-name{font-size:12.5px;font-weight:600;color:#fff}
.user-mail{font-size:11px;color:var(--sidebar-fg-mute);
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.user-caret{color:var(--sidebar-fg-mute)}

.main{flex:1;padding:24px 28px 32px;min-width:0;max-width:100%;overflow-x:hidden}
.topbar{display:flex;justify-content:space-between;align-items:flex-start;
  gap:16px;flex-wrap:wrap;margin-bottom:20px}
.hamburger{display:none;border:1px solid var(--border);background:#fff;
  width:38px;height:38px;border-radius:8px;font-size:18px}
.topbar-left h1{margin:0 0 4px;font-size:22px;font-weight:700}
.topbar-left p{margin:0;color:var(--text-mute);font-size:13px}
.topbar-right{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.bell{position:relative;width:38px;height:38px;border:1px solid var(--border);
  background:#fff;border-radius:50%;font-size:16px}
.bell-badge{position:absolute;top:-2px;right:-2px;background:var(--red);color:#fff;
  border-radius:999px;padding:1px 6px;font-size:10px;font-weight:700;border:2px solid #fff}
.range-panel{display:flex;align-items:center;gap:8px;flex-wrap:wrap;justify-content:flex-end}
.range-panel .select{min-width:160px}
.custom-range{display:flex;align-items:center;gap:6px}
.custom-range[hidden]{display:none}
.date-input{border:1px solid var(--border);padding:8px 10px;border-radius:8px;
  background:#fff;font-size:13px;color:var(--text);min-width:136px}
.date-label{background:#fff;border:1px solid var(--border);padding:8px 12px;
  border-radius:8px;font-size:12.5px;color:var(--text-mute);white-space:nowrap}
.btn-primary{background:var(--primary);color:#fff;border:none;padding:9px 16px;
  border-radius:8px;font-weight:600;font-size:13px;display:inline-flex;align-items:center;gap:6px;
  transition:background .15s}
.btn-primary:hover{background:var(--primary-2)}

.card{background:var(--card);border-radius:var(--radius);border:1px solid var(--border);
  box-shadow:var(--shadow);padding:20px;min-width:0}
.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:18px;margin-bottom:18px}

.stat{position:relative;display:grid;grid-template-columns:64px minmax(0,1fr);grid-template-rows:auto auto;
  align-items:center;column-gap:14px;overflow:hidden;padding-bottom:26px}
.stat-icon{width:56px;height:56px;border-radius:14px;display:grid;place-items:center;grid-row:span 2}
.icon-blue{background:var(--blue-soft);color:#3b82f6}
.icon-red{background:var(--red-soft);color:var(--red)}
.icon-green{background:var(--green-soft);color:var(--green)}
.icon-slate{background:#e2e8f0;color:#475569}
.stat-body{min-width:0}
.stat-title{font-size:13px;color:var(--text-mute);font-weight:600;margin-bottom:4px}
.stat-value{font-size:30px;font-weight:700;line-height:1.1;overflow-wrap:anywhere}
.stat-sub{font-size:12px;color:var(--text-mute);margin-top:2px}
.stat-trend{grid-column:2/3;text-align:left;margin-top:8px;min-width:0}
.badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:700}
.badge-up{background:var(--green-soft);color:#047857}
.badge-down{background:var(--red-soft);color:#b91c1c}
.badge-flat{background:#f3f4f6;color:#4b5563}
.trend-sub{display:block;font-size:11px;color:var(--text-mute);margin-top:4px}
.stat-bar{position:absolute;bottom:0;left:0;right:0;height:4px;background:#f1f3f7}
.stat-bar-fill{height:100%;transition:width .6s ease}
.stat-bar-fill.blue{background:var(--primary)}
.stat-bar-fill.red{background:var(--red)}
.stat-bar-fill.green{background:var(--green)}

.charts{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(360px,.75fr);gap:18px;margin-bottom:18px}
.chart-card{display:flex;flex-direction:column;min-width:0}
.chart-head{display:flex;justify-content:space-between;align-items:center;
  margin-bottom:14px;gap:12px;flex-wrap:wrap}
.chart-head h3{margin:0;font-size:15px;font-weight:600}
.chart-controls{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.info{color:var(--text-mute);font-size:12px;cursor:help}
.chart-body{position:relative;height:260px}
.donut-wrap{display:grid;grid-template-columns:minmax(170px,230px) minmax(0,1fr);
  align-items:center;justify-content:center;gap:18px;overflow:hidden}
.donut-wrap canvas{max-width:230px;max-height:230px}
.legend{list-style:none;padding:0;margin:0;display:flex;flex-direction:column;gap:8px;min-width:0;overflow:hidden}
.legend li{display:grid;grid-template-columns:minmax(0,1fr) auto;align-items:center;
  font-size:13px;gap:10px;min-width:0}
.legend .label{display:flex;align-items:center;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.legend .dot{width:9px;height:9px;border-radius:50%;display:inline-block;margin-right:8px;flex:0 0 auto}
.legend .value{font-weight:600;white-space:nowrap;font-size:12.5px}
.legend .pct{color:var(--text-mute);margin-left:4px}

.table-card{padding-bottom:14px;margin-bottom:18px;min-width:0}
.table-head{display:flex;justify-content:space-between;align-items:center;
  margin-bottom:14px;gap:12px;flex-wrap:wrap}
.table-title h3{margin:0;font-size:15px;font-weight:600}
.table-title p{margin:4px 0 0;color:var(--text-mute);font-size:12.5px}
.table-controls{display:flex;gap:10px;flex-wrap:wrap}
.search{position:relative}
.search-icon{position:absolute;left:10px;top:50%;transform:translateY(-50%);
  font-size:12px;color:var(--text-mute)}
.search input{border:1px solid var(--border);padding:8px 12px 8px 30px;border-radius:8px;
  background:#fff;font-size:13px;width:220px}
.select{border:1px solid var(--border);padding:8px 32px 8px 12px;border-radius:8px;
  background:#fff url("data:image/svg+xml;utf8,<svg fill='none' stroke='%236b7280' stroke-width='2' viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'><polyline points='6 9 12 15 18 9'/></svg>") no-repeat right 10px center;
  background-size:12px;appearance:none;font-size:13px;min-width:150px}

.table-wrap{overflow-x:auto}
.robot-table{width:100%;border-collapse:collapse;font-size:13.5px}
.robot-table thead th{text-align:left;padding:10px 14px;border-bottom:1px solid var(--border);
  color:var(--text-mute);font-weight:600;font-size:12.5px;background:#fafbfc}
.robot-table .action-col{text-align:right}
.robot-table tbody td{padding:14px;border-bottom:1px solid var(--border);vertical-align:middle}
.robot-table tbody tr:hover{background:#f9fafc}
.robot-table tbody tr:last-child td{border-bottom:none}
.robot-id-cell{display:flex;align-items:center;gap:10px}
.robot-thumb{width:40px;height:40px;border-radius:50%;background:#f1f5f9;
  display:grid;place-items:center;flex-shrink:0;color:#475569}
.robot-id{font-weight:600}
.robot-area{font-size:11.5px;color:var(--text-mute)}

.status{display:inline-flex;align-items:center;gap:6px;padding:4px 10px;border-radius:999px;
  font-size:12px;font-weight:600}
.status::before{content:"";width:6px;height:6px;border-radius:50%}
.status.critical{background:var(--red-soft);color:#b91c1c}
.status.critical::before{background:var(--red)}
.status.warning{background:var(--amber-soft);color:#b45309}
.status.warning::before{background:var(--amber)}
.status.normal{background:var(--green-soft);color:#047857}
.status.normal::before{background:var(--green)}

.fault-name{font-weight:500}
.fault-detail{font-size:11.5px;color:var(--text-mute)}
.confidence{display:flex;align-items:center;gap:10px;min-width:130px}
.confidence-num{width:36px;font-weight:600;font-size:12.5px}
.confidence-bar{flex:1;height:6px;border-radius:999px;background:#f1f3f7;overflow:hidden}
.confidence-fill{height:100%;border-radius:999px}
.confidence-fill.red{background:var(--red)}
.confidence-fill.amber{background:var(--amber)}
.confidence-fill.green{background:var(--green)}

.btn-secondary{background:var(--primary);color:#fff;border:none;padding:6px 14px;
  border-radius:8px;font-size:12.5px;font-weight:600;transition:background .15s}
.btn-secondary:hover{background:var(--primary-2)}
.kebab{background:transparent;border:none;color:var(--text-mute);padding:4px 8px;font-size:16px;line-height:1}
.empty{text-align:center;padding:40px;color:var(--text-mute)}

.table-foot{display:flex;justify-content:space-between;align-items:center;
  padding:14px 4px 4px;flex-wrap:wrap;gap:10px}
#paginationInfo,#logPaginationInfo{font-size:12.5px;color:var(--text-mute)}
.pagination{display:flex;gap:4px;flex-wrap:wrap;justify-content:flex-end}
.pagination button{border:1px solid var(--border);background:#fff;width:32px;height:32px;
  border-radius:6px;font-size:12.5px;color:var(--text)}
.pagination button.active{background:var(--primary);color:#fff;border-color:var(--primary)}
.pagination button:disabled{opacity:.4;cursor:not-allowed}

.log-table{min-width:1180px}
.log-table .time-col{white-space:nowrap}
.log-table .detail-col{max-width:360px}
.log-detail{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.level-pill{display:inline-flex;align-items:center;border-radius:999px;padding:4px 9px;
  font-size:12px;font-weight:600;background:#eef2ff;color:#4338ca}
.level-pill.event{background:#e0f2fe;color:#0369a1}
.level-pill.warning{background:var(--amber-soft);color:#b45309}
.level-pill.error{background:var(--red-soft);color:#b91c1c}
.level-pill.fatal{background:#111827;color:#fff}
.metric{font-variant-numeric:tabular-nums;white-space:nowrap}
.page-size{min-width:112px}

.modal-backdrop[hidden]{display:none}
.modal-backdrop{position:fixed;inset:0;background:rgba(15,23,42,.55);
  display:grid;place-items:center;padding:20px;z-index:100}
.modal{background:#fff;border-radius:14px;width:min(640px,100%);max-height:85vh;
  overflow:hidden;display:flex;flex-direction:column;
  box-shadow:0 20px 50px rgba(15,23,42,.25)}
.modal-head{display:flex;justify-content:space-between;align-items:center;
  padding:16px 20px;border-bottom:1px solid var(--border)}
.modal-head h2{margin:0;font-size:16px}
.modal-close{border:none;background:transparent;font-size:18px;color:var(--text-mute)}
.modal-body{padding:18px 20px;overflow:auto;font-size:13.5px}
.modal-body .meta-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));
  gap:8px 20px;margin-bottom:16px}
.modal-body .meta-grid div{display:flex;flex-direction:column}
.modal-body .meta-grid .k{font-size:11.5px;color:var(--text-mute);
  text-transform:uppercase;letter-spacing:.04em}
.modal-body .meta-grid .v{font-weight:600;word-break:break-all}
.modal-body h4{margin:18px 0 8px;font-size:13px}
.modal-body table{width:100%;border-collapse:collapse;font-size:12.5px}
.modal-body table th,.modal-body table td{padding:8px 6px;
  border-bottom:1px solid var(--border);text-align:left}

@media (max-width:1500px){.cards{grid-template-columns:repeat(2,minmax(0,1fr))}
  .charts{grid-template-columns:1fr}}
@media (max-width:880px){.cards{grid-template-columns:1fr}
  .stat{grid-template-columns:56px minmax(0,1fr)}}
@media (max-width:760px){.sidebar{position:fixed;left:0;top:0;transform:translateX(-100%);
  transition:transform .25s;z-index:50;box-shadow:4px 0 20px rgba(0,0,0,.15)}
  .sidebar.open{transform:translateX(0)}
  .main{padding:18px 16px 28px}
  .hamburger{display:inline-flex;align-items:center;justify-content:center}
  .topbar{align-items:center}
  .topbar-right{width:100%;justify-content:flex-end}
  .range-panel{width:100%}.range-panel .select,.date-label{flex:1}
  .chart-body.donut-wrap{height:auto;min-height:0}
  .donut-wrap{grid-template-columns:1fr}
  .donut-wrap canvas{max-width:220px;margin:0 auto}
  .topbar-left h1{font-size:19px}
  .search input{width:100%}.search{flex:1;min-width:200px}.select{flex:1;min-width:140px}}
@media (max-width:520px){.stat{grid-template-columns:48px 1fr;padding-bottom:24px}
  .stat-trend{grid-column:2/3;text-align:left;margin-top:4px}
  .stat-icon{grid-row:span 2;width:44px;height:44px}
  .stat-value{font-size:24px}
  .custom-range{width:100%}.date-input{flex:1;min-width:0}}
</style>
</head>
<body>
<div class="app">
  <aside class="sidebar" id="sidebar">
    <div class="sidebar-top">
      <div class="brand">
        <div class="brand-logo">
          <svg viewBox="0 0 32 32" width="32" height="32" fill="none" stroke="currentColor"
               stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <rect x="6" y="10" width="20" height="14" rx="3" />
            <circle cx="12" cy="17" r="1.5" fill="currentColor" />
            <circle cx="20" cy="17" r="1.5" fill="currentColor" />
            <path d="M16 10V6" /><circle cx="16" cy="5" r="1.5" />
          </svg>
        </div>
        <div>
          <div class="brand-title">RoboClean</div>
          <div class="brand-sub">Predictive Maintenance</div>
        </div>
      </div>
      <nav class="nav">
        <a class="nav-item active" href="#"><span class="nav-icon">⌂</span><span class="nav-label">Dashboard</span></a>
        <a class="nav-item" href="#"><span class="nav-icon">⚙</span><span class="nav-label">Robot Monitoring</span><span class="nav-arrow">›</span></a>
        <a class="nav-item" href="#"><span class="nav-icon">▣</span><span class="nav-label">Predictions &amp; Analysis</span><span class="nav-arrow">›</span></a>
        <a class="nav-item" href="#"><span class="nav-icon">⏲</span><span class="nav-label">Fault History</span><span class="nav-arrow">›</span></a>
        <a class="nav-item" href="#"><span class="nav-icon">▤</span><span class="nav-label">Maintenance Logs</span></a>
        <a class="nav-item" href="#"><span class="nav-icon">◈</span><span class="nav-label">Model Performance</span></a>
        <a class="nav-item" href="#"><span class="nav-icon">⚙</span><span class="nav-label">Settings</span></a>
      </nav>
    </div>
    <div class="sidebar-bottom">
      <div class="status-card">
        <span class="status-dot"></span>
        <div><div class="status-title">System Status</div><div class="status-sub">All Systems Operational</div></div>
      </div>
      <div class="user-card">
        <div class="user-avatar">AE</div>
        <div class="user-meta"><div class="user-name">Admin User</div><div class="user-mail">admin@roboclean.com</div></div>
        <span class="user-caret">▾</span>
      </div>
    </div>
  </aside>

  <main class="main">
    <header class="topbar">
      <button class="hamburger" id="hamburger" aria-label="Open menu">☰</button>
      <div class="topbar-left">
        <h1>Dashboard</h1>
        <p>Real-time overview of your autonomous cleaning robots</p>
      </div>
      <div class="topbar-right">
        <button class="bell" aria-label="Notifications"><span>🔔</span><span class="bell-badge">3</span></button>
        <div class="range-panel">
          <select class="select" id="rangeSelect">
            <option value="all">All data</option>
            <option value="last_30_days">Last 30 data days</option>
            <option value="last_7_days">Last 7 data days</option>
            <option value="custom">Custom range</option>
          </select>
          <div class="custom-range" id="customRange" hidden>
            <input class="date-input" type="date" id="startDate" aria-label="Start date" />
            <input class="date-input" type="date" id="endDate" aria-label="End date" />
          </div>
          <div class="date-label" id="dateRange">Loading…</div>
        </div>
      </div>
    </header>

    <section class="cards">
      <div class="card stat">
        <div class="stat-icon icon-blue">
          <svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <rect x="4" y="8" width="16" height="11" rx="2"></rect><path d="M8 8V6a4 4 0 0 1 8 0v2"></path>
            <circle cx="9" cy="13" r="1"></circle><circle cx="15" cy="13" r="1"></circle></svg>
        </div>
        <div class="stat-body">
          <div class="stat-title">Active Robots</div>
          <div class="stat-value" id="activeRobotsValue">—</div>
          <div class="stat-sub" id="activeRobotsSub">of — total robots</div>
        </div>
        <div class="stat-trend" id="activeRobotsTrend"><span class="badge badge-flat">Loading</span><span class="trend-sub">selected range</span></div>
        <div class="stat-bar"><div class="stat-bar-fill blue" id="activeRobotsBar" style="width:0%"></div></div>
      </div>

      <div class="card stat">
        <div class="stat-icon icon-red">
          <svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 3 L22 20 L2 20 Z"></path><path d="M12 9v5"></path><circle cx="12" cy="17" r="1" fill="currentColor"></circle></svg>
        </div>
        <div class="stat-body">
          <div class="stat-title">Critical Fault Alerts</div>
          <div class="stat-value" id="criticalValue">—</div>
          <div class="stat-sub">robots require attention</div>
        </div>
        <div class="stat-trend" id="criticalTrend"><span class="badge badge-flat">Loading</span><span class="trend-sub">selected range</span></div>
        <div class="stat-bar"><div class="stat-bar-fill red" id="criticalBar" style="width:0%"></div></div>
      </div>

      <div class="card stat">
        <div class="stat-icon icon-green">
          <svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"></path></svg>
        </div>
        <div class="stat-body">
          <div class="stat-title">Overall Fleet Health</div>
          <div class="stat-value" id="fleetValue">—%</div>
          <div class="stat-sub">healthy robots</div>
        </div>
        <div class="stat-trend" id="fleetTrend"><span class="badge badge-flat">Loading</span><span class="trend-sub">selected range</span></div>
        <div class="stat-bar"><div class="stat-bar-fill green" id="fleetBar" style="width:0%"></div></div>
      </div>

      <div class="card stat">
        <div class="stat-icon icon-slate">
          <svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M8 6h13"></path><path d="M8 12h13"></path><path d="M8 18h13"></path><path d="M3 6h.01"></path><path d="M3 12h.01"></path><path d="M3 18h.01"></path></svg>
        </div>
        <div class="stat-body">
          <div class="stat-title">Fault Log Records</div>
          <div class="stat-value" id="logRecordsValue">—</div>
          <div class="stat-sub" id="logRecordsSub">records in range</div>
        </div>
        <div class="stat-trend" id="logRecordsTrend"><span class="badge badge-flat">All data</span><span class="trend-sub">selected range</span></div>
        <div class="stat-bar"><div class="stat-bar-fill blue" id="logRecordsBar" style="width:0%"></div></div>
      </div>
    </section>

    <section class="charts">
      <div class="card chart-card">
        <div class="chart-head">
          <h3>Sensor Anomaly Trend <span class="info" title="Mean hourly anomaly ratio (%) per day across all logs in window">ⓘ</span></h3>
          <div class="chart-controls">
            <select class="select" id="anomalyRobotSelect"><option value="">All Robots</option></select>
          </div>
        </div>
        <div class="chart-body"><canvas id="anomalyChart"></canvas></div>
      </div>
      <div class="card chart-card">
        <div class="chart-head">
          <h3>Fault Distribution <span class="info" title="Top error_type categories in window">ⓘ</span></h3>
        </div>
        <div class="chart-body donut-wrap"><canvas id="faultChart"></canvas><ul class="legend" id="faultLegend"></ul></div>
      </div>
    </section>

    <section class="card table-card">
      <div class="table-head">
        <div class="table-title">
          <h3>Robot Snapshot</h3>
          <p>Latest record per robot in the selected range</p>
        </div>
        <div class="table-controls">
          <div class="search">
            <span class="search-icon">🔎</span>
            <input type="search" id="robotSearch" placeholder="Search robot ID..." />
          </div>
          <select class="select" id="statusFilter"></select>
          <select class="select" id="faultFilter"></select>
        </div>
      </div>
      <div class="table-wrap">
        <table class="robot-table">
          <thead><tr>
            <th>Robot ID</th><th>Status</th><th>Predicted Fault</th>
            <th>Confidence</th><th>Last Updated <span>↕</span></th><th class="action-col">Action</th>
          </tr></thead>
          <tbody id="robotTableBody"><tr><td colspan="6" class="empty">Loading…</td></tr></tbody>
        </table>
      </div>
      <div class="table-foot">
        <div id="paginationInfo">Showing 0 of 0 robots</div>
        <nav class="pagination" id="pagination"></nav>
      </div>
    </section>

    <section class="card table-card log-card">
      <div class="table-head">
        <div class="table-title">
          <h3>Fault Log Explorer</h3>
          <p id="logScope">All source log rows in the selected range</p>
        </div>
        <div class="table-controls">
          <div class="search">
            <span class="search-icon">🔎</span>
            <input type="search" id="logSearch" placeholder="Search logs..." />
          </div>
          <select class="select" id="logRobotFilter"><option value="all">All Robots</option></select>
          <select class="select" id="logLevelFilter"></select>
          <select class="select" id="logFaultFilter"></select>
          <select class="select page-size" id="logPageSize">
            <option value="25">25 / page</option>
            <option value="50">50 / page</option>
            <option value="100">100 / page</option>
          </select>
        </div>
      </div>
      <div class="table-wrap">
        <table class="robot-table log-table">
          <thead><tr>
            <th>Time</th><th>Robot ID</th><th>Level</th><th>Fault Type</th>
            <th>Detail</th><th>Hourly Ratio</th><th>Source</th>
          </tr></thead>
          <tbody id="logTableBody"><tr><td colspan="7" class="empty">Loading…</td></tr></tbody>
        </table>
      </div>
      <div class="table-foot">
        <div id="logPaginationInfo">Showing 0 of 0 logs</div>
        <nav class="pagination" id="logPagination"></nav>
      </div>
    </section>
  </main>
</div>

<div class="modal-backdrop" id="modalBackdrop" hidden>
  <div class="modal" role="dialog" aria-modal="true">
    <div class="modal-head">
      <h2 id="modalTitle">Robot Details</h2>
      <button class="modal-close" id="modalClose" aria-label="Close">✕</button>
    </div>
    <div class="modal-body" id="modalBody">Loading…</div>
  </div>
</div>

<script>
const API = {
  stats: "/api/stats", trend: "/api/anomaly-trend", faults: "/api/fault-distribution",
  robots: "/api/robots", logs: "/api/logs", options: "/api/filter-options",
  robot: (id)=>`/api/robot/${encodeURIComponent(id)}`, model: "/api/model-info",
};
const FAULT_COLORS = ["#3b82f6","#10b981","#f59e0b","#ef4444","#8b5cf6","#14b8a6","#94a3b8"];
const state = {
  window:"all", startDate:"", endDate:"", totalLogCount:0,
  robot:{ page:1, pageSize:10, search:"", status:"All Statuses", faultType:"All Fault Types" },
  logs:{ page:1, pageSize:25, search:"", robotId:"all", faultType:"All Fault Types", errorLevel:"All Levels" },
};
let anomalyChart=null, faultChart=null;

document.addEventListener("DOMContentLoaded", ()=>{ bindUI(); loadAll(); });

function bindUI(){
  document.getElementById("hamburger")?.addEventListener("click", ()=>{
    document.getElementById("sidebar").classList.toggle("open");
  });
  document.getElementById("rangeSelect").addEventListener("change",(e)=>{
    state.window = e.target.value;
    toggleCustomRange();
    resetWindowedPages();
    if(state.window !== "custom" || customRangeReady()) refreshWindowedData();
  });
  document.getElementById("startDate").addEventListener("change",(e)=>{
    state.startDate = e.target.value;
    resetWindowedPages();
    if(customRangeReady()) refreshWindowedData();
  });
  document.getElementById("endDate").addEventListener("change",(e)=>{
    state.endDate = e.target.value;
    resetWindowedPages();
    if(customRangeReady()) refreshWindowedData();
  });
  document.getElementById("robotSearch").addEventListener("input", debounce((e)=>{
    state.robot.search = e.target.value.trim(); state.robot.page=1; loadRobots();
  }, 300));
  document.getElementById("statusFilter").addEventListener("change",(e)=>{ state.robot.status=e.target.value; state.robot.page=1; loadRobots(); });
  document.getElementById("faultFilter").addEventListener("change",(e)=>{ state.robot.faultType=e.target.value; state.robot.page=1; loadRobots(); });
  document.getElementById("anomalyRobotSelect").addEventListener("change",(e)=> loadAnomalyTrend(e.target.value || null));
  document.getElementById("logSearch").addEventListener("input", debounce((e)=>{
    state.logs.search = e.target.value.trim(); state.logs.page=1; loadLogs();
  }, 300));
  document.getElementById("logRobotFilter").addEventListener("change",(e)=>{ state.logs.robotId=e.target.value; state.logs.page=1; loadLogs(); });
  document.getElementById("logLevelFilter").addEventListener("change",(e)=>{ state.logs.errorLevel=e.target.value; state.logs.page=1; loadLogs(); });
  document.getElementById("logFaultFilter").addEventListener("change",(e)=>{ state.logs.faultType=e.target.value; state.logs.page=1; loadLogs(); });
  document.getElementById("logPageSize").addEventListener("change",(e)=>{ state.logs.pageSize=parseInt(e.target.value,10); state.logs.page=1; loadLogs(); });
  document.getElementById("modalClose").addEventListener("click", closeModal);
  document.getElementById("modalBackdrop").addEventListener("click",(e)=>{ if(e.target.id==="modalBackdrop") closeModal(); });
}

async function loadAll(){
  await loadFilterOptions();
  await refreshWindowedData();
}

async function refreshWindowedData(){
  await Promise.all([loadStats(), loadAnomalyTrend(), loadFaultDistribution(), loadRobots(), loadLogs()]);
}

function resetWindowedPages(){
  state.robot.page = 1;
  state.logs.page = 1;
}

function toggleCustomRange(){
  document.getElementById("customRange").hidden = state.window !== "custom";
}

function customRangeReady(){
  return Boolean(state.startDate && state.endDate);
}

function rangeParams(){
  const params = new URLSearchParams({ window:state.window });
  if(state.window === "custom"){
    if(state.startDate) params.set("start_date", state.startDate);
    if(state.endDate) params.set("end_date", state.endDate);
  }
  return params;
}

async function loadStats(){
  try{
    const data = await fetchJson(`${API.stats}?${rangeParams()}`);
    setStatCard("active", data.active_robots, `of ${data.active_robots.total} total robots`, false, data.comparison_label);
    setStatCard("critical", data.critical_alerts, "robots require attention", false, data.comparison_label);
    setStatCard("fleet", data.fleet_health, "healthy robots", true, data.comparison_label);
    setLogRecordsCard(data.log_records?.value || 0);
    if(data.range?.start && data.range?.end){
      document.getElementById("dateRange").textContent =
        `${formatRange(data.range.start)} - ${formatRange(data.range.end)}`;
    }
  }catch(e){ console.error("loadStats", e); }
}

function setStatCard(kind, payload, subText, isPercent=false, comparisonLabel="vs previous range"){
  const map = {
    active:   {v:"activeRobotsValue", s:"activeRobotsSub", t:"activeRobotsTrend", b:"activeRobotsBar"},
    critical: {v:"criticalValue",     s:null,              t:"criticalTrend",     b:"criticalBar"},
    fleet:    {v:"fleetValue",        s:null,              t:"fleetTrend",        b:"fleetBar"},
  }[kind];
  const v = document.getElementById(map.v);
  v.textContent = isPercent ? `${payload.value}%` : formatNumber(payload.value);
  if(map.s) document.getElementById(map.s).textContent = subText;
  const t = document.getElementById(map.t);
  const d = payload.delta_pct;
  if(d === null || d === undefined){
    t.innerHTML = `<span class="badge badge-flat">${escapeHtml(comparisonLabel)}</span><span class="trend-sub">selected range</span>`;
  }else{
    const arrow = d>0 ? "↑" : d<0 ? "↓" : "→";
    const cls = d>0 ? "badge-up" : d<0 ? "badge-down" : "badge-flat";
    t.innerHTML = `<span class="badge ${cls}">${arrow} ${Math.abs(d).toFixed(1)}%</span><span class="trend-sub">${escapeHtml(comparisonLabel)}</span>`;
  }
  const bar = document.getElementById(map.b);
  let pct = 0;
  if(kind==="active") pct = payload.total ? (payload.value/payload.total)*100 : 0;
  else if(kind==="fleet") pct = payload.value;
  else pct = Math.min(100, payload.value*8);
  bar.style.width = `${pct}%`;
}

function setLogRecordsCard(value){
  document.getElementById("logRecordsValue").textContent = formatNumber(value);
  document.getElementById("logRecordsSub").textContent = "records in range";
  const pct = state.totalLogCount ? Math.max(2, Math.min(100, value / state.totalLogCount * 100)) : 100;
  document.getElementById("logRecordsBar").style.width = `${pct}%`;
  document.getElementById("logRecordsTrend").innerHTML =
    `<span class="badge badge-flat">${formatNumber(state.totalLogCount)} total</span><span class="trend-sub">source rows</span>`;
}

async function loadAnomalyTrend(robotId=null){
  try{
    const params = rangeParams();
    if(robotId) params.set("robot_id", robotId);
    const data = await fetchJson(`${API.trend}?${params}`);
    const labels = data.points.map(p=>formatBucketDate(p.date, data.bucket));
    const scores = data.points.map(p=>p.score);
    renderAnomalyChart(labels, scores);
  }catch(e){ console.error("loadAnomalyTrend", e); }
}

function renderAnomalyChart(labels, scores){
  const ctx = document.getElementById("anomalyChart").getContext("2d");
  const grad = ctx.createLinearGradient(0,0,0,260);
  grad.addColorStop(0,"rgba(59,130,246,0.30)");
  grad.addColorStop(1,"rgba(59,130,246,0.00)");
  if(anomalyChart) anomalyChart.destroy();
  anomalyChart = new Chart(ctx, {
    type:"line",
    data:{ labels, datasets:[{
      label:"Anomaly Score", data:scores, borderColor:"#3b82f6",
      backgroundColor:grad, fill:true, tension:0.35, borderWidth:2,
      pointRadius:3, pointBackgroundColor:"#3b82f6", pointHoverRadius:6,
    }]},
    options:{
      responsive:true, maintainAspectRatio:false,
      plugins:{ legend:{display:false}, tooltip:{ backgroundColor:"#1f2937", padding:10,
        callbacks:{ label:(c)=>`Anomaly Score: ${c.parsed.y.toFixed(1)}` } } },
      scales:{
        x:{ grid:{display:false}, ticks:{color:"#94a3b8"} },
        y:{ beginAtZero:true, max:100, grid:{color:"#eef2f7"}, ticks:{color:"#94a3b8"} },
      },
    },
  });
}

async function loadFaultDistribution(){
  try{
    const params = rangeParams();
    params.set("limit", "6");
    const data = await fetchJson(`${API.faults}?${params}`);
    renderFaultChart(data);
  }catch(e){ console.error("loadFaultDistribution", e); }
}

function renderFaultChart({items, total}){
  const ctx = document.getElementById("faultChart").getContext("2d");
  if(!items || !items.length){
    if(faultChart) faultChart.destroy();
    document.getElementById("faultLegend").innerHTML = `<li>No data</li>`;
    return;
  }
  const labels = items.map(x=>x.label);
  const values = items.map(x=>x.count);
  const colors = items.map((_,i)=>FAULT_COLORS[i % FAULT_COLORS.length]);
  if(faultChart) faultChart.destroy();
  faultChart = new Chart(ctx, {
    type:"doughnut",
    data:{ labels, datasets:[{ data:values, backgroundColor:colors, borderWidth:0 }] },
    options:{ responsive:true, maintainAspectRatio:false, cutout:"70%",
      plugins:{ legend:{display:false},
        tooltip:{ callbacks:{ label:(c)=>`${c.label}: ${c.parsed}` } } } },
    plugins:[{ id:"center", beforeDraw(chart){
      const { ctx, chartArea:{left,right,top,bottom} } = chart;
      const cx=(left+right)/2, cy=(top+bottom)/2;
      ctx.save();
      ctx.fillStyle="#0f172a"; ctx.font="700 24px Inter, sans-serif";
      ctx.textAlign="center"; ctx.textBaseline="middle";
      ctx.fillText(total.toLocaleString(), cx, cy-8);
      ctx.fillStyle="#94a3b8"; ctx.font="500 11px Inter, sans-serif";
      ctx.fillText("Total Faults", cx, cy+12);
      ctx.restore();
    }}],
  });
  document.getElementById("faultLegend").innerHTML = items.map((it,i)=>`
    <li>
      <span class="label"><span class="dot" style="background:${colors[i]}"></span>${escapeHtml(it.label)}</span>
      <span><span class="value">${it.count}</span> <span class="pct">(${it.pct}%)</span></span>
    </li>`).join("");
}

async function loadFilterOptions(){
  try{
    const data = await fetchJson(API.options);
    state.totalLogCount = data.counts?.logs || 0;
    configureDateInputs(data.range);
    document.getElementById("statusFilter").innerHTML = data.statuses.map(s=>`<option value="${s}">${s}</option>`).join("");
    document.getElementById("faultFilter").innerHTML = data.fault_types.map(s=>`<option value="${s}">${s}</option>`).join("");
    document.getElementById("logFaultFilter").innerHTML = data.fault_types.map(s=>`<option value="${s}">${s}</option>`).join("");
    document.getElementById("logLevelFilter").innerHTML = data.error_levels.map(s=>`<option value="${s}">${s}</option>`).join("");

    const robotOptions = [`<option value="">All Robots</option>`]
      .concat(data.robot_ids.map(id=>`<option value="${escapeHtml(id)}">${escapeHtml(shortenId(id))}</option>`));
    document.getElementById("anomalyRobotSelect").innerHTML = robotOptions.join("");
    document.getElementById("logRobotFilter").innerHTML = [`<option value="all">All Robots</option>`]
      .concat(data.robot_ids.map(id=>`<option value="${escapeHtml(id)}">${escapeHtml(shortenId(id))}</option>`)).join("");
  }catch(e){ console.error("loadFilterOptions", e); }
}

function configureDateInputs(range){
  if(!range?.extent_start || !range?.extent_end) return;
  const min = toInputDate(range.extent_start);
  const max = toInputDate(range.extent_end);
  for(const id of ["startDate","endDate"]){
    const el = document.getElementById(id);
    el.min = min; el.max = max;
  }
  state.startDate = min;
  state.endDate = max;
  document.getElementById("startDate").value = min;
  document.getElementById("endDate").value = max;
}

async function loadRobots(){
  const params = rangeParams();
  params.set("page", String(state.robot.page));
  params.set("page_size", String(state.robot.pageSize));
  if(state.robot.search) params.set("search", state.robot.search);
  if(state.robot.status && state.robot.status!=="All Statuses") params.set("status", state.robot.status);
  if(state.robot.faultType && state.robot.faultType!=="All Fault Types") params.set("fault_type", state.robot.faultType);
  try{
    const data = await fetchJson(`${API.robots}?${params}`);
    renderRobotTable(data); renderPagination(data);
  }catch(e){
    document.getElementById("robotTableBody").innerHTML =
      `<tr><td colspan="6" class="empty">Failed to load: ${escapeHtml(String(e))}</td></tr>`;
  }
}

function renderRobotTable({items}){
  const body = document.getElementById("robotTableBody");
  if(!items.length){ body.innerHTML = `<tr><td colspan="6" class="empty">No robots found</td></tr>`; return; }
  body.innerHTML = items.map(r=>{
    const conf = Math.round(r.confidence ?? 0);
    let cc = "green"; if(conf>=60) cc="red"; else if(conf>=30) cc="amber";
    return `
      <tr>
        <td>
          <div class="robot-id-cell">
            <div class="robot-thumb">
              <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                <rect x="4" y="8" width="16" height="11" rx="2"></rect><path d="M8 8V6a4 4 0 0 1 8 0v2"></path>
                <circle cx="9" cy="13" r="1"></circle><circle cx="15" cy="13" r="1"></circle></svg>
            </div>
            <div>
              <div class="robot-id">${escapeHtml(shortenId(r.robot_id))}</div>
              <div class="robot-area">${escapeHtml(r.area || "")}</div>
            </div>
          </div>
        </td>
        <td><span class="status ${r.status.toLowerCase()}">${r.status}</span></td>
        <td>
          <div class="fault-name">${escapeHtml(r.predicted_fault)}</div>
          <div class="fault-detail">${escapeHtml(r.predicted_detail || "")}</div>
        </td>
        <td>
          <div class="confidence">
            <span class="confidence-num">${conf}%</span>
            <div class="confidence-bar"><div class="confidence-fill ${cc}" style="width:${conf}%"></div></div>
          </div>
        </td>
        <td>${formatLong(r.last_updated)}</td>
        <td class="action-col">
          <button class="btn-secondary" data-id="${escapeHtml(r.robot_id)}">View Details</button>
          <button class="kebab" aria-label="More">⋮</button>
        </td>
      </tr>`;
  }).join("");
  body.querySelectorAll(".btn-secondary").forEach(btn=>{
    btn.addEventListener("click", ()=> openRobotModal(btn.dataset.id));
  });
}

function renderPagination({total, page, page_size}){
  const totalPages = Math.max(1, Math.ceil(total/page_size));
  const startN = total===0 ? 0 : (page-1)*page_size + 1;
  const endN = Math.min(total, page*page_size);
  document.getElementById("paginationInfo").textContent = `Showing ${startN} to ${endN} of ${total} robots`;
  const pag = document.getElementById("pagination");
  const buttons = [`<button ${page===1?"disabled":""} data-go="${page-1}">‹</button>`];
  for(const p of pageWindow(page, totalPages, 5)){
    buttons.push(`<button class="${p===page?"active":""}" data-go="${p}">${p}</button>`);
  }
  buttons.push(`<button ${page===totalPages?"disabled":""} data-go="${page+1}">›</button>`);
  pag.innerHTML = buttons.join("");
  pag.querySelectorAll("button[data-go]").forEach(btn=>{
    btn.addEventListener("click", ()=>{
      const t = parseInt(btn.dataset.go,10);
      if(!isNaN(t) && t>=1 && t<=totalPages){ state.robot.page=t; loadRobots(); }
    });
  });
}

function pageWindow(current, total, span=5){
  const half = Math.floor(span/2);
  let start = Math.max(1, current-half);
  let end = Math.min(total, start+span-1);
  start = Math.max(1, end-span+1);
  const out=[]; for(let i=start;i<=end;i++) out.push(i);
  return out;
}

async function loadLogs(){
  const params = rangeParams();
  params.set("page", String(state.logs.page));
  params.set("page_size", String(state.logs.pageSize));
  if(state.logs.search) params.set("search", state.logs.search);
  if(state.logs.robotId && state.logs.robotId !== "all") params.set("robot_id", state.logs.robotId);
  if(state.logs.errorLevel && state.logs.errorLevel !== "All Levels") params.set("error_level", state.logs.errorLevel);
  if(state.logs.faultType && state.logs.faultType !== "All Fault Types") params.set("fault_type", state.logs.faultType);
  try{
    const data = await fetchJson(`${API.logs}?${params}`);
    renderLogTable(data);
    renderLogPagination(data);
    if(data.range?.start && data.range?.end){
      document.getElementById("logScope").textContent =
        `${formatNumber(data.total)} rows from ${formatRange(data.range.start)} to ${formatRange(data.range.end)}`;
    }
  }catch(e){
    document.getElementById("logTableBody").innerHTML =
      `<tr><td colspan="7" class="empty">Failed to load: ${escapeHtml(String(e))}</td></tr>`;
  }
}

function renderLogTable({items}){
  const body = document.getElementById("logTableBody");
  if(!items.length){ body.innerHTML = `<tr><td colspan="7" class="empty">No logs found</td></tr>`; return; }
  body.innerHTML = items.map(r=>{
    const ratio = ((r.hourly_ratio || 0) * 100).toFixed(1);
    const level = r.error_level || "Unknown";
    return `
      <tr>
        <td class="time-col">${formatLong(r.task_time)}</td>
        <td>
          <div class="fault-name">${escapeHtml(shortenId(r.robot_id || ""))}</div>
          <div class="fault-detail">${escapeHtml(r.product_code || "")}</div>
        </td>
        <td><span class="level-pill ${levelClass(level)}">${escapeHtml(level)}</span></td>
        <td>
          <div class="fault-name">${escapeHtml(r.error_type || "Unknown")}</div>
          <div class="fault-detail">${escapeHtml(r.error_id || r.record_id || "")}</div>
        </td>
        <td class="detail-col"><div class="log-detail" title="${escapeHtml(r.error_detail || "")}">${escapeHtml(r.error_detail || "-")}</div></td>
        <td class="metric">${ratio}%</td>
        <td>
          <div class="fault-name">${escapeHtml(r.source_group || "-")}</div>
          <div class="fault-detail">${escapeHtml(r.source_file || "")}</div>
        </td>
      </tr>`;
  }).join("");
}

function renderLogPagination({total, page, page_size}){
  const totalPages = Math.max(1, Math.ceil(total/page_size));
  const startN = total===0 ? 0 : (page-1)*page_size + 1;
  const endN = Math.min(total, page*page_size);
  document.getElementById("logPaginationInfo").textContent = `Showing ${formatNumber(startN)} to ${formatNumber(endN)} of ${formatNumber(total)} logs`;
  const pag = document.getElementById("logPagination");
  const buttons = [`<button ${page===1?"disabled":""} data-go="${page-1}">‹</button>`];
  for(const p of pageWindow(page, totalPages, 5)){
    buttons.push(`<button class="${p===page?"active":""}" data-go="${p}">${p}</button>`);
  }
  buttons.push(`<button ${page===totalPages?"disabled":""} data-go="${page+1}">›</button>`);
  pag.innerHTML = buttons.join("");
  pag.querySelectorAll("button[data-go]").forEach(btn=>{
    btn.addEventListener("click", ()=>{
      const t = parseInt(btn.dataset.go,10);
      if(!isNaN(t) && t>=1 && t<=totalPages){ state.logs.page=t; loadLogs(); }
    });
  });
}

async function openRobotModal(robotId){
  const backdrop = document.getElementById("modalBackdrop");
  const body = document.getElementById("modalBody");
  document.getElementById("modalTitle").textContent = `Robot ${shortenId(robotId)}`;
  body.innerHTML = "Loading…"; backdrop.hidden = false;
  try{
    const data = await fetchJson(API.robot(robotId));
    body.innerHTML = `
      <div class="meta-grid">
        <div><span class="k">Robot ID</span><span class="v">${escapeHtml(data.robot_id)}</span></div>
        <div><span class="k">Product</span><span class="v">${escapeHtml(data.product_code || "-")}</span></div>
        <div><span class="k">SN</span><span class="v">${escapeHtml(data.sn || "-")}</span></div>
        <div><span class="k">MAC</span><span class="v">${escapeHtml(data.mac || "-")}</span></div>
        <div><span class="k">Software</span><span class="v">${escapeHtml(data.soft_version || "-")}</span></div>
        <div><span class="k">OS</span><span class="v">${escapeHtml(data.os_version || "-")}</span></div>
      </div>
      <h4>Recent Logs (latest 20)</h4>
      <table>
        <thead><tr><th>Time</th><th>Type</th><th>Detail</th><th>Level</th><th>Hourly Ratio</th></tr></thead>
        <tbody>
          ${data.recent_logs.map(l=>`
            <tr>
              <td>${formatLong(l.task_time)}</td>
              <td>${escapeHtml(l.error_type || "")}</td>
              <td>${escapeHtml(l.error_detail || "")}</td>
              <td>${escapeHtml(l.error_level || "")}</td>
              <td>${(l.hourly_ratio*100).toFixed(1)}%</td>
            </tr>`).join("")}
        </tbody>
      </table>`;
  }catch(e){ body.innerHTML = `<p>Failed to load: ${escapeHtml(String(e))}</p>`; }
}

function closeModal(){ document.getElementById("modalBackdrop").hidden = true; }

async function fetchJson(url){
  const r = await fetch(url);
  if(!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

function debounce(fn, ms){ let t; return (...args)=>{ clearTimeout(t); t=setTimeout(()=>fn(...args),ms); }; }
function escapeHtml(s){ return String(s).replace(/[&<>"']/g, c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c])); }
function shortenId(id){ if(!id) return ""; return id.length>12 ? "RC-"+id.slice(-8) : id; }
function formatNumber(n){ return Number(n || 0).toLocaleString("en-US"); }
function formatRange(iso){ return new Date(iso).toLocaleDateString("en-US",{month:"short",day:"numeric",year:"numeric"}); }
function formatBucketDate(iso, bucket){
  const d = new Date(iso);
  if(bucket === "month") return d.toLocaleDateString("en-US",{month:"short",year:"numeric"});
  if(bucket === "week") return d.toLocaleDateString("en-US",{month:"short",day:"numeric",year:"numeric"});
  return d.toLocaleDateString("en-US",{month:"short",day:"numeric"});
}
function formatLong(iso){ if(!iso) return "—";
  return new Date(iso).toLocaleString("en-US",{month:"short",day:"numeric",year:"numeric",hour:"2-digit",minute:"2-digit",hour12:false}); }
function toInputDate(iso){ return String(iso).slice(0,10); }
function levelClass(level){ return String(level || "unknown").toLowerCase().replace(/[^a-z0-9_-]/g,""); }
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
    return HTMLResponse(content=INDEX_HTML)


# ============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
