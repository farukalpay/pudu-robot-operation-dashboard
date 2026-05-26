"""
RoboClean Predictive Maintenance Dashboard - single-file edition.

Reads the published Hugging Face dataset snapshot
    Lightcap/pudu-robot-operation-logs-bau-capstone-2026
via DuckDB - no live PostgreSQL connection required.

Pages:
    /#dashboard       - Real-time fleet overview
    /#fault-history   - Historical fault browser
    /#predictions     - AI predictions & analysis

Run:
    pip install -r requirements.txt
    python app.py
Open:
    http://127.0.0.1:8000
"""
from __future__ import annotations

import json
import logging
import os
import threading
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta
from typing import Any

import duckdb
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

load_dotenv()

from pudu_model_runtime import MODEL_RUNTIME, ModelHeadMetric, RuntimeSnapshot

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
log = logging.getLogger("dashboard")

MODEL_BINARY_THRESHOLD = 0.5

# ============================================================================
# Data source - Hugging Face + DuckDB
# ============================================================================
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "Lightcap/pudu-robot-operation-logs-bau-capstone-2026")
HF_DATASET_REVISION = os.getenv("HF_DATASET_REVISION", "main")
DUCKDB_PATH = os.getenv("DASHBOARD_DUCKDB_PATH", ":memory:")
DATASET_TABLES = {
    "public.robot_logs_error":             "data/public_robot_logs_error.parquet",
    "public.robot_logs_error_training":    "data/public_robot_logs_error_training.parquet",
    "public.robot_logs_error_validation":  "data/public_robot_logs_error_validation.parquet",
    "public.robot_logs_error_test":        "data/public_robot_logs_error_test.parquet",
    "public.robot_logs_info":              "data/public_robot_logs_info.parquet",
    "model_training.training_runs":        "data/model_training_training_runs.parquet",
    "model_training.training_artifacts":   "data/model_training_training_artifacts.parquet",
    "model_training.training_source_tables": "data/model_training_training_source_tables.parquet",
}

DATA_CONN: duckdb.DuckDBPyConnection | None = None
DATA_LOCK = threading.RLock()


def _sql_path(path: str) -> str:
    return str(path).replace("'", "''")


def init_pool() -> None:
    """Download the Parquet snapshot from Hugging Face and register as views."""
    global DATA_CONN
    if DATA_CONN is not None:
        return
    with DATA_LOCK:
        if DATA_CONN is not None:
            return
        log.info("Initialising dataset from %s@%s ...", HF_DATASET_REPO, HF_DATASET_REVISION)
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
        log.info("Dataset ready from Hugging Face")


class DuckDictCursor:
    """psycopg2-RealDictCursor look-alike on top of DuckDB."""

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
        return [self._as_dict(r) for r in self.result.fetchall()]

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
    try:
        init_pool()
    except Exception as e:
        log.warning("Dataset not ready at startup (%s); will retry on first request.", e)
    try:
        snapshot = MODEL_RUNTIME.ensure_loaded()
        log.info(
            "Model runtime %s from %s (%s)",
            snapshot.status,
            snapshot.repo_url,
            snapshot.git_commit or snapshot.error or "no commit",
        )
    except Exception as e:
        log.warning("Model runtime not ready at startup (%s); API will expose the error.", e)
    yield


# ============================================================================
# FastAPI
# ============================================================================
app = FastAPI(title="RoboClean Dashboard", version="2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------- helpers ----------
def _json_value(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _data_window(cur) -> tuple[datetime, datetime]:
    """Full range: from earliest to latest log in the dataset (naive)."""
    cur.execute("SELECT MIN(task_time) AS min_t, MAX(task_time) AS max_t FROM public.robot_logs_error;")
    row = cur.fetchone()
    start = row["min_t"]
    end = row["max_t"] or datetime.utcnow()
    if start is None:
        start = end - timedelta(days=7)
    if isinstance(start, datetime) and start.tzinfo is not None:
        start = start.replace(tzinfo=None)
    if isinstance(end, datetime) and end.tzinfo is not None:
        end = end.replace(tzinfo=None)
    return start, end


def _parse_date(s: str | None, *, end_of_day: bool = False) -> datetime | None:
    """Parse YYYY-MM-DD or ISO format; returns naive datetime."""
    if not s:
        return None
    try:
        d = datetime.fromisoformat(s)
    except ValueError:
        return None
    if d.tzinfo is not None:
        d = d.replace(tzinfo=None)
    # If only date was provided (no time component), interpret 'end_date' as end of day.
    if end_of_day and d.hour == 0 and d.minute == 0 and d.second == 0:
        d = d.replace(hour=23, minute=59, second=59)
    return d


def _resolve_window(cur, start_date: str | None, end_date: str | None) -> tuple[datetime, datetime]:
    """Honour user-supplied date range, clamped to the dataset extent."""
    ext_start, ext_end = _data_window(cur)
    s = _parse_date(start_date) or ext_start
    e = _parse_date(end_date, end_of_day=True) or ext_end
    if s < ext_start: s = ext_start
    if e > ext_end:   e = ext_end
    if s > e:         s, e = ext_start, ext_end
    return s, e


def _trend_bucket(start: datetime, end: datetime) -> str:
    days = (end - start).days
    if days <= 60:
        return "day"
    if days <= 365 * 2:
        return "week"
    return "month"


def _classify_status(error_level: str | None, hourly_ratio: float | None = None) -> str:
    """Map external model severity contract to the dashboard's compact status labels."""
    score = MODEL_RUNTIME.severity_score(error_level)
    if MODEL_RUNTIME.is_failure_level(error_level) or (score is not None and score >= 2):
        return "Critical"
    if score == 1:
        return "Warning"
    if score is None and error_level:
        return "Unknown"
    return "Normal"


def _status_code(error_level: str | None, prob: float, forecast: float = 0.0) -> str:
    """DrGb24-style status codes used by the dashboard table & i18n labels.

    FAULTED     = fatal level OR (error level AND prob>=0.8)
    MAINTENANCE = error level OR prob>=0.55
    MONITOR     = warning level OR prob>=0.30
    OPERATIONAL = otherwise
    """
    lvl = (error_level or "").lower()
    if lvl == "fatal" or (lvl == "error" and prob >= 0.8):
        return "FAULTED"
    if lvl == "error" or prob >= 0.55:
        return "MAINTENANCE"
    if lvl == "warning" or prob >= 0.30:
        return "MONITOR"
    return "OPERATIONAL"


def _normalize_severity(error_level: str | None) -> str:
    lvl = (error_level or "").strip().lower()
    if lvl in ("event", "info"):       return "Event"
    if lvl == "warning":               return "Warning"
    if lvl == "error":                 return "Error"
    if lvl in ("critical", "fatal"):   return "Fatal"
    return error_level or "Event"


def _category_for_error_type(error_type: str | None) -> str:
    return MODEL_RUNTIME.category_for_error_type(error_type)


def _failure_levels() -> list[str]:
    return MODEL_RUNTIME.snapshot().failure_levels


def _failure_condition_sql(column: str = "error_level") -> tuple[str, list[str]]:
    levels = _failure_levels()
    if not levels:
        return "FALSE", []
    placeholders = ",".join(["%s"] * len(levels))
    return f"COALESCE({column}, '') IN ({placeholders})", levels


def _category_order_from_types(error_types: list[str | None]) -> list[str]:
    categories = {_category_for_error_type(error_type) for error_type in error_types}
    return sorted(categories)


def _metric_dict(snapshot: RuntimeSnapshot, head_id: str) -> dict[str, Any] | None:
    for metric in snapshot.metrics:
        if metric.id == head_id:
            return metric.__dict__
    return None


def _metric_display(metric: ModelHeadMetric | dict[str, Any] | None) -> str | None:
    if metric is None:
        return None
    if isinstance(metric, dict):
        name = metric.get("metric")
        result = metric.get("result")
    else:
        name = metric.metric
        result = metric.result
    if not name and not result:
        return None
    if not result:
        return str(name)
    if not name:
        return str(result)
    return f"{name} · {result}"


def _pct(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value) * 100, 1)


def _format_hours(hours: float | None) -> str:
    if hours is None:
        return "No failure observed"
    if hours < 24:
        return f"{hours:.1f} saat"
    return f"{hours / 24:.1f} gün"


def _prediction_horizon_hours(snapshot: RuntimeSnapshot, days: int) -> int:
    return max(1, int(days)) * 24


def _bucket_keys(start: datetime, end: datetime, grain: str) -> list[datetime]:
    if grain == "week":
        cursor = datetime(start.year, start.month, start.day) - timedelta(days=start.weekday())
        step = timedelta(days=7)
    else:
        cursor = datetime(start.year, start.month, start.day)
        step = timedelta(days=1)
    last = datetime(end.year, end.month, end.day)
    keys: list[datetime] = []
    while cursor <= last:
        keys.append(cursor)
        cursor += step
    return keys

# ============================================================================
# API: shared
# ============================================================================
@app.get("/api/health")
def api_health() -> dict[str, Any]:
    try:
        with get_cursor() as cur:
            cur.execute("SELECT 1 AS ok;")
            cur.fetchone()
        runtime = MODEL_RUNTIME.snapshot()
        return {
            "ok": True,
            "source": "huggingface",
            "dataset": HF_DATASET_REPO,
            "revision": HF_DATASET_REVISION,
            "model_runtime": runtime.as_dict(),
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/api/filter-options")
def api_filter_options() -> dict[str, Any]:
    with get_cursor() as cur:
        cur.execute("SELECT DISTINCT error_type FROM public.robot_logs_error WHERE error_type IS NOT NULL ORDER BY error_type;")
        ftypes = [r["error_type"] for r in cur.fetchall()]
        cur.execute("SELECT DISTINCT robot_id FROM public.robot_logs_error WHERE robot_id IS NOT NULL ORDER BY robot_id;")
        robots = [r["robot_id"] for r in cur.fetchall()]
        categories = ["All Components"] + _category_order_from_types(ftypes)
        return {
            "robot_statuses": ["All Statuses", "FAULTED", "MAINTENANCE", "MONITOR", "OPERATIONAL"],
            "statuses": ["All Statuses", "Critical", "Warning", "Normal", "Unknown"],
            "fault_types": ["All Fault Types"] + ftypes,
            "robots": ["All Robots"] + robots,
            "categories": categories,
        }


@app.get("/api/model-info")
def api_model_info() -> dict[str, Any]:
    runtime = MODEL_RUNTIME.snapshot()
    with get_cursor() as cur:
        cur.execute("""
            SELECT model_name, status, retrained, dataset_row_count, metrics, created_at
            FROM model_training.training_runs ORDER BY created_at DESC LIMIT 1;
        """)
        r = cur.fetchone()
        if not r:
            return {"runtime": runtime.as_dict(), "heads": [m.__dict__ for m in runtime.metrics]}
        return {
            "runtime": runtime.as_dict(),
            "heads": [m.__dict__ for m in runtime.metrics],
            "model_name": r["model_name"], "status": r["status"], "retrained": r["retrained"],
            "dataset_row_count": r["dataset_row_count"], "metrics": _json_value(r["metrics"]),
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
        }


# ============================================================================
# API: dashboard
# ============================================================================
@app.get("/api/stats")
def api_stats(start_date: str | None = None, end_date: str | None = None) -> dict[str, Any]:
    with get_cursor() as cur:
        start, end = _resolve_window(cur, start_date, end_date)
        midpoint = start + (end - start) / 2

        cur.execute("SELECT COUNT(DISTINCT robot_id) AS n FROM public.robot_logs_error WHERE robot_id IS NOT NULL;")
        total = cur.fetchone()["n"] or 0
        cur.execute("SELECT COUNT(DISTINCT robot_id) AS n FROM public.robot_logs_error WHERE task_time BETWEEN %s AND %s AND robot_id IS NOT NULL;", (start, end))
        active = cur.fetchone()["n"] or 0
        cur.execute("SELECT COUNT(DISTINCT robot_id) AS n FROM public.robot_logs_error WHERE task_time BETWEEN %s AND %s AND robot_id IS NOT NULL;", (start, midpoint))
        active_prev = cur.fetchone()["n"] or 0

        # Critical = robots whose latest state in the window matches the
        # external model's failure-level contract.
        failure_sql, failure_params = _failure_condition_sql("error_level")
        crit_q = f"""
            WITH latest AS (
                SELECT DISTINCT ON (robot_id) robot_id, error_level, hourly_ratio
                FROM public.robot_logs_error
                WHERE task_time BETWEEN %s AND %s AND robot_id IS NOT NULL
                ORDER BY robot_id, task_time DESC
            )
            SELECT COUNT(*) AS n
            FROM latest
            WHERE {failure_sql};
        """
        cur.execute(crit_q, [start, end, *failure_params])
        critical = cur.fetchone()["n"] or 0
        cur.execute(crit_q, [start, midpoint, *failure_params])
        critical_prev = cur.fetchone()["n"] or 0

        fleet = (1 - critical / active) * 100 if active else 0
        fleet_prev = (1 - critical_prev / active_prev) * 100 if active_prev else 0

        def pct(now_v: float, prev_v: float) -> float:
            return 0.0 if prev_v == 0 else round((now_v - prev_v) / prev_v * 100, 1)

        return {
            "range": {"start": start.isoformat(), "end": end.isoformat()},
            "active_robots":   {"value": active,   "total": total, "delta_pct": pct(active, active_prev)},
            "critical_alerts": {"value": critical, "delta_pct": pct(critical, critical_prev)},
            "fleet_health":    {"value": round(fleet, 1), "delta_pct": round(fleet - fleet_prev, 1)},
        }


@app.get("/api/anomaly-trend")
def api_anomaly_trend(
    robot_id: str | None = Query(default=None),
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    with get_cursor() as cur:
        start, end = _resolve_window(cur, start_date, end_date)
        bucket = _trend_bucket(start, end)
        params: list[Any] = [bucket, start, end]
        sql = """
            SELECT date_trunc(%s, task_time) AS bkt,
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
            "range": {"start": start.isoformat(), "end": end.isoformat()},
            "bucket": bucket,
            "points": [
                {"date": r["bkt"].isoformat(), "score": round(float(r["score"] or 0), 1), "samples": int(r["sample_n"] or 0)}
                for r in rows
            ],
        }


@app.get("/api/fault-distribution")
def api_fault_distribution(start_date: str | None = None, end_date: str | None = None) -> dict[str, Any]:
    with get_cursor() as cur:
        start, end = _resolve_window(cur, start_date, end_date)
        cur.execute("""
            SELECT COALESCE(error_type,'Unknown') AS category, COUNT(*) AS cnt
            FROM public.robot_logs_error
            WHERE task_time BETWEEN %s AND %s
            GROUP BY 1 ORDER BY cnt DESC;
        """, (start, end))
        rows = cur.fetchall()
        if not rows:
            return {"total": 0, "items": []}
        total = sum(r["cnt"] for r in rows)
        top, other = rows[:4], rows[4:]
        items = [{"label": r["category"], "count": int(r["cnt"]), "pct": round(r["cnt"] * 100 / total, 1)} for r in top]
        if other:
            on = sum(r["cnt"] for r in other)
            items.append({"label": "Other", "count": int(on), "pct": round(on * 100 / total, 1)})
        return {"total": int(total), "items": items}


@app.get("/api/robots")
def api_robots(
    page: int = Query(1, ge=1),
    page_size: int = Query(5, ge=1, le=100),
    search: str | None = None,
    status: str | None = None,
    fault_type: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    with get_cursor() as cur:
        start, end = _resolve_window(cur, start_date, end_date)

        # Latest log per robot in window (for the "current state" columns)
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
        if fault_type and fault_type.lower() not in ("all", "all fault types"):
            sql += " AND error_type = %s"
            params.append(fault_type)
        cur.execute(sql, params)
        rows = cur.fetchall()

        # Per-robot supplementary aggregates over the same window:
        #   - 7-day forecast = AVG(hourly_ratio) over the most-recent 7 days
        #   - active_errors  = distinct error_type values in the most-recent 30 days
        seven_day_cut = end - timedelta(days=7)
        cur.execute(
            """
            SELECT robot_id, AVG(hourly_ratio) AS r7
            FROM public.robot_logs_error
            WHERE robot_id IS NOT NULL
              AND task_time BETWEEN %s AND %s
            GROUP BY robot_id;
            """,
            (seven_day_cut, end),
        )
        seven_day = {r["robot_id"]: float(r["r7"] or 0) for r in cur.fetchall()}

        active_errs_cut = end - timedelta(days=30)
        cur.execute(
            """
            SELECT robot_id, error_type, COUNT(*) AS cnt
            FROM public.robot_logs_error
            WHERE robot_id IS NOT NULL AND error_type IS NOT NULL
              AND task_time BETWEEN %s AND %s
            GROUP BY robot_id, error_type
            ORDER BY robot_id, cnt DESC;
            """,
            (active_errs_cut, end),
        )
        active_map: dict[str, list[str]] = {}
        for r in cur.fetchall():
            active_map.setdefault(r["robot_id"], [])
            if len(active_map[r["robot_id"]]) < 3:
                active_map[r["robot_id"]].append(r["error_type"])

        out = []
        for r in rows:
            prob = float(r["hourly_ratio"] or 0)
            forecast = seven_day.get(r["robot_id"], prob)
            code = _status_code(r["error_level"], prob, forecast)
            legacy_status = _classify_status(r["error_level"], prob)
            severity = _normalize_severity(r["error_level"])
            if status and status.lower() not in ("all", "all statuses") and status.upper() != code:
                continue
            # Estimated remaining time before predicted failure (synthetic 0-168h)
            est_hours = max(0, round(168 * max(0.0, 1 - max(prob, forecast)), 0))
            out.append({
                "robot_id": r["robot_id"],
                "area": r["product_code"] or "Unknown",
                "status": legacy_status,             # kept for back-compat
                "status_code": code,                 # new: FAULTED/MAINTENANCE/MONITOR/OPERATIONAL
                "severity": severity,                # new: Event/Warning/Error/Fatal
                "predicted_fault": r["error_type"] or "No Fault Detected",
                "predicted_detail": r["error_detail"] or "All Systems Normal",
                "confidence": round(prob * 100, 0),
                "fault_probability": round(prob * 100, 1),
                "seven_day_forecast": round(forecast * 100, 1),
                "estimated_hours": int(est_hours),
                "active_errors": active_map.get(r["robot_id"], []),
                "last_updated": r["task_time"].isoformat() if r["task_time"] else None,
            })

        order = {"FAULTED": 0, "MAINTENANCE": 1, "MONITOR": 2, "OPERATIONAL": 3}
        out.sort(key=lambda x: (order.get(x["status_code"], 9), x["robot_id"]))
        start_idx = (page - 1) * page_size
        return {"total": len(out), "page": page, "page_size": page_size, "items": out[start_idx:start_idx + page_size]}


@app.get("/api/robot/{robot_id}")
def api_robot_detail(robot_id: str) -> dict[str, Any]:
    with get_cursor() as cur:
        cur.execute("""
            SELECT robot_id, product_code, sn, mac, soft_version, hard_version, os_version,
                   error_type, error_detail, error_level, hourly_ratio, task_time
            FROM public.robot_logs_error
            WHERE robot_id = %s
            ORDER BY task_time DESC LIMIT 20;
        """, (robot_id,))
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


# ============================================================================
# API: fault history
# ============================================================================
@app.get("/api/fault-history/frequency")
def api_fault_frequency(start_date: str | None = None, end_date: str | None = None) -> dict[str, Any]:
    with get_cursor() as cur:
        start, end = _resolve_window(cur, start_date, end_date)
        cur.execute("""
            SELECT date_trunc('month', task_time) AS bkt,
                   COALESCE(error_type, 'Unknown') AS etype,
                   COUNT(*) AS cnt
            FROM public.robot_logs_error
            WHERE task_time BETWEEN %s AND %s
            GROUP BY 1, 2 ORDER BY 1;
        """, (start, end))
        # Return month buckets as ISO timestamps so the frontend can format
        # them with the user's chosen locale (Feb / Şub etc).
        agg: dict[str, dict[str, int]] = {}
        order_keys: list[str] = []
        category_totals: dict[str, int] = {}
        for r in cur.fetchall():
            iso = r["bkt"].isoformat()
            if iso not in agg:
                agg[iso] = {}
                order_keys.append(iso)
            cat = _category_for_error_type(r["etype"])
            count = int(r["cnt"])
            agg[iso][cat] = agg[iso].get(cat, 0) + count
            category_totals[cat] = category_totals.get(cat, 0) + count
        labels_iso = order_keys[-6:]
        category_order = sorted(category_totals, key=lambda cat: (-category_totals[cat], cat))
        datasets = [{"label": cat, "data": [agg.get(l, {}).get(cat, 0) for l in labels_iso]} for cat in category_order]
        return {"labels_iso": labels_iso, "datasets": datasets}


@app.get("/api/fault-history/list")
def api_fault_list(
    page: int = Query(1, ge=1),
    page_size: int = Query(8, ge=1, le=100),
    search: str | None = None,
    robot: str | None = None,
    fault_type: str | None = None,
    status: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    with get_cursor() as cur:
        win_start, win_end = _data_window(cur)
        sql = """
            SELECT robot_id, error_id, error_type, error_detail, error_level, hourly_ratio, task_time,
                   hourly_error_count, pair_max_hourly_count
            FROM public.robot_logs_error
            WHERE task_time BETWEEN %s AND %s
        """
        params: list[Any] = [win_start, win_end]
        if start_date:
            try:
                d = datetime.fromisoformat(start_date)
                if d.tzinfo is not None: d = d.replace(tzinfo=None)
                sql += " AND task_time >= %s"; params.append(d)
            except ValueError:
                pass
        if end_date:
            try:
                d = datetime.fromisoformat(end_date) + timedelta(days=1)
                if d.tzinfo is not None: d = d.replace(tzinfo=None)
                sql += " AND task_time < %s"; params.append(d)
            except ValueError:
                pass
        if search:
            # The search box now reads "Search Error ID..." and filters
            # against error_id only.
            sql += " AND COALESCE(error_id,'') ILIKE %s"
            params.append(f"%{search}%")
        if robot and robot.lower() not in ("all", "all robots"):
            sql += " AND robot_id = %s"; params.append(robot)
        if fault_type and fault_type.lower() not in ("all", "all fault types"):
            sql += " AND error_type = %s"; params.append(fault_type)

        sql += " ORDER BY task_time DESC LIMIT 5000;"
        cur.execute(sql, params)
        rows = cur.fetchall()

        items = []
        for r in rows:
            st = _classify_status(r["error_level"], r["hourly_ratio"])
            if status and status.lower() not in ("all", "all statuses") and st.lower() != status.lower():
                continue
            mins = int((r["hourly_error_count"] or 1) * 8)
            downtime = f"{mins // 60}h {mins % 60:02d}m" if mins >= 60 else f"{mins}m"
            tt = r["task_time"]
            if isinstance(tt, datetime) and tt.tzinfo is not None: tt = tt.replace(tzinfo=None)
            age_days = (win_end - tt).days if tt else 0
            resolution = "In Progress" if (st == "Critical" and age_days < 30) else "Resolved"
            items.append({
                "task_time": r["task_time"].isoformat() if r["task_time"] else None,
                "robot_id": r["robot_id"] or "",
                "error_id": r["error_id"] or "",
                "fault_type_raw": r["error_type"] or "Unknown",
                "category": _category_for_error_type(r["error_type"]),
                "diagnosed_issue": r["error_detail"] or r["error_type"] or "Unknown issue",
                "downtime": downtime,
                "resolution": resolution,
                "status": st,
            })

        total = len(items)
        start_idx = (page - 1) * page_size
        return {"total": total, "page": page, "page_size": page_size, "items": items[start_idx:start_idx + page_size]}


# ============================================================================
# API: predictions
# ============================================================================
@app.get("/api/predictions/heatmap")
def api_pred_heatmap(
    weeks: int = Query(8, ge=2, le=52),
    days: int | None = Query(default=None, ge=1, le=730),
    robot_id: str | None = Query(default=None),
) -> dict[str, Any]:
    """Per-robot risk grid.

    Always renders multiple time buckets (daily when the window is <=14
    days, weekly otherwise) so the heatmap shows columns instead of one
    full-width bar. Engine-available mode is used only to score WHICH
    robots show up in the grid; the cell colors come from the historical
    fault-risk bucketed query below.
    """
    with get_cursor() as cur:
        snapshot = MODEL_RUNTIME.snapshot()
        horizon_days = days if days is not None else weeks * 7
        horizon_hours = _prediction_horizon_hours(snapshot, horizon_days)
        robot = robot_id if robot_id and robot_id.lower() not in ("", "all", "all robots") else None

        # When the live engine is available, use its ranking to pick the top-N
        # robots; otherwise the fallback _head_reference_rows query does the
        # same. Either way we always run the bucketed historical risk query
        # below so the grid has visible columns.
        chosen_robot_ids: list[str] = []
        if snapshot.engine_available:
            items, _meta = _head_rows(cur, None, None, horizon_hours, robot)
            if items:
                ranked = items if robot else sorted(
                    items,
                    key=lambda item: (
                        -(item["head_3"]["next_7d_fail_prob"] or 0),
                        item["robot_id"],
                    ),
                )
                chosen_robot_ids = [item["robot_id"] for item in ranked][:8 if not robot else None]

        reference_rows, _start, reference = _head_reference_rows(cur, None, None, horizon_hours, robot)
        robot_ids = [r["robot_id"] for r in reference_rows if r["robot_id"]]
        if not robot_ids:
            return {"robot_ids": [], "weeks": [], "grid": []}

        # If engine-mode produced a ranking, prefer that list and keep
        # the fallback robot_ids only as a top-up.
        if chosen_robot_ids:
            robot_ids = chosen_robot_ids

        # Look BACKWARDS from the reference time. Historical snapshots have
        # no future data, so forward-looking buckets were always empty,
        # which is exactly why the previous heatmap collapsed to a single
        # bar per robot. Past N days produces a real bucketed risk grid.
        observed_start = reference - timedelta(hours=horizon_hours)
        grain = "day" if horizon_hours <= 14 * 24 else "week"
        failure_sql, failure_params = _failure_condition_sql("error_level")
        placeholders = ",".join(["%s"] * len(robot_ids))
        cur.execute(
            f"""
            SELECT robot_id,
                   date_trunc('{grain}', task_time) AS bkt,
                   COUNT(*) AS events,
                   MAX(hourly_ratio) AS risk
            FROM public.robot_logs_error
            WHERE task_time >= %s
              AND task_time <= %s
              AND robot_id IN ({placeholders})
              AND ({failure_sql})
            GROUP BY robot_id, bkt
            ORDER BY robot_id, bkt;
            """,
            [observed_start, reference, *robot_ids, *failure_params],
        )
        rows = cur.fetchall()
        bucket_keys = _bucket_keys(observed_start, reference, grain)
        robots: dict[str, dict[datetime, float]] = {rid: {b: 0.0 for b in bucket_keys} for rid in robot_ids}
        for r in rows:
            bucket = r["bkt"]
            if bucket in robots.get(r["robot_id"], {}):
                robots[r["robot_id"]][bucket] = round(float(r["risk"] or 0) * 100, 1)
        bucket_labels = [bucket.strftime("%b %d") for bucket in bucket_keys]
        if robot:
            active_robots = [(rid, robots[rid]) for rid in robot_ids]
        else:
            active_robots = sorted(robots.items(), key=lambda kv: -sum(kv[1].values()))[:8]
        return {
            "robot_ids": [rid for rid, _ in active_robots],
            "weeks": bucket_labels,
            "grid": [[active_robots[i][1].get(bucket, 0) for bucket in bucket_keys] for i in range(len(active_robots))],
            "source": "historical_lookback",
            "reference_time": reference.isoformat(),
            "horizon_hours": horizon_hours,
        }


@app.get("/api/predictions/degradation")
def api_pred_degradation(
    category: str | None = None,
    robot_id: str | None = Query(default=None),
    days: int = Query(7, ge=1, le=365),
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    with get_cursor() as cur:
        snapshot = MODEL_RUNTIME.snapshot()
        horizon_hours = _prediction_horizon_hours(snapshot, days)
        ext_start, ext_end = _data_window(cur)
        reference = _parse_date(end_date, end_of_day=True)
        if reference is None:
            reference = ext_end if snapshot.engine_available else max(ext_start, ext_end - timedelta(hours=horizon_hours))
        if reference > ext_end:
            reference = ext_end
        start = _parse_date(start_date) or max(ext_start, reference - timedelta(days=days))
        if start < ext_start:
            start = ext_start
        params: list[Any] = [start, reference]
        cur.execute("""
            SELECT date_trunc('day', task_time) AS bkt,
                   robot_id,
                   COALESCE(error_type,'') AS etype,
                   error_level,
                   AVG(hourly_ratio) AS risk
            FROM public.robot_logs_error
            WHERE task_time BETWEEN %s AND %s
            GROUP BY 1, 2, 3, 4 ORDER BY 1;
        """, params)
        trend_rows = cur.fetchall()

        by_week: dict[datetime, list[float]] = {}
        active_robots: set[str] = set()
        for r in trend_rows:
            if robot_id and robot_id.lower() not in ("", "all", "all robots") and r["robot_id"] != robot_id:
                continue
            if category and category.lower() not in ("", "all", "all components") and _category_for_error_type(r["etype"]) != category:
                continue
            active_robots.add(r["robot_id"])
            by_week.setdefault(r["bkt"], []).append(float(r["risk"] or 0))

        labels, actual = [], []
        for k in sorted(by_week.keys()):
            avg = sum(by_week[k]) / len(by_week[k])
            labels.append(k.strftime("%b %d"))
            actual.append(round(avg * 100, 1))

        observed_end = min(reference + timedelta(hours=horizon_hours), ext_end)
        future_labels: list[str] = []
        future_values: list[float] = []
        if snapshot.engine_available:
            robot = robot_id if robot_id and robot_id.lower() not in ("", "all", "all robots") else None
            head_rows, _meta = _head_rows(cur, None, None, horizon_hours, robot)
            probabilities = []
            for row in head_rows:
                if category and category.lower() not in ("", "all", "all components") and row["component"] != category:
                    continue
                probabilities.append(float(row["head_3"]["next_7d_fail_prob"] or 0))
            if probabilities:
                future_labels.append(f"Next {days}d")
                future_values.append(round(sum(probabilities) / len(probabilities), 1))
        elif observed_end > reference:
            failure_sql, failure_params = _failure_condition_sql("error_level")
            future_params: list[Any] = [reference, observed_end, *failure_params]
            cur.execute(
                f"""
                SELECT date_trunc('day', task_time) AS bkt,
                       robot_id,
                       COALESCE(error_type,'') AS etype,
                       COUNT(*) AS events
                FROM public.robot_logs_error
                WHERE task_time > %s
                  AND task_time <= %s
                  AND ({failure_sql})
                GROUP BY 1, 2, 3 ORDER BY 1;
                """,
                future_params,
            )
            future_by_day: dict[datetime, set[str]] = {}
            for r in cur.fetchall():
                if robot_id and robot_id.lower() not in ("", "all", "all robots") and r["robot_id"] != robot_id:
                    continue
                if category and category.lower() not in ("", "all", "all components") and _category_for_error_type(r["etype"]) != category:
                    continue
                future_by_day.setdefault(r["bkt"], set()).add(r["robot_id"])
            denominator = 1 if robot_id and robot_id.lower() not in ("", "all", "all robots") else max(1, len(active_robots))
            for day in sorted(future_by_day.keys()):
                future_labels.append(day.strftime("%b %d"))
                future_values.append(round(len(future_by_day[day]) * 100 / denominator, 1))

        labels = labels[-14:]
        actual = actual[-14:]
        lstm_pred = [None] * len(actual) + future_values
        rf_pred = list(lstm_pred)

        return {
            "labels": labels + future_labels,
            "actual": actual + [None] * len(future_labels),
            "lstm_pred": lstm_pred,
            "rf_pred": rf_pred,
            "predicted_failure_label": next((label for label, value in zip(future_labels, future_values) if value > 0), None),
        }


@app.get("/api/predictions/stats")
def api_pred_stats(
    days: int = Query(7, ge=1, le=365),
    robot_id: str | None = Query(default=None),
    category: str | None = Query(default=None),
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    with get_cursor() as cur:
        snapshot = MODEL_RUNTIME.snapshot()
        horizon_hours = _prediction_horizon_hours(snapshot, days)
        robot = robot_id if robot_id and robot_id.lower() not in ("", "all", "all robots") else None
        items, meta = _head_rows(cur, start_date, end_date, horizon_hours, robot)
        # Apply the top-bar Category filter on the head outputs. Each item
        # already carries a "component" derived via _category_for_error_type,
        # so post-filtering here keeps both KPIs (High Severity + Predicted
        # Failures) and the Model Accuracy summary in sync.
        if category and category.lower() not in ("", "all", "all components"):
            items = [it for it in items if it.get("component") == category]
        total = len(items)
        current_failures = sum(1 for item in items if item["head_1"]["is_failure_now"])
        future_failures = sum(1 for item in items if item["head_3"]["future_failure_observed"])
        high_severity = sum(1 for item in items if (item["head_2"]["severity_score"] or 0) >= 2)
        eta_values = [
            item["head_4"]["est_hours_to_failure"]
            for item in items
            if item["head_4"]["est_hours_to_failure"] is not None
        ]
        fleet_health = round((1 - current_failures / total) * 100, 1) if total else 0.0

        metrics = {metric.id: metric for metric in snapshot.metrics}
        average_eta = round(sum(eta_values) / len(eta_values), 1) if eta_values else None
        model_heads = {
            "1": {
                "value": current_failures,
                "unit": "robots",
                "bar_pct": round(current_failures * 100 / total, 1) if total else 0,
                "label_key": "head1Label",
                "metric_text": _metric_display(metrics.get("head_1")),
            },
            "2": {
                "value": high_severity,
                "unit": "robots",
                "bar_pct": round(high_severity * 100 / total, 1) if total else 0,
                "label_key": "head2Label",
                "metric_text": _metric_display(metrics.get("head_2")),
            },
            "3": {
                "value": future_failures,
                "unit": "robots",
                "bar_pct": round(future_failures * 100 / total, 1) if total else 0,
                "label_key": "head3Label",
                "metric_text": _metric_display(metrics.get("head_3")),
            },
            "4": {
                "value": average_eta,
                "unit": "h",
                "bar_pct": 0 if average_eta is None else max(0, round(100 - min(100, average_eta / max(1, horizon_hours) * 100), 1)),
                "label_key": "head4Label",
                "metric_text": _metric_display(metrics.get("head_4")),
            },
        }

        return {
            **meta,
            "fleet_health":   {"value": fleet_health, "delta_pct": None},
            "high_risk":      {"value": high_severity, "delta_pct": None},
            "predicted_fail": {"value": future_failures, "delta_pct": None},
            "model_heads":    model_heads,
        }


@app.get("/api/notifications")
def api_notifications(limit: int = Query(10, ge=1, le=50)) -> dict[str, Any]:
    """Recent high-severity events as notifications for the bell dropdown."""
    with get_cursor() as cur:
        _start, end = _data_window(cur)
        failure_sql, failure_params = _failure_condition_sql()
        cur.execute(f"""
            SELECT robot_id, error_type, error_detail, error_level, hourly_ratio, task_time
            FROM public.robot_logs_error
            WHERE robot_id IS NOT NULL
              AND ({failure_sql})
            ORDER BY task_time DESC
            LIMIT %s;
        """, [*failure_params, limit])
        rows = cur.fetchall()
        items = []
        for r in rows:
            tt = r["task_time"]
            if isinstance(tt, datetime) and tt.tzinfo is not None:
                tt = tt.replace(tzinfo=None)
            score = MODEL_RUNTIME.severity_score(r["error_level"])
            severity = "critical" if score is not None and score >= 2 else "warning"
            items.append({
                "robot_id": r["robot_id"],
                "title": r["error_type"] or "Unknown fault",
                "detail": r["error_detail"] or "",
                "level": r["error_level"] or "",
                "severity": severity,
                "ratio": round(float(r["hourly_ratio"] or 0) * 100, 0),
                "task_time": tt.isoformat() if tt else None,
            })
        return {"unread": len(items), "items": items}


@app.get("/api/predictions/top-failures")
def api_top_failures(
    days: int = Query(30, ge=1, le=365),
    robot_id: str | None = Query(default=None),
    category: str | None = Query(default=None),
    head: str = Query(default="3"),
    limit: int = Query(50, ge=1, le=500),
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    with get_cursor() as cur:
        snapshot = MODEL_RUNTIME.snapshot()
        horizon_hours = _prediction_horizon_hours(snapshot, days)
        robot = robot_id if robot_id and robot_id.lower() not in ("", "all", "all robots") else None
        rows, meta = _head_rows(cur, start_date, end_date, horizon_hours, robot)
        selected_head = head if head in {"1", "2", "3", "4"} else "3"
        items: list[dict[str, Any]] = []
        for row in rows:
            if category and category.lower() not in ("", "all", "all components") and row["component"] != category:
                continue
            severity_score = row["head_2"]["severity_score"]
            eta = row["head_4"]["est_hours_to_failure"]
            if selected_head == "1":
                value = row["head_1"]["failure_prob_now"] or 0
                unit = "%"
                risk = "Critical" if row["head_1"]["is_failure_now"] else "Low"
                sort_key = (not row["head_1"]["is_failure_now"], -value, row["robot_id"])
                value_label = "Head 1"
                estimate_label = row["last_observed_at"]
            elif selected_head == "2":
                value = severity_score
                unit = "/3"
                if severity_score is None:
                    risk = "Low"
                elif severity_score >= 3:
                    risk = "Critical"
                elif severity_score >= 2:
                    risk = "High"
                elif severity_score == 1:
                    risk = "Medium"
                else:
                    risk = "Low"
                sort_key = (-(severity_score if severity_score is not None else -1), row["robot_id"])
                value_label = row["head_2"]["severity_now_tr"]
                estimate_label = row["last_observed_at"]
            elif selected_head == "4":
                value = eta
                unit = "h"
                risk = "High" if eta is not None else "Low"
                sort_key = (eta is None, eta if eta is not None else 10**9, row["robot_id"])
                value_label = "Head 4"
                estimate_label = row["head_4"]["est_time_label"]
            else:
                value = row["head_3"]["next_7d_fail_prob"] or 0
                unit = "%"
                risk = "High" if row["head_3"]["future_failure_observed"] else "Low"
                sort_key = (not row["head_3"]["future_failure_observed"], -value, eta if eta is not None else 10**9, row["robot_id"])
                value_label = "Head 3"
                estimate_label = row["head_4"]["est_time_label"]
            items.append({
                "robot_id": row["robot_id"],
                "area": row["area"],
                "value": value,
                "unit": unit,
                "value_label": value_label,
                "failure_probability": value if unit == "%" else None,
                "risk_level": risk,
                "predicted_issue": row["error_type"],
                "predicted_detail": row["error_detail"] or "Operational",
                "estimated_time": row["last_observed_at"],
                "estimated_time_label": estimate_label,
                "category": row["component"],
                "head": selected_head,
                "source": row[f"head_{selected_head}"]["source"],
                "_sort": sort_key,
            })
        items.sort(key=lambda x: x["_sort"])
        for item in items:
            item.pop("_sort", None)
        return {**meta, "head": selected_head, "items": items[:limit]}


# ============================================================================
# API: model-head dashboard
# ============================================================================
def _head_reference_rows(
    cur,
    start_date: str | None,
    end_date: str | None,
    horizon_hours: int,
    robot: str | None = None,
) -> tuple[list[dict[str, Any]], datetime, datetime]:
    ext_start, ext_end = _data_window(cur)
    start = _parse_date(start_date) or ext_start
    reference = _parse_date(end_date, end_of_day=True)
    if reference is None:
        snapshot = MODEL_RUNTIME.snapshot()
        if snapshot.engine_available:
            reference = ext_end
        else:
            reference = max(ext_start, ext_end - timedelta(hours=horizon_hours))
    if start < ext_start:
        start = ext_start
    if reference > ext_end:
        reference = ext_end
    if start > reference:
        start = ext_start
    sql = """
        SELECT DISTINCT ON (robot_id)
               robot_id, product_code, error_type, error_detail, error_level,
               hourly_ratio, hourly_error_count, pair_max_hourly_count, task_time
        FROM public.robot_logs_error
        WHERE task_time BETWEEN %s AND %s AND robot_id IS NOT NULL
    """
    params: list[Any] = [start, reference]
    if robot and robot.lower() not in ("all", "all robots"):
        sql += " AND robot_id = %s"
        params.append(robot)
    sql += " ORDER BY robot_id, task_time DESC;"
    cur.execute(sql, params)
    return cur.fetchall(), start, reference


def _future_failures_by_robot(
    cur,
    reference: datetime,
    horizon_hours: int,
    robot_ids: list[str],
) -> tuple[dict[str, dict[str, Any]], datetime, bool]:
    if not robot_ids:
        return {}, reference, False
    _data_start, data_end = _data_window(cur)
    requested_end = reference + timedelta(hours=horizon_hours)
    observed_end = min(requested_end, data_end)
    if observed_end <= reference:
        return {}, observed_end, False

    failure_sql, failure_params = _failure_condition_sql("error_level")
    placeholders = ",".join(["%s"] * len(robot_ids))
    cur.execute(
        f"""
        SELECT robot_id,
               MIN(task_time) AS next_failure_time,
               COUNT(*) AS future_failure_events,
               MAX(hourly_ratio) AS max_future_ratio
        FROM public.robot_logs_error
        WHERE task_time > %s
          AND task_time <= %s
          AND robot_id IN ({placeholders})
          AND ({failure_sql})
        GROUP BY robot_id;
        """,
        [reference, observed_end, *robot_ids, *failure_params],
    )
    rows = {r["robot_id"]: r for r in cur.fetchall()}
    return rows, observed_end, requested_end <= data_end


def _engine_history_by_robot(
    cur,
    start: datetime,
    reference: datetime,
    robot_ids: list[str],
) -> dict[str, list[dict[str, Any]]]:
    if not robot_ids:
        return {}
    history_limit = _model_history_rows_limit()
    placeholders = ",".join(["%s"] * len(robot_ids))
    cur.execute(
        f"""
        WITH ranked AS (
            SELECT robot_id, product_code, soft_version, error_type, error_level,
                   hourly_ratio, hourly_error_count, task_time,
                   date_trunc('hour', task_time) AS task_hour,
                   ROW_NUMBER() OVER (PARTITION BY robot_id ORDER BY task_time DESC) AS rn
            FROM public.robot_logs_error
            WHERE task_time BETWEEN %s AND %s
              AND robot_id IN ({placeholders})
        )
        SELECT robot_id, product_code, soft_version, error_type, error_level,
               hourly_ratio, hourly_error_count, task_time, task_hour
        FROM ranked
        WHERE rn <= %s
        ORDER BY robot_id, task_time;
        """,
        [start, reference, *robot_ids, history_limit],
    )
    by_robot: dict[str, list[dict[str, Any]]] = {}
    for row in cur.fetchall():
        by_robot.setdefault(row["robot_id"], []).append(row)
    return by_robot


def _model_history_rows_limit() -> int:
    try:
        return max(10, min(240, int(os.getenv("PUDU_MODEL_HISTORY_ROWS", "24"))))
    except ValueError:
        return 24


def _model_source(snapshot: RuntimeSnapshot) -> str:
    return snapshot.engine_kind if snapshot.engine_available and snapshot.engine_kind else "dataset_target_replay"


def _head_rows(
    cur,
    start_date: str | None,
    end_date: str | None,
    horizon_hours: int,
    robot: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    snapshot = MODEL_RUNTIME.snapshot()
    rows, start, reference = _head_reference_rows(cur, start_date, end_date, horizon_hours, robot)
    robot_ids = [r["robot_id"] for r in rows if r["robot_id"]]
    future_by_robot, observed_end, complete = _future_failures_by_robot(cur, reference, horizon_hours, robot_ids)
    engine_predictions: dict[str, dict[str, Any]] = {}
    if snapshot.engine_available:
        histories = _engine_history_by_robot(cur, start, reference, robot_ids)
        for rid, history in histories.items():
            prediction = MODEL_RUNTIME.predict_for_robot(rid, history, reference, horizon_hours=horizon_hours)
            if prediction:
                engine_predictions[rid] = prediction

    items: list[dict[str, Any]] = []
    for r in rows:
        robot_id = r["robot_id"]
        future = future_by_robot.get(robot_id)
        next_time = future["next_failure_time"] if future else None
        hours_to_failure = None
        if next_time:
            hours_to_failure = max(0.0, round((next_time - reference).total_seconds() / 3600, 1))

        prediction = engine_predictions.get(robot_id)
        if prediction:
            severity_score = prediction.get("severity_score")
            severity_label = prediction.get("severity_now") or "Unknown"
            severity_tr = prediction.get("severity_now_tr") or severity_label
            current_failure = bool(prediction.get("is_failure_now"))
            evidence_ratio = float(prediction.get("failure_prob_now") or 0)
            future_supported = prediction.get("future_horizon_supported")
            if future_supported is None:
                supported = set(snapshot.supported_horizon_hours or [])
                future_supported = not supported or horizon_hours in supported
            if future_supported:
                predicted_future = prediction.get("future_failure_prob")
                if predicted_future is None:
                    predicted_future = prediction.get("next_7d_fail_prob")
                future_ratio = float(predicted_future or 0)
                future_failure = future_ratio >= MODEL_BINARY_THRESHOLD
                hours_to_failure = prediction.get("est_hours_to_failure")
                future_source = snapshot.engine_kind or "model_inference"
            else:
                future_failure = bool(future)
                future_ratio = 1.0 if future_failure else 0.0
                future_source = "dataset_target_replay"
            active_error_types = prediction.get("active_error_types") or []
            error_details = prediction.get("error_details") or []
            source = snapshot.engine_kind or "model_inference"
        else:
            severity_score = MODEL_RUNTIME.severity_score(r["error_level"])
            severity_label = snapshot.severity_labels.get(severity_score, r["error_level"] or "Unknown") if severity_score is not None else (r["error_level"] or "Unknown")
            severity_tr = snapshot.severity_labels_tr.get(severity_score, severity_label) if severity_score is not None else severity_label
            current_failure = MODEL_RUNTIME.is_failure_level(r["error_level"])
            future_failure = bool(future)
            evidence_ratio = float(r["hourly_ratio"] or 0)
            future_ratio = 1.0 if future_failure else 0.0
            hours_to_failure = hours_to_failure
            active_error_types = [r["error_type"]] if r["error_type"] else []
            error_details = []
            source = _model_source(snapshot)
            future_source = source

        items.append({
            "robot_id": robot_id,
            "area": r["product_code"] or "Unknown",
            "last_observed_at": r["task_time"].isoformat() if r["task_time"] else None,
            "error_type": r["error_type"] or "Unknown",
            "error_detail": r["error_detail"] or "",
            "component": _category_for_error_type(r["error_type"]),
            "status": _classify_status(r["error_level"], r["hourly_ratio"]),
            "active_error_types": active_error_types,
            "error_details": error_details,
            "head_1": {
                "name": "Anlık arıza",
                "is_failure_now": current_failure,
                "failure_prob_now": _pct(evidence_ratio),
                "source": source,
            },
            "head_2": {
                "name": "Şiddet",
                "severity_now": severity_label,
                "severity_now_tr": severity_tr,
                "severity_score": severity_score,
                "source": source,
            },
            "head_3": {
                "name": "Gelecek öngörü",
                "future_failure_observed": future_failure,
                "future_failure_events": int(future["future_failure_events"]) if future else 0,
                "next_7d_fail_prob": _pct(future_ratio),
                "source": future_source,
            },
            "head_4": {
                "name": "Arıza süresi",
                "est_hours_to_failure": hours_to_failure,
                "est_time_label": _format_hours(hours_to_failure),
                "source": future_source,
            },
        })

    items.sort(
        key=lambda x: (
            not x["head_1"]["is_failure_now"],
            not x["head_3"]["future_failure_observed"],
            x["head_4"]["est_hours_to_failure"] if x["head_4"]["est_hours_to_failure"] is not None else 10**9,
            -(x["head_2"]["severity_score"] or -1),
            x["robot_id"],
        )
    )

    meta = {
        "range": {"start": start.isoformat(), "end": reference.isoformat()},
        "reference_time": reference.isoformat(),
        "horizon_hours": horizon_hours,
        "observed_horizon_end": observed_end.isoformat(),
        "future_window_complete": complete,
        "source": _model_source(snapshot),
        "runtime": snapshot.as_dict(),
    }
    return items, meta


@app.get("/api/model-runtime")
def api_model_runtime() -> dict[str, Any]:
    return MODEL_RUNTIME.snapshot().as_dict()


@app.get("/api/model-heads/summary")
def api_model_heads_summary(
    start_date: str | None = None,
    end_date: str | None = None,
    horizon_days: int = Query(7, ge=1, le=30),
) -> dict[str, Any]:
    with get_cursor() as cur:
        snapshot = MODEL_RUNTIME.snapshot()
        horizon_hours = _prediction_horizon_hours(snapshot, horizon_days)
        items, meta = _head_rows(cur, start_date, end_date, horizon_hours)
        total = len(items)
        current_failures = sum(1 for item in items if item["head_1"]["is_failure_now"])
        future_failures = sum(1 for item in items if item["head_3"]["future_failure_observed"])
        hours = [item["head_4"]["est_hours_to_failure"] for item in items if item["head_4"]["est_hours_to_failure"] is not None]
        severity_counts: dict[str, int] = {}
        component_counts: dict[str, int] = {}
        for item in items:
            sev = item["head_2"]["severity_now_tr"]
            comp = item["component"]
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
            component_counts[comp] = component_counts.get(comp, 0) + 1

        return {
            **meta,
            "total_robots": total,
            "heads": [
                {
                    "id": "head_1",
                    "name": "Anlık arıza",
                    "metric": _metric_dict(snapshot, "head_1"),
                    "value": current_failures,
                    "unit": "robots",
                    "detail": f"{current_failures} / {total} robot",
                },
                {
                    "id": "head_2",
                    "name": "Şiddet",
                    "metric": _metric_dict(snapshot, "head_2"),
                    "value": severity_counts,
                    "unit": "distribution",
                    "detail": max(severity_counts, key=severity_counts.get) if severity_counts else "No data",
                },
                {
                    "id": "head_3",
                    "name": "Gelecek öngörü",
                    "metric": _metric_dict(snapshot, "head_3"),
                    "value": future_failures,
                    "unit": "robots",
                    "detail": f"{future_failures} robot / {horizon_hours // 24} gün",
                },
                {
                    "id": "head_4",
                    "name": "Arıza süresi",
                    "metric": _metric_dict(snapshot, "head_4"),
                    "value": round(sum(hours) / len(hours), 1) if hours else None,
                    "unit": "hours",
                    "detail": _format_hours(round(sum(hours) / len(hours), 1) if hours else None),
                },
            ],
            "severity_counts": severity_counts,
            "component_counts": component_counts,
        }


@app.get("/api/model-heads/robots")
def api_model_head_robots(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    robot: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    horizon_days: int = Query(7, ge=1, le=30),
) -> dict[str, Any]:
    with get_cursor() as cur:
        snapshot = MODEL_RUNTIME.snapshot()
        horizon_hours = _prediction_horizon_hours(snapshot, horizon_days)
        items, meta = _head_rows(cur, start_date, end_date, horizon_hours, robot)
        total = len(items)
        start_idx = (page - 1) * page_size
        return {
            **meta,
            "total": total,
            "page": page,
            "page_size": page_size,
            "items": items[start_idx:start_idx + page_size],
        }


@app.get("/api/model-heads/timeline")
def api_model_head_timeline(
    start_date: str | None = None,
    end_date: str | None = None,
    horizon_days: int = Query(7, ge=1, le=30),
) -> dict[str, Any]:
    with get_cursor() as cur:
        snapshot = MODEL_RUNTIME.snapshot()
        horizon_hours = _prediction_horizon_hours(snapshot, horizon_days)
        _rows, _start, reference = _head_reference_rows(cur, start_date, end_date, horizon_hours)
        _data_start, data_end = _data_window(cur)
        requested_end = reference + timedelta(hours=horizon_hours)
        observed_end = min(requested_end, data_end)
        if observed_end <= reference:
            return {
                "reference_time": reference.isoformat(),
                "horizon_hours": horizon_hours,
                "observed_horizon_end": observed_end.isoformat(),
                "future_window_complete": False,
                "points": [],
            }

        failure_sql, failure_params = _failure_condition_sql("error_level")
        cur.execute(
            f"""
            SELECT date_trunc('day', task_time) AS bkt,
                   COUNT(DISTINCT robot_id) AS robots,
                   COUNT(*) AS events
            FROM public.robot_logs_error
            WHERE task_time > %s
              AND task_time <= %s
              AND ({failure_sql})
            GROUP BY 1 ORDER BY 1;
            """,
            [reference, observed_end, *failure_params],
        )
        return {
            "reference_time": reference.isoformat(),
            "horizon_hours": horizon_hours,
            "observed_horizon_end": observed_end.isoformat(),
            "future_window_complete": requested_end <= data_end,
            "source": _model_source(snapshot),
            "points": [
                {
                    "date": r["bkt"].isoformat(),
                    "robots": int(r["robots"] or 0),
                    "events": int(r["events"] or 0),
                }
                for r in cur.fetchall()
            ],
        }


# ============================================================================
# Frontend
# ============================================================================
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>PUDU LSTM V2 Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root{
  --bg:#f4f6f8;--panel:#fff;--panel-2:#f9fafb;--line:#dfe5ec;--text:#172033;--muted:#647084;
  --ink:#0e1726;--blue:#2563eb;--teal:#0f766e;--green:#16a34a;--amber:#d97706;--red:#dc2626;
  --blue-soft:#dbeafe;--teal-soft:#ccfbf1;--green-soft:#dcfce7;--amber-soft:#fef3c7;--red-soft:#fee2e2;
  --purple:#7c3aed;--purple-soft:#ede9fe;--card:#fff;--border:#dfe5ec;--text-mute:#647084;
  --primary:#2563eb;--primary-2:#1d4ed8;--radius:8px;
  --sidebar-bg:#0f172a;--sidebar-bg-2:#111827;--sidebar-fg:#cbd5e1;--sidebar-fg-mute:#94a3b8;--sidebar-active:#2563eb;
  --cat-brush:#3b82f6;--cat-battery:#10b981;--cat-nav:#f59e0b;--cat-vacuum:#a855f7;--cat-other:#94a3b8;
  --shadow:0 1px 2px rgba(15,23,42,.05),0 8px 24px rgba(15,23,42,.06);--r:8px;
}
*{box-sizing:border-box}html,body{margin:0;padding:0;overflow-x:hidden}
body{font-family:'Inter',-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;
  background:var(--bg);color:var(--text);font-size:14px;-webkit-font-smoothing:antialiased}
a{color:inherit;text-decoration:none}button{font-family:inherit;cursor:pointer}

.app{display:flex;min-height:100vh;max-width:100vw;overflow-x:hidden}
.sidebar{width:248px;background:linear-gradient(180deg,var(--sidebar-bg),var(--sidebar-bg-2));
  color:var(--sidebar-fg);display:flex;flex-direction:column;justify-content:space-between;
  padding:18px 14px;position:sticky;top:0;bottom:0;overflow-y:auto;flex-shrink:0}
.brand{display:flex;align-items:center;gap:10px;padding:6px 8px 18px;
  border-bottom:1px solid rgba(255,255,255,.06);margin-bottom:12px}
.brand-logo{width:38px;height:38px;border-radius:10px;
  background:linear-gradient(135deg,#3b82f6,#60a5fa);display:grid;place-items:center;color:#fff}
.brand-title{font-weight:700;font-size:16px;color:#fff}
.brand-sub{font-size:11px;color:var(--sidebar-fg-mute)}
.nav{display:flex;flex-direction:column;gap:2px}
.nav-item{display:flex;align-items:center;gap:12px;padding:10px 12px;border-radius:8px;
  color:var(--sidebar-fg);font-size:13.5px;transition:background .15s;cursor:pointer;
  border:none;background:transparent;width:100%;text-align:left}
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
	.runtime-strip{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-bottom:18px}
	.runtime-strip:empty{display:none}
	.runtime-item{background:#fff;border:1px solid var(--border);border-radius:8px;padding:10px 12px;min-width:0}
	.runtime-item .k{font-size:10.5px;color:var(--text-mute);text-transform:uppercase;letter-spacing:.04em;font-weight:700}
	.runtime-item .v{font-size:12.5px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
	.runtime-item.warn{border-color:#facc15;background:#fffbeb}
	.runtime-item.good{border-color:#86efac;background:#f0fdf4}
	.runtime-item .code{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace}
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
.date-picker{background:#fff;border:1px solid var(--border);padding:8px 14px;
  border-radius:8px;font-size:13px;cursor:pointer;font-family:inherit;color:inherit}
.date-picker:hover{background:#f9fafc}
.window-pill{display:inline-flex;align-items:center;gap:8px;background:#fff;border:1px solid var(--border);
  border-radius:999px;padding:7px 12px;color:var(--text);box-shadow:var(--shadow)}
.window-pill span{font-size:11px;color:var(--text-mute);font-weight:700;text-transform:uppercase;letter-spacing:.04em}
.window-pill strong{font-size:12.5px;font-weight:700;color:var(--primary);white-space:nowrap}
.bell{cursor:pointer}
.bell:hover{background:#f9fafc}
.topbar-right{position:relative}

/* popovers */
.popover{position:absolute;top:48px;right:0;background:#fff;border:1px solid var(--border);
  border-radius:12px;box-shadow:0 10px 30px rgba(15,23,42,.18);
  z-index:60;padding:14px;min-width:260px}
.popover[hidden]{display:none}
.popover h4{margin:0 0 10px;font-size:13px;font-weight:700}
.popover .row{display:flex;flex-direction:column;gap:4px;margin-bottom:10px}
.popover .row label{font-size:11px;color:var(--text-mute);font-weight:600;text-transform:uppercase;letter-spacing:.04em}
.popover .row input{border:1px solid var(--border);padding:8px 10px;border-radius:8px;font-size:13px;
  font-family:inherit;color:var(--text);background:#fff}
.popover .quick{display:grid;grid-template-columns:repeat(2,1fr);gap:6px;margin-bottom:10px}
.popover .quick button{font-size:11.5px;padding:6px 8px;border-radius:7px;border:1px solid var(--border);
  background:#f9fafc;color:var(--text);cursor:pointer}
.popover .quick button:hover{background:#eef2f7}
.popover .future-range{display:none}
.popover .actions{display:flex;gap:6px;justify-content:flex-end}
.popover .actions button{padding:6px 14px;border-radius:8px;font-size:12.5px;font-weight:600;cursor:pointer;
  border:1px solid var(--border);background:#fff;color:var(--text)}
.popover .actions .primary{background:var(--primary);color:#fff;border-color:var(--primary)}
.popover .actions .primary:hover{background:var(--primary-2)}

.notif-panel{position:absolute;top:48px;right:64px;background:#fff;border:1px solid var(--border);
  border-radius:12px;box-shadow:0 10px 30px rgba(15,23,42,.18);
  z-index:60;width:340px;max-width:calc(100vw - 32px);max-height:420px;display:flex;flex-direction:column}
.notif-panel[hidden]{display:none}
.notif-head{padding:12px 14px;border-bottom:1px solid var(--border);
  display:flex;justify-content:space-between;align-items:center}
.notif-head h4{margin:0;font-size:14px;font-weight:700}
.notif-head .clear{background:transparent;border:none;color:var(--primary);font-size:12px;cursor:pointer}
.notif-list{list-style:none;margin:0;padding:0;overflow:auto;flex:1}
.notif-list li{padding:12px 14px;border-bottom:1px solid var(--border);
  display:grid;grid-template-columns:6px 1fr auto;gap:10px;align-items:start}
.notif-list li:last-child{border-bottom:none}
.notif-list .sev{width:6px;align-self:stretch;border-radius:3px}
.notif-list .sev.critical{background:var(--red)}
.notif-list .sev.warning{background:var(--amber)}
.notif-list .title{font-weight:600;font-size:13px;color:var(--text);margin-bottom:2px}
.notif-list .body{font-size:11.5px;color:var(--text-mute);
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:200px}
.notif-list .when{font-size:10.5px;color:var(--text-mute);white-space:nowrap}
.notif-empty{padding:30px;text-align:center;color:var(--text-mute);font-size:13px}

/* Settings popover (above user card) */
.user-card{cursor:pointer;border:none;width:100%;font-family:inherit;text-align:left;color:inherit}
.user-card:hover{background:rgba(255,255,255,.07)}
.settings-popover{background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.08);
  border-radius:10px;padding:12px;color:var(--sidebar-fg);margin-bottom:6px}
.settings-popover[hidden]{display:none}
.settings-popover h4{margin:0 0 10px;font-size:12.5px;font-weight:700;color:#fff;
  text-transform:uppercase;letter-spacing:.06em}
.setting-row{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:8px}
.setting-row:last-child{margin-bottom:0}
.setting-row label{font-size:11.5px;color:var(--sidebar-fg-mute);font-weight:600}
.seg{display:inline-flex;background:rgba(0,0,0,.2);border-radius:7px;padding:2px;gap:1px}
.seg-btn{background:transparent;border:none;color:var(--sidebar-fg);
  padding:5px 10px;border-radius:6px;font-size:11.5px;font-weight:600;cursor:pointer}
.seg-btn.active{background:var(--primary);color:#fff}
.seg-btn:hover:not(.active){background:rgba(255,255,255,.05)}

/* DrGb24-style status pills (FAULTED/MAINTENANCE/MONITOR/OPERATIONAL) */
.status.faulted{background:var(--red-soft);color:#b91c1c}
.status.faulted::before{background:var(--red)}
.status.maintenance{background:#fed7aa;color:#9a3412}
.status.maintenance::before{background:#ea580c}
.status.monitor{background:var(--amber-soft);color:#b45309}
.status.monitor::before{background:var(--amber)}
.status.operational{background:var(--green-soft);color:#047857}
.status.operational::before{background:var(--green)}

/* Severity pill */
.sev-pill{display:inline-block;padding:2px 8px;border-radius:6px;font-size:11px;font-weight:700;letter-spacing:.02em}
.sev-pill.Event{background:#e0e7ff;color:#3730a3}
.sev-pill.Warning{background:var(--amber-soft);color:#b45309}
.sev-pill.Error{background:var(--red-soft);color:#b91c1c}
.sev-pill.Fatal{background:#1f2937;color:#fff}

/* Active error chips */
.err-chips{display:flex;flex-wrap:wrap;gap:4px;max-width:220px}
.err-chip{background:#f1f5f9;color:#475569;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:500;
  max-width:160px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.err-chip.empty{background:transparent;color:var(--text-mute);font-style:italic;padding:2px 0}

/* Column-style heatmap: date labels BELOW the cells, with a vertical guide
   line ABOVE each label that extends up through the row stack so columns
   read like the user's paint mock-up. */
.heatmap.cols{display:grid;gap:4px;width:100%;position:relative}
.heatmap.cols .hrow{display:grid;gap:6px;align-items:center}
.heatmap.cols .hcell{position:relative;height:24px;border-radius:5px;background:transparent}
.heatmap.cols .hcell .fill{position:absolute;inset:0;border-radius:5px;z-index:1}
.heatmap.cols .rlabel{font-size:12px;font-weight:600;color:var(--text);
  padding-right:8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.heatmap.cols .hlabel{font-size:10.5px;color:var(--text-mute);text-align:left;
  font-variant-numeric:tabular-nums;white-space:nowrap;padding:14px 0 0;position:relative}
/* Tick line going UP from the foot label, anchored at the LEFT edge of each
   grid cell. This shifts the date label + its tick a hair to the left of the
   bar above it without moving the bars themselves. */
.heatmap.cols .hrow.foot .hlabel::before{
  content:"";position:absolute;left:0;bottom:100%;
  width:1px;height:10px;background:var(--text-mute);opacity:.6;
}
.heatmap.cols .hrow.foot{margin-top:2px}
.heatmap.cols .empty-slot{height:1px}

/* Dark mode */
body.dark{
  --bg:#0b1220;
  --card:#111a2c;
  --border:#1f2a44;
  --text:#e2e8f0;
  --text-mute:#94a3b8;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 1px 3px rgba(0,0,0,.5);
}
	body.dark .bell,body.dark .date-picker,body.dark .window-pill,body.dark .icon-btn,body.dark .pagination button,
	body.dark table.data thead th,body.dark .filter-row,body.dark .search input,
	body.dark .select,body.dark .input-date,body.dark .btn-ghost,body.dark .pred-card .btn-view{
	  background:#172238;color:var(--text);border-color:var(--border)
	}
	body.dark .runtime-item{background:#111a2c;border-color:var(--border)}
	body.dark .runtime-item.warn{background:#2a210d;border-color:#854d0e}
	body.dark .runtime-item.good{background:#102315;border-color:#166534}
body.dark .stat-bar{background:#1e293b}
body.dark .confidence-bar{background:#1e293b}
body.dark .robot-thumb,body.dark .pred-card .thumb{background:#1e293b;color:#cbd5e1}
body.dark table.data tbody tr:hover{background:#152034}
body.dark table.data thead th{background:#152034}
body.dark .modal,body.dark .popover,body.dark .notif-panel{background:#111a2c;color:var(--text);border-color:var(--border)}
body.dark .popover .row input,body.dark .popover .quick button{background:#172238;color:var(--text);border-color:var(--border)}
body.dark .err-chip{background:#1e293b;color:#cbd5e1}
body.dark .stat-title{color:var(--text-mute)}

.card{background:var(--card);border-radius:var(--radius);border:1px solid var(--border);
  box-shadow:var(--shadow);padding:20px}
.cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px;margin-bottom:18px}
.stat{position:relative;display:grid;grid-template-columns:64px 1fr auto;grid-template-rows:auto auto;
  align-items:center;column-gap:14px;overflow:hidden;padding-bottom:26px}
.stat-icon{width:56px;height:56px;border-radius:14px;display:grid;place-items:center;grid-row:span 2}
.icon-blue{background:var(--blue-soft);color:#3b82f6}
.icon-red{background:var(--red-soft);color:var(--red)}
.icon-green{background:var(--green-soft);color:var(--green)}
.icon-purple{background:var(--purple-soft);color:var(--purple)}
.icon-amber{background:var(--amber-soft);color:var(--amber)}
.stat-body{min-width:0}
.stat-title{font-size:13px;color:var(--text-mute);font-weight:600;margin-bottom:4px;
  display:flex;align-items:center;gap:6px;flex-wrap:wrap}
.head-select{margin-left:auto;border:1px solid var(--border);background:#fff;
  font-size:11px;color:var(--text);border-radius:6px;padding:2px 18px 2px 6px;
  appearance:none;cursor:pointer;font-family:inherit;font-weight:600;
  background-image:url("data:image/svg+xml;utf8,<svg fill='none' stroke='%236b7280' stroke-width='2' viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'><polyline points='6 9 12 15 18 9'/></svg>");
  background-repeat:no-repeat;background-position:right 4px center;background-size:10px}
body.dark .head-select{background-color:#172238;color:var(--text);border-color:var(--border)}
/* Larger head dropdown when used next to the "Robot-Level Head Outputs"
   section title (instead of inside a KPI card). */
.head-select-lg{margin-left:auto;font-size:13px;padding:6px 28px 6px 12px;
  background-position:right 8px center;background-size:12px}
.section-head-with-select{display:flex;align-items:center;justify-content:space-between;
  gap:12px;flex-wrap:wrap;margin-bottom:14px}
.section-head-with-select h3{margin:0;font-size:15px;font-weight:600}
/* Selected-head summary card lives INSIDE the Robot-Level Head Outputs
   section, just under the title row, so trim its outer margin. */
.selected-head-card{margin-bottom:18px}
/* Title row of the selected-head card: "Model Accuracy · Instant Fault
   Detection (ⓘ)". Model Accuracy is a bit bolder than the other lines so
   it stands out as the card label. */
.selected-head-card .stat-title-row{display:flex;align-items:center;gap:6px;flex-wrap:wrap}
.selected-head-card .model-accuracy-label{font-size:13.5px;font-weight:700;color:var(--text)}
.selected-head-card .title-sep{color:var(--text-mute);font-weight:400}
.selected-head-card .head-name-label{font-size:13px;font-weight:600;color:var(--text)}
/* BIG line is now the actual metric values (eg "%99.4 / %94.8 / %99.9").
   Keep it readable without overflowing when the value contains slashes. */
.selected-head-card .stat-value{font-size:24px;line-height:1.2;word-break:break-word;
  font-variant-numeric:tabular-nums;margin-top:2px}
.selected-head-card .stat-sub{font-size:12.5px;color:var(--text-mute)}
/* Info tooltip (ⓘ) — hover or focus pops a bubble explaining the metrics. */
.info-tip{position:relative;display:inline-flex;align-items:center;justify-content:center;
  width:18px;height:18px;border-radius:50%;color:var(--text-mute);
  font-size:12px;cursor:help;line-height:1;user-select:none;outline:none}
.info-tip:hover,.info-tip:focus-visible{color:var(--primary)}
.info-tip-bubble{position:absolute;top:calc(100% + 8px);left:50%;transform:translateX(-50%);
  background:#1f2937;color:#fff;font-size:11.5px;line-height:1.4;
  padding:10px 12px;border-radius:8px;width:max-content;max-width:260px;
  box-shadow:0 10px 24px rgba(15,23,42,.25);opacity:0;pointer-events:none;
  transition:opacity .15s;z-index:20;font-weight:400;text-align:left;white-space:normal}
.info-tip:hover .info-tip-bubble,.info-tip:focus-visible .info-tip-bubble{opacity:1}
.info-tip-bubble::before{content:"";position:absolute;bottom:100%;left:50%;
  transform:translateX(-50%);border:6px solid transparent;border-bottom-color:#1f2937}
body.dark .info-tip-bubble{background:#0f172a}
.stat-value{font-size:30px;font-weight:700;line-height:1.1}
.stat-sub{font-size:12px;color:var(--text-mute);margin-top:2px}
.stat-trend{text-align:right}
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
.stat-bar-fill.purple{background:var(--purple)}
.stat-bar-fill.amber{background:var(--amber)}

.charts{display:grid;grid-template-columns:1.4fr 1fr;gap:18px;margin-bottom:18px}
.chart-card{display:flex;flex-direction:column}
.chart-head{display:flex;justify-content:space-between;align-items:center;
  margin-bottom:14px;gap:12px;flex-wrap:wrap}
.chart-head h3{margin:0;font-size:15px;font-weight:600}
.info{color:var(--text-mute);font-size:12px;cursor:help}
.title-with-info{display:flex;align-items:center;gap:8px;min-width:0}
.info-trigger{width:22px;height:22px;border-radius:50%;border:1px solid var(--border);
  background:#fff;color:var(--primary);font-size:12px;font-weight:800;line-height:1;
  display:inline-grid;place-items:center;box-shadow:0 1px 2px rgba(15,23,42,.05)}
.info-trigger:hover{background:var(--blue-soft);border-color:#bfdbfe}
.info-popover-panel{position:absolute;top:52px;right:20px;width:min(360px,calc(100% - 40px));
  background:#fff;border:1px solid var(--border);border-radius:12px;box-shadow:0 16px 38px rgba(15,23,42,.18);
  z-index:40;padding:14px 16px;color:var(--text)}
.info-popover-panel[hidden]{display:none}
.info-popover-panel h4{margin:0 0 8px;font-size:13.5px;font-weight:700}
.info-popover-panel p{margin:0 0 10px;color:var(--text-mute);font-size:12.5px;line-height:1.45}
.info-popover-panel ul{margin:0;padding-left:18px;display:grid;gap:7px}
.info-popover-panel li{font-size:12.5px;line-height:1.35;color:var(--text)}
.chart-body{position:relative;height:260px}
.donut-wrap{display:flex;align-items:center;gap:24px;justify-content:center;height:100%}
.donut-wrap canvas{flex:0 0 200px;max-width:200px;max-height:200px}
.donut-wrap .legend{min-width:200px;max-width:260px;flex:1}
.legend{list-style:none;padding:0;margin:0;display:flex;flex-direction:column;gap:8px}
.legend li{display:flex;align-items:center;justify-content:space-between;font-size:13px;gap:12px}
.legend .dot{width:9px;height:9px;border-radius:50%;display:inline-block;margin-right:8px}
.legend .value{font-weight:600;white-space:nowrap}
.legend .pct{color:var(--text-mute);margin-left:4px}

.table-card{padding-bottom:14px}
.table-head{display:flex;justify-content:space-between;align-items:center;
  margin-bottom:14px;gap:12px;flex-wrap:wrap}
.table-head h3{margin:0;font-size:15px;font-weight:600}
.table-controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
.search{position:relative}
.search-icon{position:absolute;left:10px;top:50%;transform:translateY(-50%);
  font-size:12px;color:var(--text-mute);pointer-events:none}
.search input{border:1px solid var(--border);padding:8px 12px 8px 30px;border-radius:8px;
  background:#fff;font-size:13px;width:220px}
.select,.input-date{border:1px solid var(--border);padding:8px 32px 8px 12px;border-radius:8px;
  background:#fff url("data:image/svg+xml;utf8,<svg fill='none' stroke='%236b7280' stroke-width='2' viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'><polyline points='6 9 12 15 18 9'/></svg>") no-repeat right 10px center;
  background-size:12px;appearance:none;font-size:13px;min-width:150px}
.input-date{background:#fff;padding:8px 12px}
.btn-ghost{background:#fff;border:1px solid var(--border);padding:8px 14px;border-radius:8px;
  font-size:13px;font-weight:500;color:var(--text);display:inline-flex;align-items:center;gap:6px}
.btn-ghost:hover{background:#f9fafc}

.table-wrap{overflow-x:auto}
table.data{width:100%;border-collapse:collapse;font-size:13.5px}
table.data thead th{text-align:left;padding:10px 14px;border-bottom:1px solid var(--border);
  color:var(--text-mute);font-weight:600;font-size:12.5px;background:#fafbfc;white-space:nowrap}
table.data tbody td{padding:14px;border-bottom:1px solid var(--border);vertical-align:middle}
table.data tbody tr:hover{background:#f9fafc}
table.data tbody tr:last-child td{border-bottom:none}
.action-col{text-align:right;white-space:nowrap}

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
.status.resolved{background:var(--green-soft);color:#047857}
.status.resolved::before{background:var(--green)}
.status.in-progress{background:var(--amber-soft);color:#b45309}
.status.in-progress::before{background:var(--amber)}

.cat-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:8px}
.mono-id{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-size:11.5px;color:var(--text);background:#f1f3f7;border:1px solid var(--border);
  padding:2px 8px;border-radius:6px;display:inline-block;max-width:140px;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;vertical-align:middle}
body.dark .mono-id{background:#172238;color:var(--text)}
.fault-name{font-weight:500}
.fault-detail{font-size:11.5px;color:var(--text-mute);max-width:300px;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
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
.icon-btn{background:#fff;border:1px solid var(--border);width:32px;height:32px;
  border-radius:8px;color:var(--text-mute);display:inline-grid;place-items:center;margin-left:4px;
  cursor:pointer}
.icon-btn:hover{color:var(--text)}
.kebab{background:transparent;border:none;color:var(--text-mute);padding:4px 8px;font-size:16px;line-height:1}
.empty{text-align:center;padding:40px;color:var(--text-mute)}

.table-foot{display:flex;justify-content:space-between;align-items:center;
  padding:14px 4px 4px;flex-wrap:wrap;gap:10px}
.pagination-info{font-size:12.5px;color:var(--text-mute)}
.pagination{display:flex;gap:4px}
.pagination button{border:1px solid var(--border);background:#fff;width:32px;height:32px;
  border-radius:6px;font-size:12.5px;color:var(--text)}
.pagination button.active{background:var(--primary);color:#fff;border-color:var(--primary)}
.pagination button:disabled{opacity:.4;cursor:not-allowed}

.modal-backdrop[hidden]{display:none}
.modal-backdrop{position:fixed;inset:0;background:rgba(15,23,42,.55);
  display:grid;place-items:center;padding:20px;z-index:100}
.modal{background:#fff;border-radius:14px;width:min(640px,100%);max-height:85vh;
  overflow:hidden;display:flex;flex-direction:column;
  box-shadow:0 20px 50px rgba(15,23,42,.25)}
.modal-head{display:flex;justify-content:space-between;align-items:center;
  padding:16px 20px;border-bottom:1px solid var(--border)}
.modal-head h2{margin:0;font-size:16px}
.modal-close{border:none;background:transparent;font-size:18px;color:var(--text-mute);cursor:pointer}
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

.page{display:none}
.page.active{display:block}

.bar-chart-wrap{position:relative;height:280px}
.filter-row{display:flex;gap:10px;flex-wrap:wrap;align-items:center;
  padding:14px;background:#fff;border:1px solid var(--border);
  border-radius:var(--radius);margin-bottom:18px}
.filter-row .arrow{color:var(--text-mute)}
.cat-legend{display:flex;flex-direction:column;gap:8px;padding-left:8px}
.cat-legend li{display:flex;align-items:center;font-size:12.5px;gap:8px;color:var(--text)}
.cat-legend .dot{width:9px;height:9px;border-radius:50%}

.heat-card,.degr-card{padding:20px}
.degr-card{position:relative}
.preds-row{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:18px}
.heatmap{display:grid;gap:6px;align-items:center;margin-top:10px}
.heatmap .row{display:flex;align-items:center;gap:6px}
.heatmap .rlabel{font-size:11px;color:var(--text-mute);width:120px;white-space:nowrap;
  overflow:hidden;text-overflow:ellipsis}
.heatmap .cell{flex:1;height:30px;border-radius:6px;min-width:30px}
.heatmap .clabels{display:flex;gap:6px;padding-left:126px;color:var(--text-mute);font-size:10.5px;margin-top:6px}
.heatmap .clabels span{flex:1;text-align:center;min-width:30px}
.heat-legend{display:flex;flex-direction:column;align-items:center;gap:6px;font-size:11px;
  color:var(--text-mute);margin-left:10px}
.heat-legend .bar{width:14px;height:100px;border-radius:6px;
  background:linear-gradient(180deg,#ef4444,#f59e0b,#10b981)}

.degr-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:10px}
.degr-grid .mini{position:relative;height:220px}
.degr-grid h4{margin:0 0 6px;font-size:13px;font-weight:600}
.degr-grid .pred-badge{position:absolute;top:0;right:0;background:var(--red-soft);
  color:#b91c1c;font-size:11px;padding:4px 8px;border-radius:6px;font-weight:600;
  text-align:center;line-height:1.2;z-index:2}
.degr-single{margin-top:10px}
.degr-single h4{margin:0 0 6px;font-size:13px;font-weight:600}
.degr-single .lstm-big{position:relative;height:300px}
.degr-single .pred-badge{position:absolute;top:0;right:0;background:var(--red-soft);
  color:#b91c1c;font-size:11px;padding:4px 8px;border-radius:6px;font-weight:600;
  text-align:center;line-height:1.2;z-index:2}

.pred-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin-top:14px}
.pred-card{background:#fff;border:1px solid var(--border);border-radius:var(--radius);
  padding:16px;display:flex;flex-direction:column;gap:10px;
  border-top:3px solid var(--cat-other)}
.pred-card.Critical{border-top-color:var(--red)}
.pred-card.High{border-top-color:var(--amber)}
.pred-card.Medium{border-top-color:#facc15}
.pred-card.Low{border-top-color:var(--green)}
.pred-card .head{display:flex;align-items:center;gap:10px}
.pred-card .head .thumb{width:36px;height:36px;border-radius:50%;background:#f1f5f9;
  display:grid;place-items:center;color:#475569}
.pred-card .head .id{font-weight:700;font-size:13.5px}
.pred-card .head .area{font-size:11px;color:var(--text-mute)}
.pred-card .prob-row{display:flex;align-items:center;justify-content:space-between}
.pred-card .prob{font-size:22px;font-weight:700}
.pred-card .risk-pill{font-size:11px;padding:3px 8px;border-radius:999px;font-weight:600}
.pred-card .risk-pill.Critical{background:var(--red-soft);color:#b91c1c}
.pred-card .risk-pill.High{background:var(--amber-soft);color:#b45309}
.pred-card .risk-pill.Medium{background:#fef9c3;color:#a16207}
.pred-card .risk-pill.Low{background:var(--green-soft);color:#047857}
.pred-card .label-sm{font-size:11px;color:var(--text-mute);text-transform:uppercase;letter-spacing:.04em}
.pred-card .issue{font-weight:600;font-size:13px}
.pred-card .time{font-size:12.5px;color:var(--text)}
.pred-card .btn-view{background:#fff;color:var(--primary);border:1px solid var(--primary);
  padding:6px 10px;border-radius:8px;font-weight:600;font-size:12.5px;width:100%;cursor:pointer}
.pred-card .btn-view:hover{background:var(--primary);color:#fff}
.pred-load-more{display:flex;align-items:center;justify-content:center;gap:12px;margin-top:14px;flex-wrap:wrap}
.pred-load-more[hidden]{display:none}
.btn-load-more{background:var(--primary);color:#fff;border:1px solid var(--primary);
  border-radius:8px;padding:8px 16px;font-size:12.5px;font-weight:700}
.btn-load-more:hover{background:var(--primary-2)}
.pred-load-more .count{font-size:12px;color:var(--text-mute)}
body.dark .info-trigger{background:#172238;color:#93c5fd;border-color:var(--border)}
body.dark .info-popover-panel{background:#111a2c;border-color:var(--border);box-shadow:0 16px 38px rgba(0,0,0,.45)}

	@media (max-width:1100px){.charts,.preds-row,.runtime-strip{grid-template-columns:1fr}
  .donut-wrap{flex-wrap:wrap}
  .degr-grid{grid-template-columns:1fr}}
@media (max-width:880px){.cards{grid-template-columns:1fr}
  .stat{grid-template-columns:56px 1fr auto}}
@media (max-width:760px){.sidebar{position:fixed;left:0;top:0;transform:translateX(-100%);
  transition:transform .25s;z-index:50;box-shadow:4px 0 20px rgba(0,0,0,.15)}
  .sidebar.open{transform:translateX(0)}
  .main{padding:18px 16px 28px}
  .hamburger{display:inline-flex;align-items:center;justify-content:center}
  .topbar{align-items:center}
  .topbar-right{width:100%;justify-content:flex-end}
  .topbar-left h1{font-size:19px}
  .search input{width:100%}.search{flex:1;min-width:200px}.select{flex:1;min-width:140px}
  .heatmap .rlabel{width:90px}
  .heatmap .clabels{padding-left:96px}
  .fault-detail{max-width:160px}}
@media (max-width:520px){.stat{grid-template-columns:48px 1fr;padding-bottom:24px}
  .stat-trend{grid-column:2/3;text-align:left;margin-top:4px}
  .stat-icon{grid-row:span 2;width:44px;height:44px}
  .stat-value{font-size:24px}
  .pred-cards{grid-template-columns:1fr}}
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
      <nav class="nav" id="mainNav">
        <button class="nav-item active" data-page="dashboard"><span class="nav-icon">⌂</span><span class="nav-label" data-i18n="navDashboard">Dashboard</span></button>
        <button class="nav-item" data-page="predictions"><span class="nav-icon">▣</span><span class="nav-label" data-i18n="navPredictions">Predictions &amp; Analysis</span><span class="nav-arrow">›</span></button>
        <button class="nav-item" data-page="fault-history"><span class="nav-icon">⏲</span><span class="nav-label" data-i18n="navFaultHistory">Fault History</span><span class="nav-arrow">›</span></button>
      </nav>
    </div>
    <div class="sidebar-bottom">
      <div class="status-card">
        <span class="status-dot"></span>
        <div><div class="status-title" data-i18n="systemStatus">System Status</div><div class="status-sub" data-i18n="allSystemsOperational">All Systems Operational</div></div>
      </div>

      <!-- Settings popover above the user card -->
      <div class="settings-popover" id="settingsPopover" hidden onclick="event.stopPropagation()">
        <h4 data-i18n="settings">Settings</h4>
        <div class="setting-row">
          <label data-i18n="language">Language</label>
          <div class="seg">
            <button class="seg-btn" data-lang="en" onclick="setLanguage('en')">EN</button>
            <button class="seg-btn" data-lang="tr" onclick="setLanguage('tr')">TR</button>
          </div>
        </div>
        <div class="setting-row">
          <label data-i18n="theme">Theme</label>
          <div class="seg">
            <button class="seg-btn" data-theme="light" onclick="setTheme('light')"><span data-i18n="lightMode">Light</span></button>
            <button class="seg-btn" data-theme="dark"  onclick="setTheme('dark')"><span data-i18n="darkMode">Dark</span></button>
          </div>
        </div>
      </div>

      <button class="user-card" onclick="toggleSettings(event)" aria-label="Settings">
        <div class="user-avatar">AE</div>
        <div class="user-meta"><div class="user-name">Admin User</div><div class="user-mail">admin@roboclean.com</div></div>
        <span class="user-caret">▴</span>
      </button>
    </div>
  </aside>

  <main class="main">
    <section class="runtime-strip" id="runtimeStrip"></section>

    <section class="page active" id="page-dashboard">
      <header class="topbar">
        <button class="hamburger" onclick="toggleSidebar()" aria-label="Open menu">☰</button>
        <div class="topbar-left">
          <h1 data-i18n="dashboardTitle">Dashboard</h1>
          <p data-i18n="dashboardSubtitle">Real-time overview of your autonomous cleaning robots</p>
        </div>
        <div class="topbar-right">
          <button class="bell" id="bellBtn" aria-label="Notifications" onclick="toggleNotifPanel(event)"><span>🔔</span><span class="bell-badge" id="bellBadge">0</span></button>
          <button class="date-picker" id="dateRange" onclick="toggleDatePicker(event,'dateRange')">Date range ▾</button>
        </div>
      </header>
      <section class="cards">
        <div class="card stat">
          <div class="stat-icon icon-blue"><svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="8" width="16" height="11" rx="2"></rect><path d="M8 8V6a4 4 0 0 1 8 0v2"></path><circle cx="9" cy="13" r="1"></circle><circle cx="15" cy="13" r="1"></circle></svg></div>
          <div class="stat-body">
            <div class="stat-title" data-i18n="activeRobots">Active Robots</div>
            <div class="stat-value" id="activeRobotsValue">—</div>
            <div class="stat-sub" id="activeRobotsSub">of — total robots</div>
          </div>
          <div class="stat-trend" id="activeRobotsTrend"></div>
          <div class="stat-bar"><div class="stat-bar-fill blue" id="activeRobotsBar" style="width:0%"></div></div>
        </div>
        <div class="card stat">
          <div class="stat-icon icon-red"><svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3 L22 20 L2 20 Z"></path><path d="M12 9v5"></path><circle cx="12" cy="17" r="1" fill="currentColor"></circle></svg></div>
          <div class="stat-body">
            <div class="stat-title" data-i18n="criticalAlerts">Critical Fault Alerts</div>
            <div class="stat-value" id="criticalValue">—</div>
            <div class="stat-sub" data-i18n="requireAttention">robots require attention</div>
          </div>
          <div class="stat-trend" id="criticalTrend"></div>
          <div class="stat-bar"><div class="stat-bar-fill red" id="criticalBar" style="width:0%"></div></div>
        </div>
        <div class="card stat">
          <div class="stat-icon icon-green"><svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"></path></svg></div>
          <div class="stat-body">
            <div class="stat-title" data-i18n="fleetHealth">Overall Fleet Health</div>
            <div class="stat-value" id="fleetValue">—%</div>
            <div class="stat-sub" data-i18n="healthyRobots">healthy robots</div>
          </div>
          <div class="stat-trend" id="fleetTrend"></div>
          <div class="stat-bar"><div class="stat-bar-fill green" id="fleetBar" style="width:0%"></div></div>
        </div>
      </section>

      <section class="charts">
        <div class="card chart-card">
          <div class="chart-head">
            <h3 data-i18n="sensorTrend">Sensor Anomaly Trend</h3>
            <select class="select" id="anomalyRobotSelect"><option value="">All Robots</option></select>
          </div>
          <div class="chart-body"><canvas id="anomalyChart"></canvas></div>
        </div>
        <div class="card chart-card">
          <div class="chart-head"><h3 data-i18n="faultDist">Fault Distribution</h3></div>
          <div class="chart-body donut-wrap">
            <canvas id="faultChart"></canvas>
            <ul class="legend" id="faultLegend"></ul>
          </div>
        </div>
      </section>

      <section class="card table-card">
        <div class="table-head">
          <h3 data-i18n="robotHealth">Robot Health Overview</h3>
          <div class="table-controls">
            <div class="search"><span class="search-icon">🔎</span><input type="search" id="robotSearch" placeholder="Search robot ID..." data-i18n-ph="searchRobot" /></div>
            <select class="select" id="statusFilter"></select>
            <select class="select" id="faultFilter"></select>
          </div>
        </div>
        <div class="table-wrap">
          <table class="data">
            <thead><tr>
              <th data-i18n="colRobot">Robot</th>
              <th data-i18n="colStatus">Status</th>
              <th data-i18n="colSemantic">Semantic Score</th>
              <th data-i18n="colSeverity">Severity</th>
              <th data-i18n="colForecast">7-Day Forecast</th>
              <th data-i18n="colErrors">Active Errors</th>
              <th data-i18n="colLastUpdated">Last Updated</th>
              <th class="action-col" data-i18n="colAction">Action</th>
            </tr></thead>
            <tbody id="robotTableBody"><tr><td colspan="8" class="empty">Loading…</td></tr></tbody>
          </table>
        </div>
        <div class="table-foot">
          <div class="pagination-info" id="paginationInfo">Showing 0 of 0 robots</div>
          <nav class="pagination" id="pagination"></nav>
        </div>
      </section>
    </section>

    <section class="page" id="page-fault-history">
      <header class="topbar">
        <button class="hamburger" onclick="toggleSidebar()" aria-label="Open menu">☰</button>
        <div class="topbar-left">
          <h1 data-i18n="faultHistoryTitle">Fault History</h1>
          <p data-i18n="faultHistorySub">Browse and analyze historical faults and system issues</p>
        </div>
        <div class="topbar-right">
          <button class="bell" aria-label="Notifications" onclick="toggleNotifPanel(event)"><span>🔔</span><span class="bell-badge" id="bellBadge2">0</span></button>
          <button class="date-picker" id="fhDateRange" onclick="toggleDatePicker(event,'fhDateRange')">Date range ▾</button>
        </div>
      </header>

      <div class="card" style="margin-bottom:18px">
        <div class="chart-head">
          <h3 data-i18n="faultFreq">Fault Frequency Over Time (Last 6 Months)</h3>
        </div>
        <div style="display:grid;grid-template-columns:1fr auto;gap:20px;align-items:center">
          <div class="bar-chart-wrap"><canvas id="freqChart"></canvas></div>
          <ul class="cat-legend">
            <li><span class="dot" style="background:var(--cat-brush)"></span><span data-i18n="catBrush">Brush Motor Issues</span></li>
            <li><span class="dot" style="background:var(--cat-battery)"></span><span data-i18n="catBattery">Battery &amp; Power</span></li>
            <li><span class="dot" style="background:var(--cat-nav)"></span><span data-i18n="catNav">Navigation System</span></li>
            <li><span class="dot" style="background:var(--cat-vacuum)"></span><span data-i18n="catVacuum">Vacuum System</span></li>
            <li><span class="dot" style="background:var(--cat-other)"></span><span data-i18n="catOther">Other</span></li>
          </ul>
        </div>
      </div>

      <div class="filter-row">
        <div class="search" style="flex:1;min-width:180px">
          <span class="search-icon">🔎</span>
          <input type="search" id="fhSearch" placeholder="Search Error ID..." data-i18n-ph="searchErrorId" />
        </div>
        <select class="select" id="fhRobotFilter"></select>
        <select class="select" id="fhFaultFilter"></select>
        <input class="input-date" type="date" id="fhStartDate" />
        <span class="arrow">→</span>
        <input class="input-date" type="date" id="fhEndDate" />
        <button class="btn-ghost" onclick="clearFhFilters()">↻ <span data-i18n="clearFilters">Clear Filters</span></button>
      </div>

      <section class="card table-card">
        <div class="table-wrap">
          <table class="data">
            <thead><tr>
              <th data-i18n="fhColDateTime">Date &amp; Time</th>
              <th data-i18n="fhColRobotId">Robot ID</th>
              <th data-i18n="fhColErrorId">Error ID</th>
              <th data-i18n="fhColFaultType">Fault Type</th>
              <th data-i18n="fhColDiagnosed">Diagnosed Issue (From Logs)</th>
              <th data-i18n="fhColDowntime">Downtime Duration</th>
              <th data-i18n="fhColResolution">Resolution Status</th>
              <th class="action-col" data-i18n="fhColActions">Actions</th>
            </tr></thead>
            <tbody id="fhTableBody"><tr><td colspan="8" class="empty">Loading…</td></tr></tbody>
          </table>
        </div>
        <div class="table-foot">
          <div class="pagination-info" id="fhPaginationInfo">Showing 0 of 0 faults</div>
          <nav class="pagination" id="fhPagination"></nav>
        </div>
      </section>
    </section>

    <section class="page" id="page-predictions">
      <header class="topbar">
        <button class="hamburger" onclick="toggleSidebar()" aria-label="Open menu">☰</button>
        <div class="topbar-left">
          <h1 data-i18n="predictionsTitle">Predictions &amp; Analysis</h1>
          <p data-i18n="predictionsSubtitle">AI-powered insights and predictive analytics for your robot fleet</p>
        </div>
        <div class="topbar-right">
          <button class="bell" aria-label="Notifications" onclick="toggleNotifPanel(event)"><span>🔔</span><span class="bell-badge" id="bellBadge3">0</span></button>
          <span class="window-pill" id="predWindowPill">
            <span data-i18n="predictionWindow">Prediction Window</span>
            <strong id="predWindowPillValue" data-i18n="next7Days">Next 7 Days</strong>
          </span>
        </div>
      </header>

      <div class="filter-row">
        <select class="select" id="predRobotFilter"></select>
        <select class="select" id="predCategoryFilter"></select>
      </div>

      <div class="preds-row">
        <div class="card heat-card">
          <div class="chart-head">
            <h3 data-i18n="fleetRiskHeatmap">Fleet Risk Assessment Heatmap</h3>
          </div>
          <div style="display:flex;gap:12px;align-items:flex-end">
            <div class="heatmap cols" id="heatmapGrid" style="flex:1"></div>
            <div class="heat-legend"><div data-i18n="heatHigh">High</div><div class="bar"></div><div data-i18n="heatLow">Low</div></div>
          </div>
        </div>

        <div class="card degr-card">
          <div class="chart-head">
            <div class="title-with-info">
              <h3 data-i18n="componentDegradation">Component Degradation Over Time</h3>
              <button class="info-trigger" type="button" aria-label="How to read this chart" onclick="toggleDegradationInfo(event)">i</button>
            </div>
            <select class="select" id="degrCategoryFilter"></select>
          </div>
          <div class="info-popover-panel" id="degrInfoPopover" hidden onclick="event.stopPropagation()">
            <h4 data-i18n="degradationInfoTitle">How to read this chart</h4>
            <p data-i18n="degradationInfoBody">The solid line is observed component risk. The dashed line is the model forecast for the fixed prediction window.</p>
            <ul>
              <li data-i18n="degradationInfoLow">Low and flat values mean the component is stable.</li>
              <li data-i18n="degradationInfoRising">A rising line means the component is degrading and should be monitored.</li>
              <li data-i18n="degradationInfoHigh">Values near the top mean higher failure probability and stronger maintenance priority.</li>
            </ul>
          </div>
          <div class="degr-single">
            <div class="mini lstm-big">
              <canvas id="lstmChart"></canvas>
            </div>
          </div>
        </div>
      </div>

      <section class="cards" style="grid-template-columns:repeat(2,minmax(0,1fr))">
        <div class="card stat">
          <div class="stat-icon icon-red"><svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3 L22 20 L2 20 Z"></path><path d="M12 9v5"></path><circle cx="12" cy="17" r="1" fill="currentColor"></circle></svg></div>
          <div class="stat-body"><div class="stat-title" data-i18n="highRiskRobots">High Severity Robots</div><div class="stat-value" id="predHighRisk">—</div></div>
          <div class="stat-trend" id="predHighRiskTrend"></div>
          <div class="stat-bar"><div class="stat-bar-fill red" id="predHighRiskBar" style="width:0%"></div></div>
        </div>
        <div class="card stat">
          <div class="stat-icon icon-blue"><svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"></circle><path d="M12 7v6l4 2"></path></svg></div>
          <div class="stat-body"><div class="stat-title" data-i18n="predFailures">Predicted Failures</div><div class="stat-value" id="predFail">—</div></div>
          <div class="stat-trend" id="predFailTrend"></div>
          <div class="stat-bar"><div class="stat-bar-fill blue" id="predFailBar" style="width:0%"></div></div>
        </div>
      </section>

      <section class="card">
        <!-- Title row with the Head selector to its right. The dropdown
             controls everything rendered inside this card: the Model
             Accuracy summary AND the per-robot head output cards below. -->
        <div class="chart-head section-head-with-select">
          <h3 data-i18n="topFailurePred">Robot-Level Head Outputs</h3>
          <select class="head-select head-select-lg" id="modelHeadSelect">
            <option value="1">Head 1</option>
            <option value="2">Head 2</option>
            <option value="3">Head 3</option>
            <option value="4">Head 4</option>
          </select>
        </div>

        <!-- Selected-head summary (formerly the 4th KPI card). Reads from
             the same /api/predictions/stats payload and reacts to both the
             top filters AND the head selector right above. -->
        <div class="card stat selected-head-card">
          <div class="stat-icon icon-purple"><svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"></circle><path d="M8 12l2.5 2.5L16 9"></path></svg></div>
          <div class="stat-body">
            <div class="stat-title stat-title-row">
              <span class="model-accuracy-label" data-i18n="modelAccuracy">Model Accuracy</span>
              <span class="title-sep">·</span>
              <span class="head-name-label" id="predHeadName">Instant Fault Detection</span>
              <span class="info-tip" id="predHeadInfo" tabindex="0" aria-label="info">ⓘ
                <span class="info-tip-bubble" id="predHeadInfoBubble"></span>
              </span>
            </div>
            <div class="stat-value" id="predAcc">—%</div>
            <div class="stat-sub" id="predHeadHint"></div>
          </div>
          <div class="stat-trend" id="predAccTrend"></div>
          <div class="stat-bar"><div class="stat-bar-fill purple" id="predAccBar" style="width:0%"></div></div>
        </div>

      <section class="card">
        <div class="chart-head">
          <h3 data-i18n="topFailurePred">Top Failure Predictions (Next 48 Hours)</h3>
        </div>
        <div class="pred-cards" id="predCardsContainer"><div class="empty">Loading…</div></div>
        <div class="pred-load-more" id="predLoadMoreWrap" hidden>
          <button class="btn-load-more" type="button" onclick="loadMoreTopFailures()" data-i18n="loadMoreRobots">Load more</button>
          <span class="count" id="predLoadMoreInfo"></span>
        </div>
      </section>
    </section>

  </main>
</div>

<div class="modal-backdrop" id="modalBackdrop" hidden>
  <div class="modal" role="dialog" aria-modal="true">
    <div class="modal-head">
      <h2 id="modalTitle">Robot Details</h2>
      <button class="modal-close" onclick="closeModal()">✕</button>
    </div>
    <div class="modal-body" id="modalBody">Loading…</div>
  </div>
</div>

<!-- Date range popover (single global instance, repositioned per page) -->
<div class="popover" id="datePopover" hidden onclick="event.stopPropagation()">
  <h4 data-i18n="selectRange">Select Date Range</h4>
  <div class="quick">
    <button class="past-range" onclick="applyQuickRange(7)" data-i18n="last7">Last 7 days</button>
    <button class="past-range" onclick="applyQuickRange('all')" data-i18n="allTime">All time</button>
    <button class="future-range" onclick="applyFutureRange(7)" data-i18n="next7Days">Next 7 Days</button>
  </div>
  <div class="row past-range"><label data-i18n="startDate">Start date</label><input type="date" id="rangeStartInput" /></div>
  <div class="row past-range"><label data-i18n="endDate">End date</label><input type="date" id="rangeEndInput" /></div>
  <div class="actions past-range">
    <button onclick="resetRange()" data-i18n="reset">Reset</button>
    <button class="primary" onclick="applyRange()" data-i18n="apply">Apply</button>
  </div>
</div>

<!-- Notifications panel -->
<div class="notif-panel" id="notifPanel" hidden onclick="event.stopPropagation()">
  <div class="notif-head">
    <h4 data-i18n="notifications">Notifications</h4>
    <button class="clear" onclick="markAllRead()" data-i18n="markAllRead">Mark all read</button>
  </div>
  <ul class="notif-list" id="notifList"><li class="notif-empty">Loading…</li></ul>
</div>

<script>
// ===== i18n =====
const I18N = {
  en: {
    navDashboard: "Dashboard", navPredictions: "Predictions & Analysis", navFaultHistory: "Fault History",
    dashboardTitle: "Dashboard", dashboardSubtitle: "Real-time overview of your autonomous cleaning robots",
    activeRobots: "Active Robots", criticalAlerts: "Critical Fault Alerts", fleetHealth: "Overall Fleet Health",
    healthyRobots: "healthy robots", requireAttention: "robots require attention",
    robotHealth: "Robot Health Overview", searchRobot: "Search robot ID...",
    colRobot:"Robot", colStatus:"Status", colFault:"Predicted Fault", colSemantic:"Fault Probability",
    colSeverity:"Severity", colForecast:"7-Day Forecast", colErrors:"Active Errors",
    colLastUpdated:"Last Updated", colAction:"Action",
    statusFAULTED:"Faulted", statusMAINTENANCE:"Needs Maintenance", statusMONITOR:"Monitor", statusOPERATIONAL:"Operational",
    statusCritical:"Critical", statusWarning:"Warning", statusNormal:"Normal", statusUnknown:"Unknown",
    predictionsTitle: "Predictions & Analysis", predictionsSubtitle: "AI-powered insights and predictive analytics for your robot fleet",
    fleetRiskHeatmap: "Fleet Risk Assessment Heatmap", componentDegradation: "Component Degradation Over Time",
    lstmModelPred: "LSTM Model Prediction", heatHigh:"High", heatLow:"Low",
    faultHistoryTitle: "Fault History", faultHistorySub: "Browse and analyze historical faults and system issues",
    settings: "Settings", language: "Language", theme: "Theme", lightMode:"Light", darkMode:"Dark",
    systemStatus: "System Status", allSystemsOperational: "All Systems Operational",
    vsPrev:"vs prev period", searchFaults:"Search faults...", searchErrorId:"Search Error ID...",
    noRobots:"No robots found", noFaults:"No faults match the filters", noAlerts:"No active alerts",
    showing: "Showing", to: "to", of: "of", robots: "robots", faults: "faults",
    notifications: "Notifications", markAllRead: "Mark all read",
    selectRange: "Select Date Range", predictionWindow:"Prediction Window",
    last7:"Last 7 days", last30:"Last 30 days", last90:"Last 90 days", allTime:"All time",
    next7Days:"Next 7 Days",
    forecastLabel:"Forecast", loadMoreRobots:"Load more", showingPredictions:"Showing {0} of {1}",
    degradationInfoTitle:"How to read this chart",
    degradationInfoBody:"The solid line is observed component risk. The dashed line is the model forecast for the fixed prediction window.",
    degradationInfoLow:"Low and flat values mean the component is stable.",
    degradationInfoRising:"A rising line means the component is degrading and should be monitored.",
    degradationInfoHigh:"Values near the top mean higher failure probability and stronger maintenance priority.",
    startDate:"Start date", endDate:"End date", reset:"Reset", apply:"Apply",
    failureProb: "Failure Probability", predictedIssue: "Predicted Issue", estimatedTime: "Estimated Time", viewDetails: "View Details",
    avgFleetHealth: "Average Fleet Health", highRiskRobots: "High Severity Robots", predFailures: "Predicted Failures", modelAccuracy: "Model Accuracy",
    topFailurePred: "Robot-Level Head Outputs",
    legendActual:"Actual", legendPredicted:"Predicted", predictedFailure:"Predicted Failure",
    riskCritical:"Critical Risk", riskHigh:"High Risk", riskMedium:"Medium Risk", riskLow:"Low Risk",
    allRobots:"All Robots", allComponents:"All Components", allStatuses:"All Statuses", allFaultTypes:"All Fault Types",
    last7Days:"Last 7 Days", last30Days:"Last 30 Days", last90Days:"Last 90 Days",
    totalFaults:"Total Faults", ofNRobots:"of {0} total robots",
    showingNofM:"Showing {0} to {1} of {2} {3}",
    sensorTrend:"Sensor Anomaly Trend", faultDist:"Fault Distribution", faultFreq:"Fault Frequency Over Time (Last 6 Months)",
    fhColDateTime:"Date & Time", fhColRobotId:"Robot ID", fhColErrorId:"Error ID", fhColFaultType:"Fault Type",
    fhColDiagnosed:"Diagnosed Issue (From Logs)", fhColDowntime:"Downtime Duration",
    fhColResolution:"Resolution Status", fhColActions:"Actions",
    clearFilters:"Clear Filters",
    catBrush:"Brush Motor Issues", catBattery:"Battery & Power",
    catNav:"Navigation System", catVacuum:"Vacuum System", catOther:"Other",
    resolved:"Resolved", inProgress:"In Progress",
    head1Label:"Instant Fault Detection", head2Label:"Fault Severity",
    head3Label:"Future Forecast", head4Label:"Fault ETA",
    metricAccuracy:"Accuracy", metricAuc:"AUC-ROC", metricMae:"MAE",
    headTip1: "Instant fault detection metrics. Accuracy = correct prediction rate, F1 = precision + recall balance, AUC = how well the model separates failing from healthy robots. Higher is better.",
    headTip2: "Fault severity classification accuracy across four levels (Event / Warning / Error / Fatal).",
    headTip3: "AUC-ROC of the 7-day failure forecast. 1.0 = perfect separation, 0.5 = random guessing.",
    headTip4: "Mean Absolute Error of the fault-ETA regression head, expressed in hours. Lower is better — it tells you how far the predicted time-to-failure is off on average.",
  },
  tr: {
    navDashboard: "Gösterge Paneli", navPredictions: "Tahminler & Analiz", navFaultHistory: "Arıza Geçmişi",
    dashboardTitle: "Gösterge Paneli", dashboardSubtitle: "Otonom temizlik robotlarınızın gerçek zamanlı görünümü",
    activeRobots: "Aktif Robotlar", criticalAlerts: "Kritik Arıza Uyarıları", fleetHealth: "Genel Filo Sağlığı",
    healthyRobots: "sağlıklı robot", requireAttention: "robot ilgi bekliyor",
    robotHealth: "Robot Sağlık Özeti", searchRobot: "Robot ID ara...",
    colRobot:"Robot", colStatus:"Durum", colFault:"Tahmin Edilen Arıza", colSemantic:"Arıza Olasılığı",
    colSeverity:"Şiddet", colForecast:"7 Günlük Öngörü", colErrors:"Aktif Hatalar",
    colLastUpdated:"Son Güncelleme", colAction:"İşlem",
    statusFAULTED:"Arızalı", statusMAINTENANCE:"Bakım Gerekli", statusMONITOR:"Takipte", statusOPERATIONAL:"Çalışır Durumda",
    statusCritical:"Kritik", statusWarning:"Uyarı", statusNormal:"Normal", statusUnknown:"Bilinmeyen",
    predictionsTitle: "Tahminler & Analiz", predictionsSubtitle: "Robot filonuz için yapay zeka destekli öngörüler",
    fleetRiskHeatmap: "Filo Risk Haritası", componentDegradation: "Bileşen Yıpranma Trendi",
    lstmModelPred: "LSTM Model Tahmini", heatHigh:"Yüksek", heatLow:"Düşük",
    faultHistoryTitle: "Arıza Geçmişi", faultHistorySub: "Tarihsel arızaları ve sistem sorunlarını incele",
    settings: "Ayarlar", language: "Dil", theme: "Tema", lightMode:"Aydınlık", darkMode:"Karanlık",
    systemStatus: "Sistem Durumu", allSystemsOperational: "Tüm Sistemler Çalışıyor",
    vsPrev:"önceki döneme göre", searchFaults:"Arıza ara...", searchErrorId:"Hata ID ara...",
    noRobots:"Robot bulunamadı", noFaults:"Filtrelere uyan arıza yok", noAlerts:"Aktif uyarı yok",
    showing:"Gösteriliyor", to:"-", of:"/", robots:"robot", faults:"arıza",
    notifications: "Bildirimler", markAllRead: "Tümünü okundu say",
    selectRange: "Tarih Aralığı Seç", predictionWindow:"Tahmin Penceresi",
    last7:"Son 7 gün", last30:"Son 30 gün", last90:"Son 90 gün", allTime:"Tüm zaman",
    next7Days:"Sonraki 7 Gün",
    forecastLabel:"Öngörü", loadMoreRobots:"Daha fazla yükle", showingPredictions:"{1} robottan {0} gösteriliyor",
    degradationInfoTitle:"Bu grafik nasıl okunur?",
    degradationInfoBody:"Düz çizgi gözlenen bileşen riskini, kesikli çizgi sabit tahmin penceresi için model öngörüsünü gösterir.",
    degradationInfoLow:"Düşük ve yatay değerler bileşenin stabil olduğunu gösterir.",
    degradationInfoRising:"Yükselen çizgi bileşenin yıprandığını ve izlenmesi gerektiğini gösterir.",
    degradationInfoHigh:"Üst seviyelere yaklaşan değerler daha yüksek arıza olasılığı ve bakım önceliği demektir.",
    startDate:"Başlangıç tarihi", endDate:"Bitiş tarihi", reset:"Sıfırla", apply:"Uygula",
    failureProb: "Arıza Olasılığı", predictedIssue: "Tahmin Edilen Sorun", estimatedTime: "Tahmini Zaman", viewDetails: "Detayları Gör",
    avgFleetHealth: "Ortalama Filo Sağlığı", highRiskRobots: "Yüksek Şiddetli Robotlar", predFailures: "Tahmini Arızalar", modelAccuracy: "Model Doğruluğu",
    topFailurePred: "Robot Bazlı Head Çıktıları",
    legendActual:"Gerçek", legendPredicted:"Tahmin", predictedFailure:"Tahmin Edilen Arıza",
    riskCritical:"Kritik Risk", riskHigh:"Yüksek Risk", riskMedium:"Orta Risk", riskLow:"Düşük Risk",
    allRobots:"Tüm Robotlar", allComponents:"Tüm Bileşenler", allStatuses:"Tüm Durumlar", allFaultTypes:"Tüm Arıza Tipleri",
    last7Days:"Son 7 Gün", last30Days:"Son 30 Gün", last90Days:"Son 90 Gün",
    totalFaults:"Toplam Arıza", ofNRobots:"toplam {0} robottan",
    showingNofM:"{2} {3}, {0}-{1} arası gösteriliyor",
    sensorTrend:"Sensör Anomali Trendi", faultDist:"Arıza Dağılımı", faultFreq:"Zaman İçinde Arıza Sıklığı (Son 6 Ay)",
    fhColDateTime:"Tarih & Saat", fhColRobotId:"Robot ID", fhColErrorId:"Hata ID", fhColFaultType:"Arıza Tipi",
    fhColDiagnosed:"Teşhis Edilen Sorun (Loglardan)", fhColDowntime:"Duraklama Süresi",
    fhColResolution:"Çözüm Durumu", fhColActions:"İşlemler",
    clearFilters:"Filtreleri Temizle",
    catBrush:"Fırça/Motor Sorunları", catBattery:"Pil & Güç",
    catNav:"Navigasyon Sistemi", catVacuum:"Süpürge/Temizlik", catOther:"Diğer",
    resolved:"Çözüldü", inProgress:"Devam Ediyor",
    head1Label:"Anlık Arıza Tespiti", head2Label:"Arıza Şiddeti",
    head3Label:"Gelecek Öngörüsü", head4Label:"Arıza Süresi",
    metricAccuracy:"Doğruluk", metricAuc:"AUC-ROC", metricMae:"Ortalama Hata",
    headTip1: "Anlık arıza tespitinin doğruluk metrikleri. Accuracy = doğru tahmin oranı, F1 = precision + recall dengesi, AUC = modelin arızalı/sağlıklı ayırt etme gücü. Değer yüksek olursa iyi.",
    headTip2: "Arıza şiddeti sınıflandırması (Event / Warning / Error / Fatal — 4 seviye) doğruluğu.",
    headTip3: "7 günlük arıza olasılık tahmininin AUC-ROC değeri. 1.0 = kusursuz, 0.5 = rastgele tahmin.",
    headTip4: "Arıza süresi regresyon head'inin MAE (Ortalama Mutlak Hata) değeri, saat cinsinden. Daha düşük daha iyi — tahmin edilen zamanın gerçek değerden ortalama sapması.",
  }
};
let currentLang = localStorage.getItem("lang") || "en";
let currentTheme = localStorage.getItem("theme") || "light";
function t(key){ return (I18N[currentLang] && I18N[currentLang][key]) || I18N.en[key] || key; }
function tf(key, ...args){ return t(key).replace(/\{(\d+)\}/g, (_, i) => args[i] ?? ""); }
function locale(){ return currentLang === "tr" ? "tr-TR" : "en-US"; }
function applyLanguage(lang){
  currentLang = (I18N[lang] ? lang : "en");
  localStorage.setItem("lang", currentLang);
  document.documentElement.lang = currentLang;
  document.querySelectorAll("[data-i18n]").forEach(el => { el.textContent = t(el.dataset.i18n); });
  document.querySelectorAll("[data-i18n-ph]").forEach(el => { el.placeholder = t(el.dataset.i18nPh); });
  document.querySelectorAll(".seg-btn[data-lang]").forEach(b => b.classList.toggle("active", b.dataset.lang === currentLang));
  // Translate the sentinel ("All XXX") dropdown options. Only options whose
  // value attribute is "" are touched -- real robot IDs / fault names keep
  // their literal text.
  const sentinelMap = [
    ["statusFilter", "allStatuses"],
    ["faultFilter", "allFaultTypes"],
    ["fhFaultFilter", "allFaultTypes"],
    ["fhRobotFilter", "allRobots"],
    ["predRobotFilter", "allRobots"],
    ["predCategoryFilter", "allComponents"],
    ["degrCategoryFilter", "allComponents"],
  ];
  sentinelMap.forEach(([id, key]) => {
    const sel = document.getElementById(id);
    if (!sel) return;
    const sentinel = sel.querySelector('option[value=""]');
    if (sentinel) sentinel.textContent = t(key);
  });
  updatePredictionWindowLabel();
  // Category option labels (Brush Motor Issues / Battery & Power / ...)
  const catKeyMap = {
    "Brush Motor Issues":"catBrush","Battery & Power":"catBattery",
    "Navigation System":"catNav","Vacuum System":"catVacuum","Other":"catOther",
  };
  const statusCodeMap = {
    "FAULTED":"statusFAULTED","MAINTENANCE":"statusMAINTENANCE",
    "MONITOR":"statusMONITOR","OPERATIONAL":"statusOPERATIONAL",
  };
  const legacyStatusMap = {
    "Critical":"statusCritical","Warning":"statusWarning",
    "Normal":"statusNormal","Unknown":"statusUnknown",
  };
  ["statusFilter"].forEach(id=>{
    const sel = document.getElementById(id);
    if (!sel) return;
    [...sel.options].forEach(o=>{
      if (!o.value) return;
      const key = statusCodeMap[o.value];
      if (key) o.textContent = t(key);
    });
  });
  ["degrCategoryFilter","predCategoryFilter"].forEach(id=>{
    const sel = document.getElementById(id);
    if (!sel) return;
    [...sel.options].forEach(o=>{
      const k = catKeyMap[o.value];
      if (k) o.textContent = t(k);
    });
  });
  // Model head select: "Head 1 · Anlık Arıza Tespiti" etc.
  const mhs = document.getElementById("modelHeadSelect");
  if (mhs){
    [...mhs.options].forEach(o=>{
      const key = "head"+o.value+"Label";
      o.textContent = `Head ${o.value} · ${t(key)}`;
    });
  }
  // Re-render the model head card so the hint (label · metric) follows the lang.
  if (state && state.pred && state.pred.heads) renderModelHeadCard();
  if (state && state.pred && state.pred.topFailureItems?.length) renderTopFailures();
}
function setLanguage(lang){ applyLanguage(lang); reloadCurrentPage(); refreshNotificationBadge(); }
function applyTheme(theme){
  currentTheme = (theme === "dark" ? "dark" : "light");
  localStorage.setItem("theme", currentTheme);
  document.body.classList.toggle("dark", currentTheme === "dark");
  document.querySelectorAll(".seg-btn[data-theme]").forEach(b => b.classList.toggle("active", b.dataset.theme === currentTheme));
}
function setTheme(theme){ applyTheme(theme); }
function toggleSettings(ev){
  ev.stopPropagation();
  const p = document.getElementById("settingsPopover");
  p.hidden = !p.hidden;
}

const FAULT_COLORS = ["#3b82f6","#10b981","#f59e0b","#a855f7","#94a3b8"];
const CAT_COLORS = {
  // English canonical keys — what the old _category_for_error_type returned.
  "Brush Motor Issues":"#3b82f6","Battery & Power":"#10b981","Navigation System":"#f59e0b",
  "Vacuum System":"#a855f7","Other":"#94a3b8",
  // Old localized aliases from the previous TR build.
  "Fırça/Motor Sorunları":"#3b82f6","Pil & Güç":"#10b981","Navigasyon Sistemi":"#f59e0b",
  "Süpürge/Temizlik":"#a855f7","Diğer":"#94a3b8",
  // Categories actually emitted by the live backend on the deployed
  // server (already in Turkish). Each gets a distinct hue so the
  // stacked bar chart isn't a wall of grey.
  "Navigasyon":"#3b82f6",         // navigation -> blue
  "Harita/Durum":"#f59e0b",       // map / localization -> amber
  "Sensör Kaybı":"#a855f7",       // sensor loss -> purple
  "Hareket":"#22c55e",            // motion / drive -> green
  "Temizlik":"#06b6d4",           // cleaning -> cyan
  "Donanım/İletişim":"#ef4444",   // hw / comm -> red
  "Güç/Batarya":"#10b981",        // power / battery -> emerald
  "Bilinmiyor":"#94a3b8",         // unknown -> slate
};
// Reverse lookup the canonical category code for a possibly-translated label.
function canonicalCategory(label){
  const m = {
    "Fırça/Motor Sorunları":"Brush Motor Issues","Pil & Güç":"Battery & Power",
    "Navigasyon Sistemi":"Navigation System","Süpürge/Temizlik":"Vacuum System","Diğer":"Other",
    "Navigasyon":"Navigation System","Harita/Durum":"Navigation System",
    "Hareket":"Brush Motor Issues","Temizlik":"Vacuum System",
    "Güç/Batarya":"Battery & Power","Donanım/İletişim":"Other","Bilinmiyor":"Other",
    "Sensör Kaybı":"Other",
  };
  return m[label] || label;
}
// Reverse lookup the canonical category code for a possibly-translated label.
const state = {
  page:1, pageSize:5, search:"", status:"All Statuses", faultType:"All Fault Types",
  fh:{page:1, pageSize:8, search:"", robot:"All Robots", fault_type:"All Fault Types", status:"All Statuses", start_date:"", end_date:""},
  pred:{category:"", robot:"All Robots", windowDays:7, head:"1", heads:null, topVisible:5, topFailureItems:[]},
  // global date filter shared across dashboard/predictions/fault-history
  range:{ start:"", end:"", extentStart:"", extentEnd:"" },
  notifications:{ items:[], dismissedAt:null },
};
let charts = {};
const FIXED_PREDICTION_WINDOW_DAYS = 7;

document.addEventListener("DOMContentLoaded", () => {
  applyTheme(currentTheme);
  applyLanguage(currentLang);
  bindNav(); bindDashboardUI(); bindFaultHistoryUI(); bindPredictionsUI();
  loadRuntime();
  const initial = (location.hash || "#dashboard").replace("#","");
  navigate(initial);
});

function bindNav(){ document.querySelectorAll('[data-page]').forEach(btn=>btn.addEventListener('click',()=>navigate(btn.dataset.page))); window.addEventListener('hashchange',()=>navigate((location.hash||'#dashboard').slice(1), false)); }
function navigate(page, updateHash=true){
  document.querySelectorAll(".page").forEach(s=>s.classList.remove("active"));
  const target = document.getElementById(`page-${page}`);
  if (!target){ navigate("dashboard"); return; }
  target.classList.add("active");
  document.querySelectorAll("[data-page]").forEach(b=>b.classList.toggle("active", b.dataset.page===page));
  if (updateHash) location.hash = page;
  document.getElementById("sidebar").classList.remove("open");
  if (page==="dashboard")     loadDashboard();
  if (page==="fault-history") loadFaultHistory();
  if (page==="predictions")   loadPredictions();
}
function toggleSidebar(){ document.getElementById("sidebar").classList.toggle("open"); }

function bindDashboardUI(){
  document.getElementById("robotSearch").addEventListener("input", debounce(e=>{
    state.search = e.target.value.trim(); state.page=1; loadRobots();
  }, 300));
  document.getElementById("statusFilter").addEventListener("change", e=>{
    state.status=e.target.value; state.page=1; loadRobots();
  });
  document.getElementById("faultFilter").addEventListener("change", e=>{
    state.faultType=e.target.value; state.page=1; loadRobots();
  });
  document.getElementById("anomalyRobotSelect").addEventListener("change", e=>{
    loadAnomalyTrend(e.target.value || null);
  });
}

async function loadDashboard(){
  await Promise.all([loadStats(), loadAnomalyTrend(), loadFaultDistribution(), loadFilterOptions()]);
  await loadRobots();
  populateAnomalyRobotSelect();
}

async function loadStats(){
  try{
    const d = await fetchJson(`/api/stats${rangeQuery()}`);
    setStatCard("active", d.active_robots, tf("ofNRobots", d.active_robots.total));
    setStatCard("critical", d.critical_alerts, t("requireAttention"));
    setStatCard("fleet", d.fleet_health, t("healthyRobots"), true);
    if (d.range?.start && d.range?.end){
      if (!state.range.extentStart){ state.range.extentStart = d.range.start; state.range.extentEnd = d.range.end; }
      updateDateLabel(d.range.start, d.range.end);
    }
  }catch(e){ console.error(e); }
}

function rangeQuery(prefix="?"){
  const parts = [];
  if (state.range.start) parts.push("start_date=" + encodeURIComponent(state.range.start));
  if (state.range.end)   parts.push("end_date="   + encodeURIComponent(state.range.end));
  return parts.length ? prefix + parts.join("&") : "";
}

function updateDateLabel(startIso, endIso){
  const label = state.range.start || state.range.end
    ? `${state.range.start || formatRange(startIso)} - ${state.range.end || formatRange(endIso)}`
    : `${formatRange(startIso)} - ${formatRange(endIso)}`;
  ["dateRange","fhDateRange"].forEach(id=>{
    const el = document.getElementById(id);
    if (el) el.textContent = label + " ▾";
  });
  updatePredictionWindowLabel();
}

function predictionWindowText(days){
  const n = Number(days) || 7;
  if (n === 7) return t("next7Days");
  return currentLang === "tr" ? `Sonraki ${n} Gün` : `Next ${n} Days`;
}

function updatePredictionWindowLabel(){
  const btn = document.getElementById("predDateRange");
  if (btn) btn.textContent = predictionWindowText(FIXED_PREDICTION_WINDOW_DAYS) + " ▾";
  const pill = document.getElementById("predWindowPillValue");
  if (pill) pill.textContent = predictionWindowText(FIXED_PREDICTION_WINDOW_DAYS);
}

async function loadRuntime(){
  try{
    const d = await fetchJson('/api/model-runtime');
    state.runtime=d; renderRuntime(d);
  }catch(e){
    document.getElementById('runtimeStrip').innerHTML =
      `<div class="runtime-item warn"><div class="k">Model runtime</div><div class="v">${escapeHtml(String(e))}</div></div>`;
  }
}
function renderRuntime(d){
  const source = d.engine_kind === 'lstm_v2_inference'
    ? 'LSTM V2 inference'
    : (d.engine_kind === 'local_head_model' ? 'Local head model' : 'Dataset target replay');
  const short = d.git_commit ? d.git_commit.slice(0,10) : 'unavailable';
  const weightCls = d.weights_available ? 'good' : 'warn';
  const artifact = d.weights_available ? 'artifacts present' : 'artifacts missing';
  document.getElementById('runtimeStrip').innerHTML = `
    <div class="runtime-item"><div class="k">Model repo</div><div class="v" title="${escapeAttr(d.repo_url)}">${escapeHtml(d.repo_url)}</div></div>
    <div class="runtime-item"><div class="k">Commit</div><div class="v code">${escapeHtml(short)}</div></div>
    <div class="runtime-item ${weightCls}"><div class="k">Model artifacts</div><div class="v">${artifact}</div></div>
    <div class="runtime-item"><div class="k">Prediction source</div><div class="v">${source}</div></div>`;
}

const STAT_CARD_IDS = {
  active:   {value:"activeRobotsValue", sub:"activeRobotsSub", trend:"activeRobotsTrend", bar:"activeRobotsBar"},
  critical: {value:"criticalValue", sub:null, trend:"criticalTrend", bar:"criticalBar"},
  fleet:    {value:"fleetValue", sub:null, trend:"fleetTrend", bar:"fleetBar"},
};
function setStatCard(prefix, data, subText, percent=false){
  const ids = STAT_CARD_IDS[prefix] || {
    value: prefix + "Value", sub: prefix + "Sub", trend: prefix + "Trend", bar: prefix + "Bar",
  };
  const value = data?.value ?? 0;
  const display = percent ? `${Number(value).toFixed(1)}%` : value;
  const valueEl = document.getElementById(ids.value);
  if (valueEl) valueEl.textContent = display;
  const sub = ids.sub ? document.getElementById(ids.sub) : null;
  if (sub) sub.textContent = subText;
  const bar = document.getElementById(ids.bar);
  if (bar) bar.style.width = `${Math.max(0, Math.min(100, Number(value) || 0))}%`;
  renderTrend(ids.trend, data?.delta_pct);
}

function renderTrend(elId, delta){
  const el = document.getElementById(elId);
  if (!el) return;
  if (delta === null || delta === undefined){
    el.innerHTML = "";
    return;
  }
  const d = delta ?? 0;
  const arrow = d>0 ? "↑" : d<0 ? "↓" : "→";
  const cls = d>0 ? "badge-up" : d<0 ? "badge-down" : "badge-flat";
  el.innerHTML =
    `<span class="badge ${cls}">${arrow} ${Math.abs(d).toFixed(1)}%</span><span class="trend-sub">${t("vsPrev")}</span>`;
}

async function loadAnomalyTrend(robotId=null){
  try{
    const params = new URLSearchParams();
    if (robotId) params.set("robot_id", robotId);
    if (state.range.start) params.set("start_date", state.range.start);
    if (state.range.end) params.set("end_date", state.range.end);
    const d = await fetchJson(`/api/anomaly-trend?${params}`);
    const labels = d.points.map(p=>formatShortDate(p.date));
    const values = d.points.map(p=>p.score);
    const ctx = document.getElementById("anomalyChart").getContext("2d");
    charts.anomalyChart?.destroy();
    charts.anomalyChart = new Chart(ctx, {
      type:"line",
      data:{labels, datasets:[{label:t("sensorTrend"), data:values, borderColor:"#2563eb", backgroundColor:"rgba(37,99,235,.08)", fill:true, borderWidth:2, tension:.3, pointRadius:2}]},
      options:{responsive:true, maintainAspectRatio:false, plugins:{legend:{display:false}},
        scales:{x:{grid:{display:false}, ticks:{color:"#94a3b8"}}, y:{beginAtZero:true, max:100, grid:{color:"#eef2f7"}, ticks:{color:"#94a3b8"}}}}
    });
  }catch(e){ console.error(e); }
}

async function loadFaultDistribution(){
  try{
    const d = await fetchJson(`/api/fault-distribution${rangeQuery()}`);
    renderFaultDonut(d);
  }catch(e){ console.error(e); }
}

function renderFaultDonut({items, total}){
  const ctx = document.getElementById("faultChart").getContext("2d");
  const labels = items.map(x=>x.label);
  const values = items.map(x=>x.count);
  const colors = items.map((_,i)=>FAULT_COLORS[i % FAULT_COLORS.length]);
  charts.faultChart?.destroy();
  charts.faultChart = new Chart(ctx, {
    type:"doughnut",
    data:{labels, datasets:[{data:values, backgroundColor:colors, borderWidth:0}]},
    options:{responsive:true, maintainAspectRatio:false, cutout:"70%",
      plugins:{legend:{display:false}, tooltip:{callbacks:{label:c=>`${c.label}: ${c.parsed}`}}} },
    plugins:[{id:"center", beforeDraw(chart){
      const {ctx, chartArea:{left,right,top,bottom}} = chart;
      const cx=(left+right)/2, cy=(top+bottom)/2;
      ctx.save(); ctx.fillStyle="#0f172a"; ctx.font="700 22px Inter, sans-serif";
      ctx.textAlign="center"; ctx.textBaseline="middle";
      ctx.fillText(total.toLocaleString(), cx, cy-8);
      ctx.fillStyle="#94a3b8"; ctx.font="500 11px Inter, sans-serif";
      ctx.fillText(t("totalFaults"), cx, cy+12);
      ctx.restore();
    }}],
  });
  document.getElementById("faultLegend").innerHTML = items.map((it,i)=>`
    <li><span><span class="dot" style="background:${colors[i]}"></span>${escapeHtml(it.label)}</span>
      <span><span class="value">${it.count}</span> <span class="pct">(${it.pct}%)</span></span></li>`).join("");
}

let filterOptionsLoaded = false;
// Build <option> markup. The FIRST item is treated as a sentinel ("All XXX")
// and given value="" so applyLanguage can rewrite its textContent without
// changing the filter value the backend sees. The other options get an
// explicit value attribute too so applyLanguage may translate the displayed
// text (e.g. category names) without disturbing the backend filter key.
function asOptions(items){
  return items.map((s,i)=>{
    const v = i===0 ? "" : s;
    return `<option value="${escapeAttr(v)}">${escapeHtml(s)}</option>`;
  }).join("");
}
async function loadFilterOptions(){
  if (filterOptionsLoaded){ applyLanguage(currentLang); return; }
  try{
    const d = await fetchJson("/api/filter-options");
    document.getElementById("statusFilter").innerHTML    = asOptions(d.robot_statuses);
    document.getElementById("faultFilter").innerHTML     = asOptions(d.fault_types);
    document.getElementById("fhRobotFilter").innerHTML   = asOptions(d.robots);
    document.getElementById("fhFaultFilter").innerHTML   = asOptions(d.fault_types);
    document.getElementById("predRobotFilter").innerHTML = asOptions(d.robots);
    document.getElementById("predCategoryFilter").innerHTML = asOptions(d.categories);
    document.getElementById("degrCategoryFilter").innerHTML = asOptions(d.categories);
    filterOptionsLoaded = true;
    applyLanguage(currentLang); // translate the freshly-injected sentinel options
  }catch(e){ console.error(e); }
}

async function loadRobots(){
  const params = new URLSearchParams({page:state.page, page_size:state.pageSize});
  if (state.search) params.set("search", state.search);
  if (state.status) params.set("status", state.status);
  if (state.faultType) params.set("fault_type", state.faultType);
  if (state.range.start) params.set("start_date", state.range.start);
  if (state.range.end)   params.set("end_date", state.range.end);
  try{
    const d = await fetchJson(`/api/robots?${params}`);
    renderRobotTable(d);
    renderPagination("pagination","paginationInfo", d, p=>{state.page=p; loadRobots();}, "robots");
  }catch(e){
    document.getElementById("robotTableBody").innerHTML =
      `<tr><td colspan="8" class="empty">Failed to load: ${escapeHtml(String(e))}</td></tr>`;
  }
}

function renderRobotTable({items}){
  const body = document.getElementById("robotTableBody");
  if (!items.length){ body.innerHTML = `<tr><td colspan="8" class="empty">${t("noRobots")}</td></tr>`; return; }
  body.innerHTML = items.map(r=>{
    const prob = Math.round(r.fault_probability ?? r.confidence ?? 0);
    const fc = Math.round(r.seven_day_forecast ?? 0);
    const code = r.status_code || "OPERATIONAL";
    const cls = code.toLowerCase();
    let pc = "green"; if (prob>=60) pc="red"; else if (prob>=30) pc="amber";
    let fcCls = "green"; if (fc>=60) fcCls="red"; else if (fc>=30) fcCls="amber";
    const errs = (r.active_errors || []);
    return `
      <tr>
        <td>
          <div class="robot-id-cell">
            <div class="robot-thumb">${robotSvg()}</div>
            <div>
              <div class="robot-id">${escapeHtml(shortenId(r.robot_id))}</div>
              <div class="robot-area">${escapeHtml(r.area || "")}</div>
            </div>
          </div>
        </td>
        <td><span class="status ${cls}">${escapeHtml(t("status"+code) || code)}</span></td>
        <td>
          <div class="confidence">
            <span class="confidence-num">${prob}%</span>
            <div class="confidence-bar"><div class="confidence-fill ${pc}" style="width:${prob}%"></div></div>
          </div>
        </td>
        <td><span class="sev-pill ${escapeAttr(r.severity || 'Event')}">${escapeHtml(r.severity || "Event")}</span></td>
        <td>
          <div class="confidence">
            <span class="confidence-num">${fc}%</span>
            <div class="confidence-bar"><div class="confidence-fill ${fcCls}" style="width:${fc}%"></div></div>
          </div>
        </td>
        <td>
          <div class="err-chips">
            ${errs.length ? errs.map(e=>`<span class="err-chip" title="${escapeAttr(e)}">${escapeHtml(e)}</span>`).join("")
              : `<span class="err-chip empty">—</span>`}
          </div>
        </td>
        <td>${formatLong(r.last_updated)}</td>
        <td class="action-col">
          <button class="btn-secondary" onclick="openRobotModal('${escapeAttr(r.robot_id)}')">${t("viewDetails")}</button>
        </td>
      </tr>`;
  }).join("");
}
async function populateAnomalyRobotSelect(){
  try{
    const d = await fetchJson("/api/robots?page=1&page_size=30");
    const ids = [...new Set(d.items.map(r=>r.robot_id))].slice(0,30);
    document.getElementById("anomalyRobotSelect").innerHTML =
      `<option value="">${escapeHtml(t("allRobots"))}</option>` +
      ids.map(id=>`<option value="${escapeAttr(id)}">${escapeHtml(shortenId(id))}</option>`).join("");
  }catch{}
}

function bindFaultHistoryUI(){
  document.getElementById("fhSearch").addEventListener("input", debounce(e=>{
    state.fh.search = e.target.value.trim(); state.fh.page = 1; loadFhList();
  }, 250));
  document.getElementById("fhRobotFilter").addEventListener("change", e=>{
    state.fh.robot = e.target.value; state.fh.page = 1; loadFhList();
  });
  document.getElementById("fhFaultFilter").addEventListener("change", e=>{
    state.fh.fault_type = e.target.value; state.fh.page = 1; loadFhList();
  });
  const startEl = document.getElementById("fhStartDate");
  const endEl   = document.getElementById("fhEndDate");
  startEl.addEventListener("change", e=>{
    state.fh.start_date = e.target.value; state.fh.page = 1;
    // Constrain the end-date input so it cannot be set earlier than the
    // chosen start. If the user already picked an end before this start,
    // clear it so the request doesn't fight the now-invalid value.
    endEl.min = e.target.value || "";
    if (endEl.value && e.target.value && endEl.value < e.target.value){
      endEl.value = "";
      state.fh.end_date = "";
    }
    loadFaultHistory();
  });
  endEl.addEventListener("change", e=>{
    state.fh.end_date = e.target.value; state.fh.page = 1;
    // Symmetric guard: the start input can't be later than the chosen end.
    startEl.max = e.target.value || "";
    loadFaultHistory();
  });
}

function clearFhFilters(){
  state.fh = {page:1, pageSize:8, search:"", robot:"", fault_type:"", status:"", start_date:"", end_date:""};
  document.getElementById("fhSearch").value = "";
  document.getElementById("fhRobotFilter").value = "";
  document.getElementById("fhFaultFilter").value = "";
  const sd = document.getElementById("fhStartDate");
  const ed = document.getElementById("fhEndDate");
  sd.value = ""; sd.removeAttribute("max");
  ed.value = ""; ed.removeAttribute("min");
  loadFaultHistory();
}

async function loadFaultHistory(){
  await Promise.all([loadFilterOptions(), loadFhFrequency(), loadFhList()]);
}

async function loadFhFrequency(){
  try{
    const d = await fetchJson(`/api/fault-history/frequency${rangeQuery()}`);
    // Format the month labels using the active locale so months render
    // as "Şub 2026" in TR and "Feb 2026" in EN.
    const labels = (d.labels_iso || d.labels || []).map(iso =>
      new Date(iso).toLocaleDateString(locale(), {month:"short", year:"numeric"})
    );
    // Resolve the bar colour through the canonical category code, so a
    // translated dataset label still maps to the right palette entry.
    const datasets = d.datasets.map(ds => {
      const key = canonicalCategory(ds.label);
      const color = CAT_COLORS[key] || CAT_COLORS[ds.label] || "#94a3b8";
      return {
        label: ds.label,
        data: ds.data,
        backgroundColor: color,
        hoverBackgroundColor: color,
        borderRadius: 4,
        borderSkipped: false,
      };
    });
    const ctx = document.getElementById("freqChart").getContext("2d");
    charts.freqChart?.destroy();
    charts.freqChart = new Chart(ctx, {
      type:"bar", data:{labels, datasets},
      options:{ responsive:true, maintainAspectRatio:false,
        plugins:{legend:{display:false}},
        scales:{ x:{stacked:true, grid:{display:false}, ticks:{color:"#94a3b8"}},
                 y:{stacked:true, grid:{color:"#eef2f7"}, ticks:{color:"#94a3b8"}} } }
    });
  }catch(e){ console.error(e); }
}

async function loadFhList(){
  const p = new URLSearchParams({page:state.fh.page, page_size:state.fh.pageSize});
  if (state.fh.search) p.set("search", state.fh.search);
  if (state.fh.robot) p.set("robot", state.fh.robot);
  if (state.fh.fault_type) p.set("fault_type", state.fh.fault_type);
  if (state.fh.status) p.set("status", state.fh.status);
  if (state.fh.start_date) p.set("start_date", state.fh.start_date);
  if (state.fh.end_date) p.set("end_date", state.fh.end_date);
  try{
    const d = await fetchJson(`/api/fault-history/list?${p}`);
    renderFhTable(d);
    renderPagination("fhPagination","fhPaginationInfo", d, x=>{state.fh.page=x; loadFhList();}, "faults");
  }catch(e){
    document.getElementById("fhTableBody").innerHTML =
      `<tr><td colspan="8" class="empty">Failed to load: ${escapeHtml(String(e))}</td></tr>`;
  }
}

const CAT_I18N = {
  "Brush Motor Issues":"catBrush","Battery & Power":"catBattery",
  "Navigation System":"catNav","Vacuum System":"catVacuum","Other":"catOther",
};
function renderFhTable({items}){
  const body = document.getElementById("fhTableBody");
  if (!items.length){ body.innerHTML = `<tr><td colspan="8" class="empty">${t("noFaults")}</td></tr>`; return; }
  body.innerHTML = items.map(it=>{
    const catKey = canonicalCategory(it.category);
    const catColor = CAT_COLORS[catKey] || CAT_COLORS[it.category] || "#94a3b8";
    const resCls = it.resolution==="Resolved" ? "resolved" : "in-progress";
    const resLabel = it.resolution==="Resolved" ? t("resolved") : t("inProgress");
    const catLabel = t(CAT_I18N[catKey] || CAT_I18N[it.category]) || it.category;
    const errorId = it.error_id || "—";
    return `
      <tr>
        <td><div style="font-weight:600">${formatDate(it.task_time)}</div>
            <div style="font-size:11.5px;color:var(--text-mute)">${formatTimeOnly(it.task_time)}</div></td>
        <td><div class="robot-id-cell"><div class="robot-thumb">${robotSvg()}</div>
              <div class="robot-id">${escapeHtml(shortenId(it.robot_id))}</div></div></td>
        <td><span class="mono-id" title="${escapeAttr(errorId)}">${escapeHtml(errorId)}</span></td>
        <td><span class="cat-dot" style="background:${catColor}"></span>${escapeHtml(catLabel)}</td>
        <td><div class="fault-name">${escapeHtml(it.diagnosed_issue)}</div>
            <div class="fault-detail">${escapeHtml(it.fault_type_raw)}</div></td>
        <td>${escapeHtml(it.downtime)}</td>
        <td><span class="status ${resCls}">${escapeHtml(resLabel)}</span></td>
        <td class="action-col">
          <button class="icon-btn" title="View" onclick="openRobotModal('${escapeAttr(it.robot_id)}')">👁</button>
          <button class="icon-btn" title="Download">⬇</button>
        </td>
      </tr>`;
  }).join("");
}

function bindPredictionsUI(){
  document.getElementById("degrCategoryFilter").addEventListener("change", e=>{
    state.pred.category = e.target.value;
    state.pred.topVisible = 5;
    const top = document.getElementById("predCategoryFilter");
    if (top && [...top.options].some(o=>o.value===e.target.value)) top.value = e.target.value;
    loadDegradation(); loadPredStats(); loadTopFailures();
  });
  document.getElementById("predRobotFilter").addEventListener("change", e=>{
    state.pred.robot = e.target.value;  // "" = all
    state.pred.topVisible = 5;
    loadHeatmap(); loadDegradation(); loadPredStats(); loadTopFailures();
  });
  document.getElementById("predCategoryFilter").addEventListener("change", e=>{
    const v = e.target.value;
    state.pred.topVisible = 5;
    if (v){  // concrete category chosen
      state.pred.category = v;
      const inner = document.getElementById("degrCategoryFilter");
      if (inner && [...inner.options].some(o=>o.value===v)) inner.value = v;
      loadDegradation();
    }
    loadPredStats(); loadTopFailures();
  });
  document.getElementById("modelHeadSelect").addEventListener("change", e=>{
    state.pred.head = e.target.value;
    state.pred.topVisible = 5;
    renderModelHeadCard();
    loadTopFailures();
  });
}

async function loadPredictions(){
  state.pred.windowDays = FIXED_PREDICTION_WINDOW_DAYS;
  updatePredictionWindowLabel();
  await Promise.all([loadFilterOptions(), loadHeatmap(), loadDegradation(), loadPredStats(), loadTopFailures()]);
}

async function loadHeatmap(){
  try{
    const params = new URLSearchParams();
    params.set("days", FIXED_PREDICTION_WINDOW_DAYS);
    if (state.pred.robot) params.set("robot_id", state.pred.robot);
    const d = await fetchJson(`/api/predictions/heatmap?${params}`);
    const grid = document.getElementById("heatmapGrid");
    if (!d.robot_ids.length || !d.weeks.length){
      grid.innerHTML = `<div class="empty">${t("noRobots")}</div>`; return;
    }
    // Column grid: first column = robot label, remaining N columns = one per week.
    // Cells render in the top rows; the last row holds the date labels.
    // Each foot label gets a small tick line going up (::before) so the
    // columns read like the user's reference mock-up.
    const cols = `120px repeat(${d.weeks.length}, 1fr)`;
    const rows = d.robot_ids.map((rid,i)=>{
      const cells = d.weeks.map((w, j)=>{
        const v = d.grid[i][j];
        return `<div class="hcell" title="${escapeAttr(w)}: ${v}%">
                  <span class="fill" style="background:${riskColor(v)}"></span>
                </div>`;
      }).join("");
      return `<div class="hrow" style="grid-template-columns:${cols}">
                <div class="rlabel" title="${escapeAttr(rid)}">${escapeHtml(shortenId(rid))}</div>
                ${cells}
              </div>`;
    }).join("");
    const footCells = d.weeks.map(w => `<div class="hlabel">${escapeHtml(predictionAxisLabel(w))}</div>`).join("");
    const foot = `<div class="hrow foot" style="grid-template-columns:${cols}">
                    <div class="empty-slot"></div>
                    ${footCells}
                  </div>`;
    grid.innerHTML = rows + foot;
  }catch(e){ console.error(e); }
}

function predictionAxisLabel(label){
  return /^Next\s+\d+d$/i.test(String(label || "")) ? t("forecastLabel") : label;
}

function riskColor(v){
  if (v>=60) return "#ef4444";
  if (v>=45) return "#f97316";
  if (v>=30) return "#f59e0b";
  if (v>=15) return "#fde047";
  return "#86efac";
}

async function loadDegradation(){
  try{
    const params = new URLSearchParams();
    if (state.pred.category) params.set("category", state.pred.category);
    if (state.pred.robot) params.set("robot_id", state.pred.robot);
    params.set("days", FIXED_PREDICTION_WINDOW_DAYS);
    const d = await fetchJson(`/api/predictions/degradation?${params}`);
    renderDegradation("lstmChart", d, "#3b82f6", false);
  }catch(e){ console.error(e); }
}

function renderDegradation(canvasId, d, color, useRf){
  charts[canvasId]?.destroy();
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext("2d");
  const predData = useRf ? d.rf_pred : d.lstm_pred;
  const actualColor = color || "#2563eb";
  const predictedColor = "#f97316";
  if (typeof Chart === "undefined"){
    renderDegradationCanvas(canvas, d.labels || [], d.actual || [], predData || [], actualColor, predictedColor);
    return;
  }
  charts[canvasId] = new Chart(ctx,{
    type:"line",
    data:{ labels:d.labels,
      datasets:[
        { label: t("legendActual") || "Actual",
          data: d.actual,
          borderColor: actualColor,
          backgroundColor: "transparent",
          borderWidth: 2.5,
          tension: 0.35,
          pointRadius: 0,
          pointHoverRadius: 4,
          pointHitRadius: 8,
          spanGaps: false },
        { label: t("legendPredicted") || "Predicted",
          data: predData,
          borderColor: predictedColor,
          backgroundColor: "transparent",
          borderWidth: 2.5,
          borderDash: [6, 4],
          tension: 0.35,
          pointRadius: 0,
          pointHoverRadius: 4,
          pointHitRadius: 8,
          spanGaps: true },
      ]
    },
    options:{
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: true,
          position: "top",
          align: "end",
          labels: {
            boxWidth: 24,
            boxHeight: 0,
            usePointStyle: false,
            font: { size: 11, weight: "600" },
            // Render each legend swatch as a line segment that mirrors the
            // dataset's borderDash, so the legend matches what's drawn.
            generateLabels(chart){
              return chart.data.datasets.map((ds, i) => ({
                text: ds.label,
                strokeStyle: ds.borderColor,
                fillStyle: ds.borderColor,
                lineWidth: 2,
                lineDash: ds.borderDash || [],
                hidden: !chart.isDatasetVisible(i),
                datasetIndex: i,
              }));
            },
          },
        },
        tooltip: { backgroundColor: "#1f2937", padding: 10 },
      },
      scales: {
        x: { grid:{display:false}, ticks:{color:"#94a3b8", font:{size:10}} },
        y: { beginAtZero:true, max:100, grid:{color:"#eef2f7"}, ticks:{color:"#94a3b8", font:{size:10}} },
      },
    }
  });
}

function renderDegradationCanvas(canvas, labels, actual, predicted, actualColor, predictedColor){
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(320, Math.floor((rect.width || canvas.parentElement?.clientWidth || 640) * dpr));
  const height = Math.max(220, Math.floor((rect.height || canvas.parentElement?.clientHeight || 300) * dpr));
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, width, height);
  ctx.save();
  ctx.scale(dpr, dpr);

  const w = width / dpr;
  const h = height / dpr;
  const pad = {top: 34, right: 18, bottom: 34, left: 42};
  const plotW = Math.max(1, w - pad.left - pad.right);
  const plotH = Math.max(1, h - pad.top - pad.bottom);
  const xFor = i => pad.left + (labels.length <= 1 ? 0 : (i * plotW / (labels.length - 1)));
  const yFor = v => pad.top + plotH - (Math.max(0, Math.min(100, Number(v) || 0)) * plotH / 100);

  ctx.strokeStyle = "#eef2f7";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#94a3b8";
  ctx.font = "10px Inter, system-ui, sans-serif";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  [0, 25, 50, 75, 100].forEach(v => {
    const y = yFor(v);
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(w - pad.right, y);
    ctx.stroke();
    ctx.fillText(String(v), pad.left - 8, y);
  });

  function drawLine(values, stroke, dashed){
    ctx.save();
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 2.5;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.setLineDash(dashed ? [6, 4] : []);
    let started = false;
    values.forEach((value, i) => {
      if (value === null || value === undefined || Number.isNaN(Number(value))){
        if (!dashed) started = false;
        return;
      }
      const x = xFor(i);
      const y = yFor(value);
      if (!started){
        ctx.beginPath();
        ctx.moveTo(x, y);
        started = true;
      } else {
        ctx.lineTo(x, y);
      }
    });
    if (started) ctx.stroke();
    ctx.restore();
  }

  drawLine(actual, actualColor, false);
  drawLine(predicted, predictedColor, true);

  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  const labelStep = Math.max(1, Math.ceil(labels.length / 6));
  labels.forEach((label, i) => {
    if (i % labelStep === 0 || i === labels.length - 1) ctx.fillText(label, xFor(i), h - pad.bottom + 12);
  });

  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillStyle = actualColor;
  ctx.fillRect(w - 150, 13, 24, 2);
  ctx.fillStyle = "#334155";
  ctx.font = "11px Inter, system-ui, sans-serif";
  ctx.fillText(t("legendActual") || "Actual", w - 120, 14);
  ctx.strokeStyle = predictedColor;
  ctx.setLineDash([6, 4]);
  ctx.beginPath();
  ctx.moveTo(w - 68, 14);
  ctx.lineTo(w - 44, 14);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillText(t("legendPredicted") || "Predicted", w - 38, 14);
  ctx.restore();
}

async function loadPredStats(){
  try{
    const params = new URLSearchParams({days: FIXED_PREDICTION_WINDOW_DAYS});
    if (state.pred.robot) params.set("robot_id", state.pred.robot);
    // Forward the top-filter category so High Severity / Predicted Failures
    // respond to the dropdown choice. Backend just ignores the param when
    // it's not a recognised category.
    const selCat = document.getElementById("predCategoryFilter")?.value;
    if (selCat) params.set("category", selCat);
    const d = await fetchJson(`/api/predictions/stats?${params}`);
    setPredCard("predHighRisk","predHighRiskTrend","predHighRiskBar", d.high_risk.value, d.high_risk.delta_pct, Math.min(100, d.high_risk.value*12));
    setPredCard("predFail","predFailTrend","predFailBar", d.predicted_fail.value, d.predicted_fail.delta_pct, Math.min(100, d.predicted_fail.value*12));
    state.pred.heads = d.model_heads || null;
    renderModelHeadCard();
  }catch(e){ console.error(e); }
}

function renderModelHeadCard(){
  const heads = state.pred.heads;
  if (!heads){
    setPredCard("predAcc","predAccTrend","predAccBar", "—", 0, 0);
    return;
  }
  const cur = heads[state.pred.head] || heads["1"];
  const headName = t(cur.label_key) || cur.label_key;
  const numeric = cur.value == null
    ? "—"
    : (cur.unit === "%" ? `${cur.value}%` : `${cur.value} ${cur.unit}`);
  // metric_text comes back as "Accuracy / F1 / AUC · %99.4 / %94.8 / %99.9"
  // — split it so the values can become the headline of the card and the
  // metric NAMES move into the small sub line.
  let metricNames = "", metricValues = "";
  if (cur.metric_text){
    const idx = cur.metric_text.indexOf(" · ");
    if (idx > -1){
      metricNames = cur.metric_text.slice(0, idx);
      metricValues = cur.metric_text.slice(idx + 3);
    } else {
      metricValues = cur.metric_text;
    }
  }
  const headline = metricValues || numeric;
  const barPct = cur.bar_pct ?? (cur.unit === "%" ? cur.value : 0);
  setPredCard("predAcc","predAccTrend","predAccBar", headline, null, barPct);

  // Title row: "Model Accuracy · <head name> (ⓘ)"
  const headNameEl = document.getElementById("predHeadName");
  if (headNameEl) headNameEl.textContent = headName;

  // Sub line under the big value: "29 robots · Accuracy / F1 / AUC"
  const hint = document.getElementById("predHeadHint");
  if (hint){
    const parts = [];
    if (numeric && numeric !== "—") parts.push(numeric);
    if (metricNames) parts.push(metricNames);
    hint.textContent = parts.join(" · ");
  }

  // Info tooltip: explanation of what metrics mean for this head
  const bubble = document.getElementById("predHeadInfoBubble");
  if (bubble){
    const tipKey = "headTip" + state.pred.head;
    bubble.textContent = t(tipKey) || "";
  }
}

function setPredCard(valId, trendId, barId, val, delta, barPct){
  document.getElementById(valId).textContent = val;
  renderTrend(trendId, delta);
  document.getElementById(barId).style.width = `${Math.max(0, Math.min(100, barPct))}%`;
}

async function loadTopFailures(){
  try{
    const params = new URLSearchParams({ days: 7, limit: 200 });
    if (state.pred.robot) params.set("robot_id", state.pred.robot);
    if (state.pred.head) params.set("head", state.pred.head);
    const selCat = document.getElementById("predCategoryFilter")?.value;
    if (selCat) params.set("category", selCat);
    const d = await fetchJson(`/api/predictions/top-failures?${params}`);
    state.pred.topFailureItems = d.items || [];
    renderTopFailures();
  }catch(e){ console.error(e); }
}

function renderTopFailures(){
  const root = document.getElementById("predCardsContainer");
  const wrap = document.getElementById("predLoadMoreWrap");
  const info = document.getElementById("predLoadMoreInfo");
  const items = state.pred.topFailureItems || [];
  if (!items.length){
    root.innerHTML = `<div class="empty">No predictions available</div>`;
    if (wrap) wrap.hidden = true;
    return;
  }
  const visible = Math.min(state.pred.topVisible || 5, items.length);
  root.innerHTML = items.slice(0, visible).map(renderTopFailureCard).join("");
  if (wrap){
    wrap.hidden = visible >= items.length;
    if (info) info.textContent = tf("showingPredictions", visible, items.length);
  }
}

function renderTopFailureCard(it){
  const rk = it.risk_level || "Low";
  const riskLabel = t("risk" + rk) || it.risk_level;
  const value = it.value == null ? "—" : `${it.value}${it.unit === "%" ? "%" : " " + it.unit}`;
  const estimate = it.estimated_time_label || formatLong(it.estimated_time);
  return `
    <div class="pred-card ${rk}">
      <div class="head">
        <div class="thumb">${robotSvg()}</div>
        <div>
          <div class="id">${escapeHtml(shortenId(it.robot_id))}</div>
          <div class="area">${escapeHtml(it.area)}</div>
        </div>
      </div>
      <div class="prob-row">
        <span class="prob" style="color:${rk==='Critical'?'var(--red)':rk==='High'?'var(--amber)':rk==='Medium'?'#a16207':'var(--green)'}">${escapeHtml(value)}</span>
        <span class="risk-pill ${rk}">${escapeHtml(riskLabel)}</span>
      </div>
      <div><div class="label-sm">${escapeHtml(it.value_label || t("failureProb"))}</div></div>
      <div><div class="label-sm">${t("predictedIssue")}</div><div class="issue">${escapeHtml(it.predicted_issue)}</div></div>
      <div><div class="label-sm">${t("estimatedTime")}</div><div class="time">${escapeHtml(estimate)}</div></div>
      <button class="btn-view" onclick="openRobotModal('${escapeAttr(it.robot_id)}')">${t("viewDetails")}</button>
    </div>`;
}

function loadMoreTopFailures(){
  state.pred.topVisible = Math.min((state.pred.topVisible || 5) + 5, (state.pred.topFailureItems || []).length);
  renderTopFailures();
}

function renderPagination(navId, infoId, {total, page, page_size}, onGo, noun){
  const totalPages = Math.max(1, Math.ceil(total/page_size));
  const startN = total===0 ? 0 : (page-1)*page_size + 1;
  const endN = Math.min(total, page*page_size);
  const nounLabel = t(noun) || noun;
  document.getElementById(infoId).textContent = tf("showingNofM", startN, endN, total, nounLabel);
  const nav = document.getElementById(navId);
  const html = [`<button ${page===1?"disabled":""} data-go="${page-1}">‹</button>`];
  for (const p of pageWindow(page, totalPages, 5)){
    html.push(`<button class="${p===page?"active":""}" data-go="${p}">${p}</button>`);
  }
  html.push(`<button ${page===totalPages?"disabled":""} data-go="${page+1}">›</button>`);
  nav.innerHTML = html.join("");
  nav.querySelectorAll("button[data-go]").forEach(btn=>{
    btn.addEventListener("click", ()=>{
      const t = parseInt(btn.dataset.go, 10);
      if (!isNaN(t) && t>=1 && t<=totalPages) onGo(t);
    });
  });
}

function pageWindow(current, total, span=5){
  const half = Math.floor(span/2);
  let start = Math.max(1, current-half);
  let end = Math.min(total, start+span-1);
  start = Math.max(1, end-span+1);
  const out=[]; for(let i=start;i<=end;i++) out.push(i); return out;
}

async function openRobotModal(robotId){
  const backdrop = document.getElementById("modalBackdrop");
  const body = document.getElementById("modalBody");
  document.getElementById("modalTitle").textContent = `Robot ${shortenId(robotId)}`;
  body.innerHTML = "Loading…"; backdrop.hidden = false;
  try{
    const d = await fetchJson(`/api/robot/${encodeURIComponent(robotId)}`);
    body.innerHTML = `
      <div class="meta-grid">
        <div><span class="k">Robot ID</span><span class="v">${escapeHtml(d.robot_id)}</span></div>
        <div><span class="k">Product</span><span class="v">${escapeHtml(d.product_code || "-")}</span></div>
        <div><span class="k">SN</span><span class="v">${escapeHtml(d.sn || "-")}</span></div>
        <div><span class="k">MAC</span><span class="v">${escapeHtml(d.mac || "-")}</span></div>
        <div><span class="k">Software</span><span class="v">${escapeHtml(d.soft_version || "-")}</span></div>
        <div><span class="k">OS</span><span class="v">${escapeHtml(d.os_version || "-")}</span></div>
      </div>
      <h4>Recent Logs (latest 20)</h4>
      <table>
        <thead><tr><th>Time</th><th>Type</th><th>Detail</th><th>Level</th><th>Ratio</th></tr></thead>
        <tbody>
          ${d.recent_logs.map(l=>`
            <tr>
              <td>${formatLong(l.task_time)}</td>
              <td>${escapeHtml(l.error_type || "")}</td>
              <td>${escapeHtml(l.error_detail || "")}</td>
              <td>${escapeHtml(l.error_level || "")}</td>
              <td>${(l.hourly_ratio*100).toFixed(1)}%</td>
            </tr>`).join("")}
        </tbody>
      </table>`;
  }catch(e){ body.innerHTML = `<p>Failed: ${escapeHtml(String(e))}</p>`; }
}
function closeModal(){ document.getElementById("modalBackdrop").hidden = true; }

document.addEventListener("keydown", e=>{
  if (e.key==="Escape"){ closeModal(); closeDatePicker(); closeNotifPanel(); closeDegradationInfo(); }
});
document.getElementById("modalBackdrop").addEventListener("click", e=>{ if (e.target.id==="modalBackdrop") closeModal(); });
document.addEventListener("click", e=>{
  // outside-click closing for popovers
  const dp = document.getElementById("datePopover");
  const np = document.getElementById("notifPanel");
  const sp = document.getElementById("settingsPopover");
  const ip = document.getElementById("degrInfoPopover");
  if (!dp.hidden && !e.target.closest(".date-picker") && !e.target.closest("#datePopover")) dp.hidden = true;
  if (!np.hidden && !e.target.closest(".bell") && !e.target.closest("#notifPanel")) np.hidden = true;
  if (sp && !sp.hidden && !e.target.closest(".user-card") && !e.target.closest("#settingsPopover")) sp.hidden = true;
  if (ip && !ip.hidden && !e.target.closest(".info-trigger") && !e.target.closest("#degrInfoPopover")) ip.hidden = true;
});

function toggleDegradationInfo(ev){
  ev.stopPropagation();
  closeDatePicker();
  closeNotifPanel();
  const ip = document.getElementById("degrInfoPopover");
  if (ip) ip.hidden = !ip.hidden;
}
function closeDegradationInfo(){
  const ip = document.getElementById("degrInfoPopover");
  if (ip) ip.hidden = true;
}

// =========================================================================
// Date picker
// =========================================================================
function toggleDatePicker(ev, triggerId){
  ev.stopPropagation();
  const pop = document.getElementById("datePopover");
  if (!pop.hidden){ pop.hidden = true; return; }
  closeNotifPanel();
  const predictionMode = triggerId === "predDateRange";
  const title = pop.querySelector("h4");
  if (title) title.textContent = predictionMode ? t("predictionWindow") : t("selectRange");
  pop.querySelectorAll(".future-range").forEach(el => { el.style.display = predictionMode ? "block" : "none"; });
  pop.querySelectorAll(".past-range").forEach(el => {
    if (predictionMode) {
      el.style.display = "none";
    } else if (el.classList.contains("row") || el.classList.contains("actions")) {
      el.style.display = "flex";
    } else {
      el.style.display = "block";
    }
  });
  // Position the popover under the trigger button
  const btn = document.getElementById(triggerId);
  const r = btn.getBoundingClientRect();
  pop.style.top = (window.scrollY + r.bottom + 6) + "px";
  pop.style.right = (window.innerWidth - r.right) + "px";
  pop.style.left = "auto";
  // populate inputs
  document.getElementById("rangeStartInput").value = state.range.start || isoDate(state.range.extentStart);
  document.getElementById("rangeEndInput").value   = state.range.end   || isoDate(state.range.extentEnd);
  pop.hidden = false;
}
function closeDatePicker(){ document.getElementById("datePopover").hidden = true; }

function isoDate(iso){
  if (!iso) return "";
  const d = new Date(iso);
  if (isNaN(d.getTime())) return "";
  return d.toISOString().slice(0, 10);
}

function applyRange(){
  state.range.start = document.getElementById("rangeStartInput").value || "";
  state.range.end   = document.getElementById("rangeEndInput").value   || "";
  closeDatePicker();
  reloadCurrentPage();
}
function resetRange(){
  state.range.start = ""; state.range.end = "";
  document.getElementById("rangeStartInput").value = isoDate(state.range.extentStart);
  document.getElementById("rangeEndInput").value   = isoDate(state.range.extentEnd);
  reloadCurrentPage();
}
function applyQuickRange(value){
  if (value === "all"){
    state.range.start = ""; state.range.end = "";
  } else {
    const days = Number(value);
    const end = state.range.extentEnd ? new Date(state.range.extentEnd) : new Date();
    const start = new Date(end); start.setDate(start.getDate() - days + 1);
    state.range.start = isoDate(start.toISOString());
    state.range.end   = isoDate(end.toISOString());
  }
  closeDatePicker();
  reloadCurrentPage();
}

function applyFutureRange(days){
  state.pred.windowDays = FIXED_PREDICTION_WINDOW_DAYS;
  closeDatePicker();
  updatePredictionWindowLabel();
  loadPredictions();
}

function reloadCurrentPage(){
  const active = document.querySelector(".page.active");
  if (!active) return;
  const id = active.id.replace("page-", "");
  if (id === "dashboard")     loadDashboard();
  if (id === "fault-history") loadFaultHistory();
  if (id === "predictions")   loadPredictions();
}

// =========================================================================
// Notifications
// =========================================================================
function toggleNotifPanel(ev){
  ev.stopPropagation();
  const np = document.getElementById("notifPanel");
  if (!np.hidden){ np.hidden = true; return; }
  closeDatePicker();
  // position near the bell that was clicked
  const btn = ev.currentTarget.getBoundingClientRect();
  np.style.top = (window.scrollY + btn.bottom + 6) + "px";
  np.style.right = (window.innerWidth - btn.right - 16) + "px";
  np.style.left = "auto";
  np.hidden = false;
  loadNotifications();
}
function closeNotifPanel(){ document.getElementById("notifPanel").hidden = true; }

async function loadNotifications(){
  const list = document.getElementById("notifList");
  list.innerHTML = `<li class="notif-empty">Loading…</li>`;
  try{
    const d = await fetchJson("/api/notifications?limit=20");
    state.notifications.items = d.items;
    updateBellBadge(d.items.length);
    if (!d.items.length){ list.innerHTML = `<li class="notif-empty">${t("noAlerts")}</li>`; return; }
    list.innerHTML = d.items.map(it=>`
      <li>
        <div class="sev ${it.severity}"></div>
        <div>
          <div class="title">${escapeHtml(it.title)}</div>
          <div class="body" title="${escapeAttr(it.detail || '')}">${escapeHtml(shortenId(it.robot_id))} · ${escapeHtml(it.detail || it.level || '')}</div>
        </div>
        <div class="when">${formatLong(it.task_time)}</div>
      </li>`).join("");
  }catch(e){
    list.innerHTML = `<li class="notif-empty">Failed: ${escapeHtml(String(e))}</li>`;
  }
}

function updateBellBadge(n){
  ["bellBadge","bellBadge2","bellBadge3"].forEach(id=>{
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = n > 99 ? "99+" : String(n);
    el.style.display = n ? "" : "none";
  });
}

function markAllRead(){
  updateBellBadge(0);
  state.notifications.dismissedAt = Date.now();
  closeNotifPanel();
}

// Refresh notification badge on first load (independent of page)
async function refreshNotificationBadge(){
  try{
    const d = await fetchJson("/api/notifications?limit=20");
    state.notifications.items = d.items;
    updateBellBadge(d.items.length);
  }catch{}
}

// Set the top-bar date label as soon as we know the data extent — this fires
// independently of whichever page is active, so the label never gets stuck on
// "Date range ▾" if the page-specific loader is slow or fails.
async function initDateLabel(){
  try{
    const d = await fetchJson("/api/stats");
    if (d.range?.start && d.range?.end){
      state.range.extentStart = d.range.start;
      state.range.extentEnd   = d.range.end;
      updateDateLabel(d.range.start, d.range.end);
    }
  }catch{}
}

document.addEventListener("DOMContentLoaded", ()=>{
  refreshNotificationBadge();
  initDateLabel();
});

async function fetchJson(url){
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}
function debounce(fn, ms){ let t; return (...a)=>{ clearTimeout(t); t=setTimeout(()=>fn(...a), ms); }; }
function escapeHtml(s){ return String(s).replace(/[&<>"']/g, c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c])); }
function escapeAttr(s){ return escapeHtml(s); }
function shortenId(id){ if (!id) return ""; return id.length>12 ? "RC-"+id.slice(-8) : id; }
function formatRange(iso){ return new Date(iso).toLocaleDateString(locale(),{month:"short",day:"numeric",year:"numeric"}); }
function formatShortDate(iso){ return new Date(iso).toLocaleDateString(locale(),{month:"short",day:"numeric"}); }
function formatLong(iso){ if (!iso) return "—";
  return new Date(iso).toLocaleString(locale(),{month:"short",day:"numeric",year:"numeric",hour:"2-digit",minute:"2-digit",hour12:false}); }
function formatDate(iso){ if (!iso) return "—";
  return new Date(iso).toLocaleDateString(locale(),{month:"short",day:"numeric",year:"numeric"}); }
function formatTimeOnly(iso){ if (!iso) return "";
  return new Date(iso).toLocaleTimeString(locale(),{hour:"2-digit",minute:"2-digit",hour12:false}); }
function robotSvg(){
  return `<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
    <rect x="4" y="8" width="16" height="11" rx="2"></rect>
    <path d="M8 8V6a4 4 0 0 1 8 0v2"></path>
    <circle cx="9" cy="13" r="1"></circle><circle cx="15" cy="13" r="1"></circle></svg>`;
}
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
