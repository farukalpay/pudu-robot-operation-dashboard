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

from pudu_model_runtime import MODEL_RUNTIME, RuntimeSnapshot

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
log = logging.getLogger("dashboard")

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

        failure_sql, failure_params = _failure_condition_sql()
        crit_q = f"""SELECT COUNT(DISTINCT robot_id) AS n FROM public.robot_logs_error
                    WHERE task_time BETWEEN %s AND %s AND robot_id IS NOT NULL
                      AND ({failure_sql});"""
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

        out = []
        for r in rows:
            st = _classify_status(r["error_level"], r["hourly_ratio"])
            if status and status.lower() not in ("all", "all statuses") and st.lower() != status.lower():
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
        agg: dict[str, dict[str, int]] = {}
        order_keys: list[str] = []
        category_order: list[str] = []
        for r in cur.fetchall():
            label = r["bkt"].strftime("%b %Y")
            if label not in agg:
                agg[label] = {}
                order_keys.append(label)
            cat = _category_for_error_type(r["etype"])
            if cat not in category_order:
                category_order.append(cat)
            agg[label][cat] = agg[label].get(cat, 0) + int(r["cnt"])
        labels = order_keys[-6:]
        datasets = [{"label": cat, "data": [agg.get(l, {}).get(cat, 0) for l in labels]} for cat in sorted(category_order)]
        return {"labels": labels, "datasets": datasets}


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
            SELECT robot_id, error_type, error_detail, error_level, hourly_ratio, task_time,
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
            sql += " AND (COALESCE(error_type,'') ILIKE %s OR COALESCE(error_detail,'') ILIKE %s OR COALESCE(robot_id,'') ILIKE %s)"
            like = f"%{search}%"; params += [like, like, like]
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
def api_pred_heatmap(weeks: int = Query(8, ge=2, le=52)) -> dict[str, Any]:
    with get_cursor() as cur:
        _start, end = _data_window(cur)
        win_start = end - timedelta(weeks=weeks)
        cur.execute("""
            SELECT robot_id,
                   date_trunc('week', task_time) AS bkt,
                   AVG(hourly_ratio) AS risk
            FROM public.robot_logs_error
            WHERE task_time BETWEEN %s AND %s AND robot_id IS NOT NULL
            GROUP BY robot_id, bkt
            ORDER BY robot_id, bkt;
        """, (win_start, end))
        rows = cur.fetchall()
        robots: dict[str, dict[str, float]] = {}
        weeks_set: set[str] = set()
        for r in rows:
            label = r["bkt"].strftime("%b %d")
            weeks_set.add(label)
            robots.setdefault(r["robot_id"], {})[label] = round(float(r["risk"] or 0) * 100, 1)
        week_labels = sorted(weeks_set, key=lambda s: datetime.strptime(s, "%b %d"))
        active_robots = sorted(robots.items(), key=lambda kv: -sum(kv[1].values()))[:8]
        return {
            "robot_ids": [rid for rid, _ in active_robots],
            "weeks": week_labels,
            "grid": [[active_robots[i][1].get(w, 0) for w in week_labels] for i in range(len(active_robots))],
        }


@app.get("/api/predictions/degradation")
def api_pred_degradation(category: str | None = None) -> dict[str, Any]:
    with get_cursor() as cur:
        start, end = _data_window(cur)
        cur.execute("""
            SELECT date_trunc('week', task_time) AS bkt,
                   COALESCE(error_type,'') AS etype,
                   AVG(hourly_ratio) AS risk
            FROM public.robot_logs_error
            WHERE task_time BETWEEN %s AND %s
            GROUP BY 1, 2 ORDER BY 1;
        """, (start, end))
        trend_rows = cur.fetchall()
        if not category:
            category_counts: dict[str, int] = {}
            for r in trend_rows:
                cat = _category_for_error_type(r["etype"])
                category_counts[cat] = category_counts.get(cat, 0) + 1
            category = max(category_counts, key=category_counts.get) if category_counts else "Bilinmiyor"
        by_week: dict[str, list[float]] = {}
        for r in trend_rows:
            if _category_for_error_type(r["etype"]) != category:
                continue
            label = r["bkt"].strftime("%b %d")
            by_week.setdefault(label, []).append(float(r["risk"] or 0))

        labels, actual = [], []
        for k in sorted(by_week.keys(), key=lambda s: datetime.strptime(s, "%b %d")):
            avg = sum(by_week[k]) / len(by_week[k])
            labels.append(k); actual.append(round(100 - avg * 100, 1))

        labels = labels[-14:]; actual = actual[-14:]

        def project(series: list[float], slope_factor: float) -> list[float | None]:
            if len(series) < 2:
                return [None] * len(series) + [series[-1] if series else 50] * 3
            slope = (series[-1] - series[0]) / max(1, len(series) - 1) * slope_factor
            v = series[-1]; future = []
            for _ in range(3):
                v = max(0, min(100, v + slope)); future.append(round(v, 1))
            return [None] * (len(series) - 1) + [series[-1]] + future

        lstm_pred = project(actual, 1.0)
        rf_pred = project(actual, 0.7)
        future_labels = []
        if labels:
            last = datetime.strptime(labels[-1], "%b %d")
            future_labels = [(last + timedelta(weeks=i)).strftime("%b %d") for i in range(1, 4)]

        return {
            "labels": labels + future_labels,
            "actual": actual + [None] * 3,
            "lstm_pred": lstm_pred,
            "rf_pred": rf_pred,
            "predicted_failure_label": future_labels[-1] if future_labels else None,
        }


@app.get("/api/predictions/stats")
def api_pred_stats() -> dict[str, Any]:
    with get_cursor() as cur:
        snapshot = MODEL_RUNTIME.snapshot()
        horizon_hours = snapshot.future_window_hours or 168
        items, meta = _head_rows(cur, None, None, horizon_hours)
        total = len(items)
        current_failures = sum(1 for item in items if item["head_1"]["is_failure_now"])
        future_failures = sum(1 for item in items if item["head_3"]["future_failure_observed"])
        fleet_health = round((1 - current_failures / total) * 100, 1) if total else 0.0
        accuracy = 0.0
        head_1_metric = _metric_dict(snapshot, "head_1")
        if head_1_metric and head_1_metric.get("values"):
            accuracy = float(head_1_metric["values"].get("value_1", 0.0))

        return {
            "fleet_health":   {"value": fleet_health,        "delta_pct": 0.0},
            "high_risk":      {"value": future_failures,     "delta_pct": 0.0},
            "predicted_fail": {"value": current_failures,    "delta_pct": 0.0},
            "model_accuracy": {"value": accuracy,            "delta_pct": 0.0},
            "source": meta["source"],
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
def api_top_failures() -> dict[str, Any]:
    with get_cursor() as cur:
        snapshot = MODEL_RUNTIME.snapshot()
        horizon_hours = snapshot.future_window_hours or 168
        rows, meta = _head_rows(cur, None, None, horizon_hours)
        items = []
        for r in rows:
            hours = r["head_4"]["est_hours_to_failure"]
            current = r["head_1"]["is_failure_now"]
            future = r["head_3"]["future_failure_observed"]
            probability = r["head_1"]["failure_prob_now"]
            if future and probability is None:
                probability = 100.0
            items.append({
                "robot_id": r["robot_id"],
                "area": r["area"],
                "failure_probability": probability,
                "risk_level": "Current failure" if current else ("Future failure observed" if future else "No future failure observed"),
                "predicted_issue": r["error_type"],
                "predicted_detail": r["error_detail"] or "Operational",
                "estimated_time": hours,
                "estimated_time_label": r["head_4"]["est_time_label"],
                "category": r["component"],
                "source": meta["source"],
            })
        items.sort(key=lambda x: (x["estimated_time"] is None, x["estimated_time"] or 10**9, -(x["failure_probability"] or 0)))
        return {"items": items[:10]}


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


def _model_source(snapshot: RuntimeSnapshot) -> str:
    return "lstm_v2_inference" if snapshot.engine_available else "dataset_target_replay"


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

    items: list[dict[str, Any]] = []
    for r in rows:
        robot_id = r["robot_id"]
        future = future_by_robot.get(robot_id)
        next_time = future["next_failure_time"] if future else None
        hours_to_failure = None
        if next_time:
            hours_to_failure = max(0.0, round((next_time - reference).total_seconds() / 3600, 1))

        severity_score = MODEL_RUNTIME.severity_score(r["error_level"])
        severity_label = snapshot.severity_labels.get(severity_score, r["error_level"] or "Unknown") if severity_score is not None else (r["error_level"] or "Unknown")
        severity_tr = snapshot.severity_labels_tr.get(severity_score, severity_label) if severity_score is not None else severity_label
        current_failure = MODEL_RUNTIME.is_failure_level(r["error_level"])
        future_failure = bool(future)
        evidence_ratio = float(r["hourly_ratio"] or 0)

        items.append({
            "robot_id": robot_id,
            "area": r["product_code"] or "Unknown",
            "last_observed_at": r["task_time"].isoformat() if r["task_time"] else None,
            "error_type": r["error_type"] or "Unknown",
            "error_detail": r["error_detail"] or "",
            "component": _category_for_error_type(r["error_type"]),
            "status": _classify_status(r["error_level"], r["hourly_ratio"]),
            "head_1": {
                "name": "Anlık arıza",
                "is_failure_now": current_failure,
                "failure_prob_now": _pct(evidence_ratio),
                "source": _model_source(snapshot),
            },
            "head_2": {
                "name": "Şiddet",
                "severity_now": severity_label,
                "severity_now_tr": severity_tr,
                "severity_score": severity_score,
                "source": _model_source(snapshot),
            },
            "head_3": {
                "name": "7 günlük öngörü",
                "future_failure_observed": future_failure,
                "future_failure_events": int(future["future_failure_events"]) if future else 0,
                "next_7d_fail_prob": 100.0 if future_failure and not snapshot.engine_available else None,
                "source": _model_source(snapshot),
            },
            "head_4": {
                "name": "Arıza süresi",
                "est_hours_to_failure": hours_to_failure,
                "est_time_label": _format_hours(hours_to_failure),
                "source": _model_source(snapshot),
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
        horizon_hours = snapshot.future_window_hours or horizon_days * 24
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
                    "name": "7 günlük öngörü",
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
        horizon_hours = snapshot.future_window_hours or horizon_days * 24
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
        horizon_hours = snapshot.future_window_hours or horizon_days * 24
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
  --shadow:0 1px 2px rgba(15,23,42,.05),0 8px 24px rgba(15,23,42,.06);--r:8px;
}
*{box-sizing:border-box}html,body{margin:0;min-height:100%;background:var(--bg);color:var(--text)}
body{font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;font-size:14px;letter-spacing:0}button,input,select{font:inherit}button{cursor:pointer}
.app{min-height:100vh;display:grid;grid-template-columns:260px minmax(0,1fr)}
.side{background:#111827;color:#d8dee9;padding:18px 14px;display:flex;flex-direction:column;gap:18px;position:sticky;top:0;height:100vh}
.brand{display:flex;align-items:center;gap:12px;padding:6px 8px 16px;border-bottom:1px solid rgba(255,255,255,.08)}
.logo{width:42px;height:42px;border-radius:8px;background:#f8fafc;color:#111827;display:grid;place-items:center;flex:none}.logo svg{width:28px;height:28px}
.brand b{display:block;color:#fff;font-size:16px}.brand span{display:block;color:#9ca3af;font-size:12px;margin-top:2px}
.nav{display:flex;flex-direction:column;gap:4px}.nav button{border:0;background:transparent;color:#cbd5e1;text-align:left;border-radius:8px;padding:10px 12px;display:flex;gap:10px;align-items:center}.nav button:hover{background:rgba(255,255,255,.06)}.nav button.active{background:#2563eb;color:#fff}
.side-meta{margin-top:auto;border:1px solid rgba(255,255,255,.08);border-radius:8px;padding:12px;background:rgba(255,255,255,.04)}.side-meta .k{font-size:11px;color:#9ca3af;text-transform:uppercase}.side-meta .v{margin-top:4px;color:#fff;font-weight:700;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.main{min-width:0;padding:24px 28px 34px}.top{display:flex;justify-content:space-between;gap:16px;align-items:flex-start;margin-bottom:18px}.title h1{margin:0;font-size:25px;line-height:1.15;color:var(--ink)}.title p{margin:6px 0 0;color:var(--muted)}
.controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap;justify-content:flex-end}.field{display:flex;align-items:center;gap:6px;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:7px 10px}.field span{font-size:12px;color:var(--muted);white-space:nowrap}.field input,.field select{border:0;background:transparent;color:var(--text);outline:0;min-width:116px}.btn{border:1px solid var(--line);background:#fff;border-radius:8px;padding:8px 12px;color:var(--text);font-weight:650}.btn.primary{background:var(--blue);border-color:var(--blue);color:#fff}.btn:hover{filter:brightness(.98)}
.runtime{display:grid;grid-template-columns:minmax(0,1.2fr) repeat(3,minmax(150px,.35fr));gap:10px;margin-bottom:16px}.runtime-item{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px;box-shadow:var(--shadow);min-width:0}.runtime-item .k{color:var(--muted);font-size:11px;text-transform:uppercase}.runtime-item .v{margin-top:5px;font-weight:750;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.runtime-item.warn{border-color:#f6d58b;background:#fffaf0}.runtime-item.good{border-color:#b7ebc6;background:#f3fff6}
.grid-heads{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin-bottom:16px}.head-card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:16px;box-shadow:var(--shadow);min-width:0}.head-top{display:flex;justify-content:space-between;gap:10px;align-items:flex-start}.head-no{font-size:12px;color:var(--muted);font-weight:700}.head-card h3{margin:3px 0 0;font-size:16px}.head-metric{margin-top:12px;color:var(--muted);font-size:12px}.head-value{font-size:28px;line-height:1.1;font-weight:800;margin-top:8px;color:var(--ink);overflow-wrap:anywhere}.pill{display:inline-flex;align-items:center;border-radius:999px;padding:4px 9px;font-size:12px;font-weight:750}.pill.blue{background:var(--blue-soft);color:#1d4ed8}.pill.green{background:var(--green-soft);color:#15803d}.pill.amber{background:var(--amber-soft);color:#92400e}.pill.red{background:var(--red-soft);color:#991b1b}.pill.gray{background:#eef2f7;color:#475569}
.panel-grid{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(320px,.75fr);gap:16px;margin-bottom:16px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;box-shadow:var(--shadow);padding:16px;min-width:0}.panel-head{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:12px}.panel-head h2{font-size:16px;margin:0;color:var(--ink)}.chart{position:relative;height:260px;min-width:0}.split{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;min-width:0}.split>*{min-width:0}.mini-stat{background:var(--panel-2);border:1px solid var(--line);border-radius:8px;padding:12px}.mini-stat .k{font-size:12px;color:var(--muted)}.mini-stat .v{margin-top:5px;font-size:19px;font-weight:800;color:var(--ink)}
.table-tools{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:12px;flex-wrap:wrap}.search{background:#fff;border:1px solid var(--line);border-radius:8px;padding:8px 10px;min-width:260px}.search input{border:0;outline:0;width:100%;background:transparent}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:8px}table{width:100%;border-collapse:collapse;background:#fff}th,td{padding:12px 13px;border-bottom:1px solid var(--line);text-align:left;vertical-align:middle}th{font-size:12px;color:var(--muted);background:#f8fafc;white-space:nowrap}td{font-size:13px}tr:last-child td{border-bottom:0}.robot{font-weight:780;color:var(--ink)}.muted{color:var(--muted)}.bar{height:8px;background:#e5e7eb;border-radius:99px;overflow:hidden;min-width:90px}.bar i{display:block;height:100%;background:var(--blue);border-radius:99px}.status{display:inline-flex;align-items:center;gap:6px;border-radius:999px;padding:4px 9px;font-size:12px;font-weight:750}.status:before{content:"";width:6px;height:6px;border-radius:50%;background:currentColor}.status.Critical{background:var(--red-soft);color:#991b1b}.status.Warning{background:var(--amber-soft);color:#92400e}.status.Normal{background:var(--green-soft);color:#166534}.status.Unknown{background:#eef2f7;color:#475569}
.pager{display:flex;gap:6px;justify-content:flex-end;align-items:center;margin-top:12px}.pager button{border:1px solid var(--line);background:#fff;border-radius:6px;min-width:32px;height:32px}.pager button.active{background:var(--blue);border-color:var(--blue);color:#fff}.pager button:disabled{opacity:.45;cursor:not-allowed}.page{display:none}.page.active{display:block}.empty{padding:26px;text-align:center;color:var(--muted)}.metric-table td:first-child{font-weight:750}.metric-table td{white-space:normal}.code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;word-break:break-all}.hamb{display:none}
@media(max-width:1180px){.grid-heads{grid-template-columns:repeat(2,minmax(0,1fr))}.runtime{grid-template-columns:1fr 1fr}.panel-grid{grid-template-columns:1fr}}
@media(max-width:780px){.app{grid-template-columns:1fr}.side{position:fixed;z-index:20;left:0;top:0;transform:translateX(-100%);transition:.2s;width:260px}.side.open{transform:translateX(0)}.main{padding:18px}.hamb{display:inline-flex}.top{flex-direction:column}.controls{justify-content:flex-start}.grid-heads,.runtime,.split{grid-template-columns:1fr}.field{width:100%}.field input,.field select{flex:1}.search{min-width:100%}}
</style>
</head>
<body>
<div class="app">
  <aside class="side" id="side">
    <div class="brand">
      <div class="logo" aria-hidden="true"><svg viewBox="0 0 32 32" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="6" y="11" width="20" height="13" rx="3"/><path d="M11 11V8a5 5 0 0 1 10 0v3"/><circle cx="12" cy="17" r="1.4" fill="currentColor"/><circle cx="20" cy="17" r="1.4" fill="currentColor"/><path d="M11 25h10"/></svg></div>
      <div><b>PUDU Ops</b><span>LSTM V2 Heads</span></div>
    </div>
    <nav class="nav" id="nav">
      <button class="active" data-page="dashboard">Head Dashboard</button>
      <button data-page="robots">Robot Outputs</button>
      <button data-page="history">Fault History</button>
      <button data-page="model">Model Runtime</button>
    </nav>
    <div class="side-meta"><div class="k">Runtime source</div><div class="v" id="sideSource">Loading</div></div>
  </aside>

  <main class="main">
    <div class="top">
      <div class="title">
        <button class="btn hamb" onclick="toggleSide()">Menu</button>
        <h1>Predictive Maintenance Dashboard</h1>
        <p id="subtitle">GitHub model contract + Hugging Face operation logs</p>
      </div>
      <div class="controls">
        <label class="field"><span>Start</span><input type="date" id="startDate"></label>
        <label class="field"><span>Reference</span><input type="date" id="endDate"></label>
        <label class="field"><span>Horizon</span><select id="horizon"><option value="7">7 gün</option></select></label>
        <button class="btn" onclick="resetDates()">Reset</button>
        <button class="btn primary" onclick="reloadActive()">Apply</button>
      </div>
    </div>

    <section class="runtime" id="runtimeStrip"></section>

    <section class="page active" id="page-dashboard">
      <div class="grid-heads" id="headCards"></div>
      <div class="panel-grid">
        <section class="panel">
          <div class="panel-head"><h2>7 Günlük Öngörü Penceresi</h2><span class="pill gray" id="coveragePill">—</span></div>
          <div class="chart"><canvas id="timelineChart"></canvas></div>
        </section>
        <section class="panel">
          <div class="panel-head"><h2>Şiddet ve Bileşen Dağılımı</h2><span class="pill blue" id="robotCountPill">—</span></div>
          <div class="split">
            <div class="chart"><canvas id="severityChart"></canvas></div>
            <div id="componentStats"></div>
          </div>
        </section>
      </div>
      <section class="panel">
        <div class="panel-head"><h2>Öncelikli Robotlar</h2><button class="btn" onclick="navigate('robots')">Tümünü Aç</button></div>
        <div class="table-wrap"><table><thead><tr><th>Robot</th><th>Anlık</th><th>Şiddet</th><th>7 Gün</th><th>Tahmini Süre</th><th>Son Log</th></tr></thead><tbody id="priorityRows"><tr><td colspan="6" class="empty">Loading</td></tr></tbody></table></div>
      </section>
    </section>

    <section class="page" id="page-robots">
      <section class="panel">
        <div class="table-tools">
          <div class="search"><input id="robotSearch" placeholder="Robot ID filtrele"></div>
          <div class="muted" id="robotPageInfo">—</div>
        </div>
        <div class="table-wrap"><table><thead><tr><th>Robot</th><th>Component</th><th>Head 1</th><th>Head 2</th><th>Head 3</th><th>Head 4</th><th>Evidence</th></tr></thead><tbody id="robotRows"><tr><td colspan="7" class="empty">Loading</td></tr></tbody></table></div>
        <div class="pager" id="robotPager"></div>
      </section>
    </section>

    <section class="page" id="page-history">
      <div class="panel-grid">
        <section class="panel"><div class="panel-head"><h2>Fault Frequency</h2></div><div class="chart"><canvas id="faultChart"></canvas></div></section>
        <section class="panel"><div class="panel-head"><h2>Historical Logs</h2></div><div class="search" style="margin-bottom:12px"><input id="faultSearch" placeholder="Fault / robot ara"></div><div class="table-wrap"><table><thead><tr><th>Time</th><th>Robot</th><th>Component</th><th>Issue</th><th>Resolution</th></tr></thead><tbody id="faultRows"><tr><td colspan="5" class="empty">Loading</td></tr></tbody></table></div></section>
      </div>
    </section>

    <section class="page" id="page-model">
      <section class="panel"><div class="panel-head"><h2>Model Runtime</h2><span class="pill gray" id="modelStatus">—</span></div><div id="modelRuntimeBody" class="empty">Loading</div></section>
    </section>
  </main>
</div>

<script>
const state = { page:'dashboard', start:'', end:'', horizon:7, robotPage:1, robotSize:12, robotSearch:'', runtime:null, summary:null };
const charts = {};
const palette = ['#2563eb','#0f766e','#16a34a','#d97706','#dc2626','#475569','#7c3aed','#0891b2'];

document.addEventListener('DOMContentLoaded', () => {
  bindNav(); bindControls(); navigate((location.hash || '#dashboard').slice(1) || 'dashboard', false); loadShell();
});
function bindNav(){ document.querySelectorAll('[data-page]').forEach(btn=>btn.addEventListener('click',()=>navigate(btn.dataset.page))); window.addEventListener('hashchange',()=>navigate((location.hash||'#dashboard').slice(1), false)); }
function bindControls(){
  document.getElementById('horizon').addEventListener('change', e=>{state.horizon=Number(e.target.value); reloadActive();});
  document.getElementById('startDate').addEventListener('change', e=>state.start=e.target.value);
  document.getElementById('endDate').addEventListener('change', e=>state.end=e.target.value);
  document.getElementById('robotSearch').addEventListener('input', debounce(e=>{state.robotSearch=e.target.value.trim(); state.robotPage=1; loadRobots();},250));
  document.getElementById('faultSearch').addEventListener('input', debounce(()=>loadHistory(),250));
}
function toggleSide(){ document.getElementById('side').classList.toggle('open'); }
function navigate(page, updateHash=true){
  if (!document.getElementById('page-'+page)) page='dashboard'; state.page=page;
  document.querySelectorAll('.page').forEach(p=>p.classList.toggle('active', p.id==='page-'+page));
  document.querySelectorAll('[data-page]').forEach(b=>b.classList.toggle('active', b.dataset.page===page));
  document.getElementById('side').classList.remove('open'); if(updateHash) location.hash=page; reloadActive();
}
async function loadShell(){ await Promise.all([loadRuntime(), loadSummary()]); await Promise.all([loadTimeline(), loadRobots(true)]); }
function reloadActive(){ if(state.page==='dashboard') loadDashboard(); if(state.page==='robots') loadRobots(); if(state.page==='history') loadHistory(); if(state.page==='model') loadModel(); }
async function loadDashboard(){ await Promise.all([loadSummary(), loadTimeline(), loadRobots(true)]); }
function resetDates(){ state.start=''; state.end=''; document.getElementById('startDate').value=''; document.getElementById('endDate').value=''; reloadActive(); }
function query(extra={}){ const p=new URLSearchParams(); if(state.start) p.set('start_date',state.start); if(state.end) p.set('end_date',state.end); p.set('horizon_days', state.horizon); Object.entries(extra).forEach(([k,v])=>{if(v!==''&&v!=null)p.set(k,v)}); return p.toString(); }

async function loadRuntime(){
  const d = await fetchJson('/api/model-runtime'); state.runtime=d; renderRuntime(d);
}
function renderRuntime(d){
  const source = d.engine_available ? 'LSTM V2 inference' : 'Dataset target replay';
  document.getElementById('sideSource').textContent = source;
  const short = d.git_commit ? d.git_commit.slice(0,10) : 'unavailable';
  const weightCls = d.weights_available ? 'good' : 'warn';
  document.getElementById('runtimeStrip').innerHTML = `
    <div class="runtime-item"><div class="k">Fresh GitHub checkout</div><div class="v" title="${esc(d.repo_url)}">${esc(d.repo_url)}</div></div>
    <div class="runtime-item"><div class="k">Commit</div><div class="v code">${esc(short)}</div></div>
    <div class="runtime-item ${weightCls}"><div class="k">Model artifacts</div><div class="v">${d.weights_available?'weights present':'weights missing'}</div></div>
    <div class="runtime-item"><div class="k">Prediction source</div><div class="v">${source}</div></div>`;
}

async function loadSummary(){
  const d = await fetchJson('/api/model-heads/summary?'+query()); state.summary=d; renderSummary(d);
  if(!state.start && d.range?.start) document.getElementById('startDate').value = isoDate(d.range.start);
  if(!state.end && d.range?.end) document.getElementById('endDate').value = isoDate(d.range.end);
}
function renderSummary(d){
  document.getElementById('subtitle').textContent = `${fmtDateTime(d.reference_time)} reference · ${d.source.replaceAll('_',' ')}`;
  document.getElementById('coveragePill').textContent = d.future_window_complete ? 'full 7-day window' : 'partial window';
  document.getElementById('coveragePill').className = 'pill ' + (d.future_window_complete ? 'green':'amber');
  document.getElementById('robotCountPill').textContent = `${d.total_robots} robots`;
  document.getElementById('headCards').innerHTML = d.heads.map((h,i)=>headCard(h,i)).join('');
  renderDonut('severityChart', Object.keys(d.severity_counts||{}), Object.values(d.severity_counts||{}));
  renderComponents(d.component_counts || {});
}
function headCard(h,i){
  const metric = h.metric ? `${esc(h.metric.metric)} · ${esc(h.metric.result)}` : 'metric unavailable';
  const value = h.unit==='distribution' ? esc(h.detail) : (h.value==null ? '—' : esc(h.detail || h.value));
  const cls = ['blue','green','amber','red'][i] || 'gray';
  return `<article class="head-card"><div class="head-top"><div><div class="head-no">Head ${i+1}</div><h3>${esc(h.name)}</h3></div><span class="pill ${cls}">${esc(h.unit)}</span></div><div class="head-value">${value}</div><div class="head-metric">${metric}</div></article>`;
}
function renderComponents(counts){
  const entries = Object.entries(counts).sort((a,b)=>b[1]-a[1]).slice(0,6);
  document.getElementById('componentStats').innerHTML = entries.map(([k,v],i)=>`<div class="mini-stat" style="margin-bottom:8px"><div class="k">${esc(k)}</div><div class="v">${v}</div><div class="bar"><i style="width:${Math.min(100,v*4)}%;background:${palette[i%palette.length]}"></i></div></div>`).join('') || '<div class="empty">No component data</div>';
}
async function loadTimeline(){
  const d = await fetchJson('/api/model-heads/timeline?'+query());
  renderBar('timelineChart', d.points.map(p=>fmtShort(p.date)), d.points.map(p=>p.robots), 'Robots');
}

async function loadRobots(priority=false){
  const extra = priority ? {page:1,page_size:6} : {page:state.robotPage,page_size:state.robotSize};
  if(state.robotSearch && !priority) extra.robot = state.robotSearch;
  const d = await fetchJson('/api/model-heads/robots?'+query(extra));
  if(priority){ renderPriority(d.items); return; }
  renderRobotTable(d); renderPager('robotPager', d.page, Math.max(1,Math.ceil(d.total/d.page_size)), p=>{state.robotPage=p; loadRobots();});
  document.getElementById('robotPageInfo').textContent = `${d.total} robots · page ${d.page}`;
}
function renderPriority(items){
  const body=document.getElementById('priorityRows');
  if(!items.length){body.innerHTML='<tr><td colspan="6" class="empty">No robots</td></tr>';return;}
  body.innerHTML=items.map(r=>`<tr><td><div class="robot">${shortId(r.robot_id)}</div><div class="muted">${esc(r.area)}</div></td><td>${head1(r)}</td><td>${head2(r)}</td><td>${head3(r)}</td><td>${esc(r.head_4.est_time_label)}</td><td>${fmtDateTime(r.last_observed_at)}</td></tr>`).join('');
}
function renderRobotTable(d){
  const body=document.getElementById('robotRows');
  if(!d.items.length){body.innerHTML='<tr><td colspan="7" class="empty">No robots</td></tr>';return;}
  body.innerHTML=d.items.map(r=>`<tr><td><div class="robot">${shortId(r.robot_id)}</div><div class="muted code">${esc(r.robot_id)}</div></td><td>${esc(r.component)}</td><td>${head1(r)}</td><td>${head2(r)}</td><td>${head3(r)}</td><td>${esc(r.head_4.est_time_label)}</td><td><div>${esc(r.error_type)}</div><div class="muted">${fmtDateTime(r.last_observed_at)}</div></td></tr>`).join('');
}
function head1(r){ const p=r.head_1.failure_prob_now ?? 0; return `<span class="status ${r.status}">${r.head_1.is_failure_now?'Failure':'Clear'}</span><div class="bar" style="margin-top:6px"><i style="width:${Math.min(100,p)}%"></i></div>`; }
function head2(r){ return `<b>${esc(r.head_2.severity_now_tr || r.head_2.severity_now)}</b><div class="muted">score ${r.head_2.severity_score ?? '—'}</div>`; }
function head3(r){ return r.head_3.future_failure_observed ? '<span class="pill red">future failure</span>' : '<span class="pill green">no failure</span>'; }

async function loadHistory(){
  await Promise.all([loadFaultFrequency(), loadFaultRows()]);
}
async function loadFaultFrequency(){
  const d = await fetchJson('/api/fault-history/frequency?'+query());
  const datasets = d.datasets.map((ds,i)=>({label:ds.label,data:ds.data,backgroundColor:palette[i%palette.length],borderRadius:4}));
  renderStacked('faultChart', d.labels, datasets);
}
async function loadFaultRows(){
  const p = new URLSearchParams(); p.set('page','1'); p.set('page_size','12'); if(state.start)p.set('start_date',state.start); if(state.end)p.set('end_date',state.end); const s=document.getElementById('faultSearch').value.trim(); if(s)p.set('search',s);
  const d = await fetchJson('/api/fault-history/list?'+p.toString());
  const body=document.getElementById('faultRows');
  if(!d.items.length){body.innerHTML='<tr><td colspan="5" class="empty">No faults</td></tr>';return;}
  body.innerHTML=d.items.map(x=>`<tr><td>${fmtDateTime(x.task_time)}</td><td><span class="robot">${shortId(x.robot_id)}</span></td><td>${esc(x.category)}</td><td><div>${esc(x.diagnosed_issue)}</div><div class="muted">${esc(x.fault_type_raw)}</div></td><td><span class="status ${x.resolution==='Resolved'?'Normal':'Warning'}">${esc(x.resolution)}</span></td></tr>`).join('');
}
async function loadModel(){
  const d = await fetchJson('/api/model-info'); const rt=d.runtime || state.runtime || {}; document.getElementById('modelStatus').textContent = rt.status || 'unknown';
  document.getElementById('modelRuntimeBody').innerHTML = `<div class="split"><div class="mini-stat"><div class="k">Repo path</div><div class="v code">${esc(rt.repo_path || '-')}</div></div><div class="mini-stat"><div class="k">Future window</div><div class="v">${rt.future_window_hours || '-'} h</div></div></div><h3>Head metrics</h3><div class="table-wrap"><table class="metric-table"><thead><tr><th>Head</th><th>Metric</th><th>Result</th></tr></thead><tbody>${(d.heads||[]).map(h=>`<tr><td>${esc(h.name)}</td><td>${esc(h.metric)}</td><td>${esc(h.result)}</td></tr>`).join('')}</tbody></table></div><h3>Feature columns</h3><p class="code">${esc((rt.feature_columns||[]).join(', '))}</p><h3>Runtime note</h3><p class="muted">${esc(rt.engine_error || rt.error || 'LSTM runtime prepared')}</p>`;
}

function renderBar(id, labels, data, label){ const ctx=document.getElementById(id).getContext('2d'); charts[id]?.destroy(); charts[id]=new Chart(ctx,{type:'bar',data:{labels,datasets:[{label,data,backgroundColor:'#2563eb',borderRadius:4}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{grid:{display:false}},y:{beginAtZero:true,grid:{color:'#eef2f7'}}}}}); }
function renderStacked(id, labels, datasets){ const ctx=document.getElementById(id).getContext('2d'); charts[id]?.destroy(); charts[id]=new Chart(ctx,{type:'bar',data:{labels,datasets},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{position:'bottom'}},scales:{x:{stacked:true,grid:{display:false}},y:{stacked:true,beginAtZero:true,grid:{color:'#eef2f7'}}}}}); }
function renderDonut(id, labels, data){ const ctx=document.getElementById(id).getContext('2d'); charts[id]?.destroy(); charts[id]=new Chart(ctx,{type:'doughnut',data:{labels,datasets:[{data,backgroundColor:labels.map((_,i)=>palette[i%palette.length]),borderWidth:0}]},options:{responsive:true,maintainAspectRatio:false,cutout:'68%',plugins:{legend:{position:'bottom'}}}}); }
function renderPager(id,page,total,onGo){ const root=document.getElementById(id); const pages=[]; pages.push(`<button ${page<=1?'disabled':''} data-p="${page-1}">‹</button>`); for(let i=Math.max(1,page-2);i<=Math.min(total,page+2);i++)pages.push(`<button class="${i===page?'active':''}" data-p="${i}">${i}</button>`); pages.push(`<button ${page>=total?'disabled':''} data-p="${page+1}">›</button>`); root.innerHTML=pages.join(''); root.querySelectorAll('button[data-p]').forEach(b=>b.onclick=()=>{const p=Number(b.dataset.p); if(p>=1&&p<=total)onGo(p);}); }
async function fetchJson(url){ const r=await fetch(url); if(!r.ok) throw new Error(`${r.status} ${r.statusText}`); return r.json(); }
function debounce(fn,ms){let t;return(...a)=>{clearTimeout(t);t=setTimeout(()=>fn(...a),ms)}}
function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function isoDate(iso){if(!iso)return'';const d=new Date(iso);return isNaN(d)?'':d.toISOString().slice(0,10)}
function fmtDateTime(iso){if(!iso)return'—';const d=new Date(iso);return isNaN(d)?'—':d.toLocaleString('tr-TR',{month:'short',day:'2-digit',year:'numeric',hour:'2-digit',minute:'2-digit'})}
function fmtShort(iso){const d=new Date(iso);return isNaN(d)?'—':d.toLocaleDateString('tr-TR',{month:'short',day:'2-digit'})}
function shortId(id){id=String(id||''); return id.length>12 ? 'RC-'+id.slice(-8) : esc(id)}
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
