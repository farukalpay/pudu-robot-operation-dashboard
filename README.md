# RoboClean Predictive Maintenance Dashboard

Single-file FastAPI dashboard for the Bahçeşehir University capstone project
**Fault Prediction and Diagnostics from Robot Operation Logs**.

The dashboard reads the published Hugging Face dataset snapshot directly via
DuckDB — no live PostgreSQL connection required.

## Pages

| Hash route          | Description |
|---------------------|-------------|
| `/#dashboard`       | Fleet KPIs, anomaly trend, fault distribution donut, robot health table |
| `/#predictions`     | Fleet risk heatmap, LSTM & RF degradation projections, 4 KPI cards, top failure prediction cards |
| `/#fault-history`   | Monthly fault frequency (stacked bar), filterable historical fault log with date range / status / resolution |
| `/#model`           | Latest training-run metrics (accuracy/recall/precision/AUC) |

Sidebar items `Robot Monitoring`, `Maintenance Logs`, `Settings` are clickable
placeholder pages.

## Data Source

Default dataset (Hugging Face):

```
Lightcap/pudu-robot-operation-logs-bau-capstone-2026
```

URL: <https://huggingface.co/datasets/Lightcap/pudu-robot-operation-logs-bau-capstone-2026>

DOI: `10.57967/hf/8635`

At startup, `app.py` downloads the Parquet files with `huggingface_hub`,
registers them as DuckDB views, and serves the same API contract the previous
PostgreSQL-backed version exposed.

## Run Locally

```bash
pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:8000>.

The first launch downloads ~30 MB of Parquet snapshots; subsequent runs are
served from the local Hugging Face cache.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `HF_DATASET_REPO` | `Lightcap/pudu-robot-operation-logs-bau-capstone-2026` | Hugging Face dataset repository |
| `HF_DATASET_REVISION` | `main` | Dataset revision/commit/branch |
| `DASHBOARD_DUCKDB_PATH` | `:memory:` | Optional DuckDB database path |

## API

| Endpoint | Purpose |
|---|---|
| `/api/health` | Dataset status |
| `/api/stats` | Fleet KPI cards (active / critical / fleet health) |
| `/api/anomaly-trend` | Time-bucketed anomaly trend (auto day/week/month) |
| `/api/fault-distribution` | Top error types + Other |
| `/api/robots` | Paginated latest robot status table |
| `/api/robot/{robot_id}` | Recent 20 logs for one robot |
| `/api/filter-options` | Robot IDs, fault types, categories, statuses |
| `/api/model-info` | Registered LSTM training metadata |
| `/api/fault-history/frequency` | Monthly stacked fault counts by category |
| `/api/fault-history/list` | Paginated, filterable fault log |
| `/api/predictions/heatmap` | Robot × weekly risk grid |
| `/api/predictions/degradation` | Per-category health-score trace + projection |
| `/api/predictions/stats` | 4 prediction KPIs |
| `/api/predictions/top-failures` | Top 10 high-probability failures |

## Tech

- Backend: FastAPI + DuckDB over Hugging Face Parquet
- Frontend: Vanilla JS + Chart.js (single inlined HTML payload)
- Hash-based SPA routing (Dashboard / Predictions / Fault History / Model)
- Mobile responsive at 1100 / 880 / 760 / 520 px breakpoints

## File layout

```
dashboard/
├── app.py            # Everything: FastAPI + DuckDB loader + HTML/CSS/JS
├── requirements.txt
├── run.bat           # Windows convenience launcher
└── README.md
```

## Project References

- Dataset: <https://huggingface.co/datasets/Lightcap/pudu-robot-operation-logs-bau-capstone-2026>
- Dashboard repository: <https://github.com/farukalpay/pudu-robot-operation-dashboard>
- Model training repository: <https://github.com/DrGb24/pudu_bot_model_training/>
