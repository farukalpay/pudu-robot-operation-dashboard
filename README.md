# RoboClean Predictive Maintenance Dashboard

Single-file FastAPI dashboard for the Bahçeşehir University capstone project **Fault Prediction and Diagnostics from Robot Operation Logs**.

The dashboard now reads the published Hugging Face dataset snapshot directly instead of requiring the live PostgreSQL server.

## Data Source

Default dataset:

```text
Lightcap/pudu-robot-operation-logs-bau-capstone-2026
```

Hugging Face repository:

```text
https://huggingface.co/datasets/Lightcap/pudu-robot-operation-logs-bau-capstone-2026
```

DOI:

```text
10.57967/hf/8635
```

At startup, `app.py` downloads the Parquet files with `huggingface_hub`, registers them as DuckDB views, and keeps the existing API contract backed by SQL queries.

## Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open:

```text
http://127.0.0.1:8000
```

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `HF_DATASET_REPO` | `Lightcap/pudu-robot-operation-logs-bau-capstone-2026` | Hugging Face dataset repository |
| `HF_DATASET_REVISION` | `main` | Dataset revision/commit/branch |
| `DASHBOARD_DUCKDB_PATH` | `:memory:` | Optional DuckDB database path |

## API

| Endpoint | Purpose |
|---|---|
| `/api/health` | Confirms dataset-backed service status |
| `/api/stats` | Fleet and log summary metrics |
| `/api/anomaly-trend` | Time-bucketed hourly-ratio trend |
| `/api/fault-distribution` | Fault/error type distribution |
| `/api/robots` | Paginated latest robot status table |
| `/api/filter-options` | Robot IDs, fault types, levels, date range |
| `/api/logs` | Paginated raw log records |
| `/api/robot/{robot_id}` | Recent records for one robot |
| `/api/model-info` | Registered LSTM training metadata |

## Project References

- Dataset: https://huggingface.co/datasets/Lightcap/pudu-robot-operation-logs-bau-capstone-2026
- Dataset DOI: https://doi.org/10.57967/hf/8635
- Dashboard repository: https://github.com/farukalpay/pudu-robot-operation-dashboard
- Model training repository: https://github.com/DrGb24/pudu_bot_model_training/
- Dashboard implementation: Faruk Alpay `<faruk.alpay@bahcesehir.edu.tr>` and Nazım Alp Batu Kardaş `<alpbatu.kardas@bahcesehir.edu.tr>`
- Model training implementation: Buğra Kılıçtaş `<bugra.kilictas@bahcesehir.edu.tr>`
- Industrial Engineering contributors: Bilge Ece Şentürk `<bilgeece.senturk@bahcesehir.edu.tr>`, Serra Uysal `<serra.uysal@bahcesehir.edu.tr>`, Ayça Yeralp `<ayca.yeralp@bahcesehir.edu.tr>`
- Advisors: Assoc. Prof. Cemal Okan Şakar `<okan.sakar@eng.bau.edu.tr>` ([profile](https://akademik.bahcesehir.edu.tr/web/cemalokansakar/tr/index.html)); Assoc. Prof. Adnan Çorum `<adnan.corum@eng.bau.edu.tr>` ([profile](https://akademik.bahcesehir.edu.tr/web/adnancorum/tr/index.html))

## DOI

Use DOI `10.57967/hf/8635` when citing the Hugging Face dataset snapshot.
