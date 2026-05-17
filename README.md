# PUDU Robot Operation Dashboard

FastAPI + DuckDB dashboard for the PUDU robot predictive maintenance dataset.
The app now treats the model-training repository as the source of truth for the
LSTM V2 four-head contract.

## Runtime Model Source

On every app process start, the dashboard clones this repository into a fresh
temporary directory:

```
https://github.com/DrGb24/pudu_bot_model_training.git
```

The dashboard reads the model contract from that checkout:

- Head 1: `Anlık arıza`
- Head 2: `Şiddet`
- Head 3: `7 günlük öngörü`
- Head 4: `Arıza süresi`
- Feature columns, severity labels, failure levels, component labels, future
  window, and README metrics

If the cloned repo contains the V2 weight/config/scaler files under
`models/lstm_v2/`, the runtime attempts to load `LSTMInferenceV2`. The current
public repo does not include those weights, so the dashboard transparently runs
in `dataset_target_replay` mode: it reconstructs the four training targets from
the Hugging Face snapshot instead of pretending to have live model inference.

## Data Source

Default Hugging Face dataset:

```
Lightcap/pudu-robot-operation-logs-bau-capstone-2026
```

The Parquet files are downloaded through `huggingface_hub` and registered as
DuckDB views. No live PostgreSQL connection is required.

## Pages

| Hash route | Description |
|---|---|
| `/#dashboard` | Four LSTM V2 head cards, 7-day horizon chart, severity/component summary, priority robots |
| `/#robots` | Robot-level Head 1-4 outputs with date/reference controls |
| `/#history` | Historical fault frequency and log browser |
| `/#model` | Fresh GitHub checkout status, parsed metrics, feature contract, runtime notes |

## Run Locally

```bash
pip install -r requirements.txt
python3 app.py
```

Open <http://127.0.0.1:8000>.

First launch downloads the dataset snapshot. Every process start also performs a
fresh temporary clone of the configured model-training repo.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `HF_DATASET_REPO` | `Lightcap/pudu-robot-operation-logs-bau-capstone-2026` | Hugging Face dataset repository |
| `HF_DATASET_REVISION` | `main` | Dataset revision/commit/branch |
| `DASHBOARD_DUCKDB_PATH` | `:memory:` | Optional DuckDB database path |
| `PUDU_MODEL_REPO_URL` | `https://github.com/DrGb24/pudu_bot_model_training.git` | Runtime model-training repository |
| `PUDU_MODEL_REPO_REF` | `main` | Branch/tag to clone each run |

## API

| Endpoint | Purpose |
|---|---|
| `/api/health` | Dataset and model-runtime status |
| `/api/model-runtime` | Fresh clone status, commit, parsed contract, artifact availability |
| `/api/model-info` | Runtime contract plus registered dataset training metadata |
| `/api/model-heads/summary` | Four-head fleet summary for the selected reference/horizon |
| `/api/model-heads/robots` | Paginated robot-level Head 1-4 outputs |
| `/api/model-heads/timeline` | Future-window fault counts for the selected reference |
| `/api/fault-history/frequency` | Monthly stacked fault counts by model-repo component labels |
| `/api/fault-history/list` | Paginated, filterable historical fault log |

Legacy endpoints are still present for compatibility, but the UI uses the
`/api/model-heads/*` surface.

## File Layout

```
dashboard/
├── app.py                  # FastAPI routes + HTML/CSS/JS shell
├── pudu_model_runtime.py   # Temporary GitHub checkout + model contract parser
├── requirements.txt
├── run.bat
└── README.md
```

## Project References

- Dataset: <https://huggingface.co/datasets/Lightcap/pudu-robot-operation-logs-bau-capstone-2026>
- Model training repository: <https://github.com/DrGb24/pudu_bot_model_training/>
