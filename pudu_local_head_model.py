"""Local model artifacts for the dashboard prediction heads.

The preferred runtime path is the external DrGb24 LSTM V2 inference engine.
That public repository intentionally excludes trained weights, so this module
provides a transparent artifact-backed fallback: train compact head models from
the same published Hugging Face training split and the same source contract.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("dashboard.local_head_model")

DEFAULT_DATASET_REPO = "Lightcap/pudu-robot-operation-logs-bau-capstone-2026"
DEFAULT_DATASET_REVISION = "main"
TRAINING_PARQUET = "data/public_robot_logs_error_training.parquet"
ARTIFACT_VERSION = 1
DEFAULT_HORIZON_HOURS = (7 * 24, 30 * 24, 90 * 24)


@dataclass(frozen=True)
class LocalModelContract:
    feature_columns: list[str]
    failure_levels: list[str]
    severity_map: dict[str, int]
    severity_labels: dict[int, str]
    severity_labels_tr: dict[int, str]
    error_category_map: dict[str, int]
    product_map: dict[str, int]
    unknown_product_code_type: int = 5

    def signature_payload(
        self,
        dataset_repo: str,
        dataset_revision: str,
        horizons: tuple[int, ...],
    ) -> dict[str, Any]:
        return {
            "artifact_version": ARTIFACT_VERSION,
            "dataset_repo": dataset_repo,
            "dataset_revision": dataset_revision,
            "training_parquet": TRAINING_PARQUET,
            "feature_columns": self.feature_columns,
            "failure_levels": sorted(self.failure_levels),
            "severity_map": self.severity_map,
            "severity_labels": _string_keyed(self.severity_labels),
            "severity_labels_tr": _string_keyed(self.severity_labels_tr),
            "error_category_map": _string_keyed(self.error_category_map),
            "product_map": self.product_map,
            "unknown_product_code_type": self.unknown_product_code_type,
            "horizon_hours": list(horizons),
        }


@dataclass
class LocalHeadModelEngine:
    artifact_path: Path
    metadata: dict[str, Any]
    failure_model: Any
    severity_model: Any
    future_models: dict[int, Any] = field(default_factory=dict)
    eta_models: dict[int, Any] = field(default_factory=dict)
    contract: LocalModelContract | None = None

    @classmethod
    def prepare(
        cls,
        artifact_dir: Path,
        contract: LocalModelContract,
        dataset_repo: str | None = None,
        dataset_revision: str | None = None,
        horizons: tuple[int, ...] = DEFAULT_HORIZON_HOURS,
    ) -> "LocalHeadModelEngine":
        dataset_repo = dataset_repo or os.getenv("PUDU_MODEL_DATASET_REPO", DEFAULT_DATASET_REPO)
        dataset_revision = dataset_revision or os.getenv("PUDU_MODEL_DATASET_REVISION", DEFAULT_DATASET_REVISION)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        model_path = artifact_dir / "local_head_models.joblib"
        meta_path = artifact_dir / "local_head_models.json"
        expected = contract.signature_payload(dataset_repo, dataset_revision, tuple(sorted(set(horizons))))

        if model_path.exists() and meta_path.exists():
            current = json.loads(meta_path.read_text(encoding="utf-8"))
            if current.get("signature") == expected:
                return cls._load(model_path, current, contract)

        if os.getenv("PUDU_LOCAL_MODEL_AUTOTRAIN", "1").lower() in {"0", "false", "no"}:
            raise RuntimeError("local model artifacts are absent and PUDU_LOCAL_MODEL_AUTOTRAIN is disabled")

        return cls._train_and_save(model_path, meta_path, contract, dataset_repo, dataset_revision, tuple(sorted(set(horizons))), expected)

    @classmethod
    def _load(cls, model_path: Path, metadata: dict[str, Any], contract: LocalModelContract) -> "LocalHeadModelEngine":
        import joblib

        payload = joblib.load(model_path)
        return cls(
            artifact_path=model_path,
            metadata=metadata,
            failure_model=payload["failure_model"],
            severity_model=payload["severity_model"],
            future_models={int(k): v for k, v in payload["future_models"].items()},
            eta_models={int(k): v for k, v in payload["eta_models"].items()},
            contract=contract,
        )

    @classmethod
    def _train_and_save(
        cls,
        model_path: Path,
        meta_path: Path,
        contract: LocalModelContract,
        dataset_repo: str,
        dataset_revision: str,
        horizons: tuple[int, ...],
        signature: dict[str, Any],
    ) -> "LocalHeadModelEngine":
        import joblib
        from huggingface_hub import hf_hub_download
        from sklearn.dummy import DummyClassifier
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

        parquet_path = hf_hub_download(
            repo_id=dataset_repo,
            repo_type="dataset",
            filename=TRAINING_PARQUET,
            revision=dataset_revision,
        )
        df = _read_training_frame(parquet_path)
        engineered = _engineer_features(df, contract)
        targets = _with_future_targets(engineered, contract, horizons)

        x = targets[contract.feature_columns].to_numpy(dtype="float32")
        y_failure = targets["is_failure_now"].to_numpy(dtype="int32")
        y_severity = targets["severity_class"].to_numpy(dtype="int32")

        failure_model = _fit_classifier(HistGradientBoostingClassifier, DummyClassifier, x, y_failure)
        severity_model = _fit_classifier(HistGradientBoostingClassifier, DummyClassifier, x, y_severity)

        future_models: dict[int, Any] = {}
        eta_models: dict[int, Any] = {}
        for horizon in horizons:
            future_y = targets[f"future_failure_{horizon}h"].to_numpy(dtype="int32")
            eta_y = targets[f"hours_to_failure_norm_{horizon}h"].to_numpy(dtype="float32")
            future_models[horizon] = _fit_classifier(HistGradientBoostingClassifier, DummyClassifier, x, future_y)
            eta_models[horizon] = HistGradientBoostingRegressor(max_iter=_max_iter(), random_state=42).fit(x, eta_y)

        metadata = {
            "signature": signature,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "artifact_version": ARTIFACT_VERSION,
            "engine_kind": "local_head_model",
            "dataset_rows": int(len(targets)),
            "horizon_hours": list(horizons),
            "model_family": "sklearn_hist_gradient_boosting",
            "source_contract": "DrGb24/pudu_bot_model_training LSTM V2 feature and target contract",
        }
        payload = {
            "failure_model": failure_model,
            "severity_model": severity_model,
            "future_models": future_models,
            "eta_models": eta_models,
        }
        joblib.dump(payload, model_path)
        meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        log.info("Local head model artifacts trained at %s", model_path)
        return cls._load(model_path, metadata, contract)

    @property
    def supported_horizon_hours(self) -> list[int]:
        return sorted(self.future_models.keys())

    def predict_for_robot(
        self,
        robot_id: str,
        rows: list[dict[str, Any]],
        reference: datetime,
        horizon_hours: int | None = None,
    ) -> dict[str, Any] | None:
        if not rows or self.contract is None:
            return None
        import pandas as pd

        df = pd.DataFrame(rows)
        if df.empty:
            return None
        df["task_time"] = pd.to_datetime(df["task_time"])
        cutoff = pd.Timestamp(reference)
        df = df[df["task_time"] <= cutoff]
        if df.empty:
            return None
        engineered = _engineer_features(df, self.contract).sort_values("task_time")
        x_last = engineered[self.contract.feature_columns].tail(1).to_numpy(dtype="float32")

        failure_prob = _positive_class_probability(self.failure_model, x_last)
        severity_score = int(self.severity_model.predict(x_last)[0])
        severity_probs = _class_probability_map(self.severity_model, x_last)
        supported = horizon_hours in self.future_models if horizon_hours is not None else False
        selected_horizon = int(horizon_hours) if supported else None

        future_prob = None
        est_hours = None
        if selected_horizon is not None:
            future_prob = _positive_class_probability(self.future_models[selected_horizon], x_last)
            eta_norm = float(self.eta_models[selected_horizon].predict(x_last)[0])
            est_hours = round(max(0.0, min(float(selected_horizon), eta_norm * selected_horizon)), 1)

        recent = engineered.tail(10)
        active = recent[recent["error_level"].isin(set(self.contract.failure_levels))]
        active_error_types = [str(v) for v in active["error_type"].dropna().unique().tolist()]
        active_categories = [
            str(self.contract.error_category_map.get(error_type, 0))
            for error_type in active_error_types
        ]
        return {
            "robot_id": robot_id,
            "reference_date": str(cutoff.date()),
            "is_failure_now": failure_prob >= 0.5,
            "failure_prob_now": round(failure_prob, 4),
            "active_error_types": active_error_types,
            "active_error_categories": active_categories,
            "error_details": [],
            "severity_now": self.contract.severity_labels.get(severity_score, "Unknown"),
            "severity_now_tr": self.contract.severity_labels_tr.get(severity_score, self.contract.severity_labels.get(severity_score, "Unknown")),
            "severity_score": severity_score,
            "severity_probs": {
                self.contract.severity_labels.get(idx, str(idx)): round(prob, 4)
                for idx, prob in severity_probs.items()
            },
            "future_horizon_supported": selected_horizon is not None,
            "future_window_hours": selected_horizon,
            "monthly_repair_prob": future_prob,
            "next_7d_fail_prob": future_prob,
            "future_failure_prob": future_prob,
            "est_hours_to_failure": est_hours,
            "est_days_to_failure": None if est_hours is None else round(est_hours / 24, 2),
        }


def default_artifact_dir() -> Path:
    base = os.getenv("PUDU_MODEL_ARTIFACT_DIR")
    if base:
        return Path(base)
    cache_root = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_root / "pudu-dashboard" / "model_artifacts" / "local_heads"


def _read_training_frame(parquet_path: str):
    import duckdb

    safe_path = str(parquet_path).replace("'", "''")
    return duckdb.connect(":memory:").execute(
        f"""
        SELECT robot_id, product_code, soft_version, error_type, error_level,
               hourly_ratio, hourly_error_count, task_time, task_hour
        FROM read_parquet('{safe_path}')
        WHERE robot_id IS NOT NULL
          AND task_time IS NOT NULL
        ORDER BY robot_id, task_time
        """
    ).fetchdf()


def _string_keyed(value: dict[Any, Any]) -> dict[str, Any]:
    return {str(k): v for k, v in value.items()}


def _engineer_features(df, contract: LocalModelContract):
    import numpy as np
    import pandas as pd

    out = df.copy()
    out["task_time"] = pd.to_datetime(out["task_time"])
    out["task_hour"] = pd.to_datetime(out.get("task_hour", out["task_time"]))
    out["error_count"] = out.get("hourly_error_count", 0)
    out["task_hour_num"] = out["task_hour"].dt.hour
    out["day_of_month"] = out["task_time"].dt.day
    out["day_of_week"] = out["task_time"].dt.dayofweek
    out["robot_id_length"] = out["robot_id"].astype(str).str.len()
    soft_version = out["soft_version"] if "soft_version" in out.columns else pd.Series([""] * len(out), index=out.index)
    product_code = out["product_code"] if "product_code" in out.columns else pd.Series([""] * len(out), index=out.index)
    error_type = out["error_type"] if "error_type" in out.columns else pd.Series([""] * len(out), index=out.index)
    out["software_version_length"] = soft_version.astype(str).str.len()
    out["product_code_type"] = (
        product_code.map(contract.product_map)
        .fillna(contract.unknown_product_code_type)
        .astype(int)
    )
    out["hourly_error_rate"] = out.get("hourly_ratio", 0)
    out["error_category"] = error_type.map(contract.error_category_map).fillna(0).astype(int)
    for col in contract.feature_columns:
        if col not in out.columns:
            out[col] = 0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).replace([np.inf, -np.inf], 0)
    return out


def _with_future_targets(df, contract: LocalModelContract, horizons: tuple[int, ...]):
    import numpy as np
    import pandas as pd

    parts = []
    failure_levels = set(contract.failure_levels)
    for _robot_id, rdf in df.groupby("robot_id", sort=False):
        rdf = rdf.sort_values("task_time").copy().reset_index(drop=True)
        rdf["is_failure_now"] = rdf["error_level"].isin(failure_levels).astype(int)
        rdf["severity_class"] = rdf["error_level"].map(contract.severity_map).fillna(0).astype(int)

        times = pd.to_datetime(rdf["task_time"]).to_numpy(dtype="datetime64[ns]").astype("int64")
        failure_times = times[rdf["is_failure_now"].to_numpy(dtype=bool)]
        if len(failure_times) == 0:
            for horizon in horizons:
                rdf[f"future_failure_{horizon}h"] = 0
                rdf[f"hours_to_failure_norm_{horizon}h"] = 1.0
            parts.append(rdf)
            continue

        next_idx = np.searchsorted(failure_times, times, side="right")
        has_next = next_idx < len(failure_times)
        delta_hours = np.full(len(rdf), np.inf, dtype="float64")
        delta_hours[has_next] = (failure_times[next_idx[has_next]] - times[has_next]) / 1e9 / 3600
        for horizon in horizons:
            future = delta_hours <= horizon
            rdf[f"future_failure_{horizon}h"] = future.astype(int)
            hours = np.where(future, delta_hours, float(horizon))
            rdf[f"hours_to_failure_norm_{horizon}h"] = np.clip(hours / float(horizon), 0.0, 1.0)
        parts.append(rdf)
    return pd.concat(parts, ignore_index=True)


def _fit_classifier(classifier_cls, dummy_cls, x, y):
    import numpy as np

    if len(np.unique(y)) < 2:
        return dummy_cls(strategy="most_frequent").fit(x, y)
    kwargs = {"max_iter": _max_iter(), "random_state": 42}
    try:
        return classifier_cls(**kwargs, class_weight="balanced").fit(x, y)
    except TypeError:
        return classifier_cls(**kwargs).fit(x, y)


def _positive_class_probability(model, x) -> float:
    if not hasattr(model, "predict_proba"):
        return float(model.predict(x)[0])
    proba = model.predict_proba(x)[0]
    classes = list(getattr(model, "classes_", []))
    if 1 in classes:
        return float(proba[classes.index(1)])
    return 0.0


def _class_probability_map(model, x) -> dict[int, float]:
    if not hasattr(model, "predict_proba"):
        pred = int(model.predict(x)[0])
        return {pred: 1.0}
    proba = model.predict_proba(x)[0]
    classes = [int(c) for c in getattr(model, "classes_", [])]
    return {cls: float(proba[i]) for i, cls in enumerate(classes)}


def _max_iter() -> int:
    try:
        return max(20, int(os.getenv("PUDU_LOCAL_MODEL_MAX_ITER", "80")))
    except ValueError:
        return 80
