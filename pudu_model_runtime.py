"""Runtime bridge for the external PUDU LSTM training repository.

The dashboard intentionally does not vendor model-training logic.  Every app
process clones the configured repository into a temporary directory, reads the
model contract from that checkout, and exposes enough metadata for the product
layer to stay transparent about what is available.
"""
from __future__ import annotations

import ast
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("dashboard.model_runtime")

DEFAULT_MODEL_REPO_URL = "https://github.com/DrGb24/pudu_bot_model_training.git"
DEFAULT_MODEL_REPO_REF = "main"


@dataclass(frozen=True)
class ModelHeadMetric:
    id: str
    name: str
    metric: str
    result: str
    values: dict[str, float] = field(default_factory=dict)
    unit: str | None = None


@dataclass(frozen=True)
class RuntimeSnapshot:
    status: str
    repo_url: str
    repo_ref: str
    repo_path: str | None = None
    git_commit: str | None = None
    checked_out_at: str | None = None
    error: str | None = None
    weights_available: bool = False
    engine_available: bool = False
    engine_error: str | None = None
    future_window_hours: int | None = None
    feature_columns: list[str] = field(default_factory=list)
    failure_levels: list[str] = field(default_factory=list)
    severity_map: dict[str, int] = field(default_factory=dict)
    severity_labels: dict[int, str] = field(default_factory=dict)
    severity_labels_tr: dict[int, str] = field(default_factory=dict)
    error_category_labels: dict[int, str] = field(default_factory=dict)
    metrics: list[ModelHeadMetric] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        data = {
            "status": self.status,
            "repo_url": self.repo_url,
            "repo_ref": self.repo_ref,
            "repo_path": self.repo_path,
            "git_commit": self.git_commit,
            "checked_out_at": self.checked_out_at,
            "error": self.error,
            "weights_available": self.weights_available,
            "engine_available": self.engine_available,
            "engine_error": self.engine_error,
            "future_window_hours": self.future_window_hours,
            "feature_columns": self.feature_columns,
            "failure_levels": self.failure_levels,
            "severity_map": self.severity_map,
            "severity_labels": self.severity_labels,
            "severity_labels_tr": self.severity_labels_tr,
            "error_category_labels": self.error_category_labels,
            "metrics": [m.__dict__ for m in self.metrics],
        }
        return data


class PuduModelRuntime:
    """Fresh temporary checkout plus parsed model contract."""

    def __init__(self, repo_url: str | None = None, repo_ref: str | None = None):
        self.repo_url = repo_url or os.getenv("PUDU_MODEL_REPO_URL", DEFAULT_MODEL_REPO_URL)
        self.repo_ref = repo_ref or os.getenv("PUDU_MODEL_REPO_REF", DEFAULT_MODEL_REPO_REF)
        self._lock = threading.RLock()
        self._tempdir: tempfile.TemporaryDirectory[str] | None = None
        self._snapshot: RuntimeSnapshot | None = None
        self._error_category_map: dict[str, int] = {}
        self._engine: Any | None = None

    def ensure_loaded(self) -> RuntimeSnapshot:
        with self._lock:
            if self._snapshot is not None:
                return self._snapshot
            self._snapshot = self._load()
            return self._snapshot

    def snapshot(self) -> RuntimeSnapshot:
        return self.ensure_loaded()

    def category_for_error_type(self, error_type: str | None) -> str:
        snapshot = self.ensure_loaded()
        if not error_type:
            return "Bilinmiyor"
        category_id = self._error_category_map.get(str(error_type))
        if category_id is None:
            return "Bilinmiyor"
        return snapshot.error_category_labels.get(category_id, "Bilinmiyor")

    def severity_score(self, error_level: str | None) -> int | None:
        if not error_level:
            return None
        snapshot = self.ensure_loaded()
        return snapshot.severity_map.get(str(error_level))

    def is_failure_level(self, error_level: str | None) -> bool:
        if not error_level:
            return False
        snapshot = self.ensure_loaded()
        return str(error_level) in set(snapshot.failure_levels)

    def metric_for_head(self, head_id: str) -> ModelHeadMetric | None:
        for metric in self.ensure_loaded().metrics:
            if metric.id == head_id:
                return metric
        return None

    def predict_for_robot(
        self,
        robot_id: str,
        rows: list[dict[str, Any]],
        reference: datetime,
    ) -> dict[str, Any] | None:
        """Run the external inference engine when trained artifacts are available."""
        self.ensure_loaded()
        if self._engine is None or not rows:
            return None
        try:
            import pandas as pd

            df = pd.DataFrame(rows)
            return self._engine.predict_for_robot(
                robot_id=robot_id,
                robot_df=df,
                reference_date=pd.Timestamp(reference.date()),
            )
        except Exception as exc:
            log.warning("Model inference failed for robot %s: %s", robot_id, exc)
            return None

    def _load(self) -> RuntimeSnapshot:
        checked_out_at = datetime.now(timezone.utc).isoformat()
        if shutil.which("git") is None:
            return RuntimeSnapshot(
                status="unavailable",
                repo_url=self.repo_url,
                repo_ref=self.repo_ref,
                checked_out_at=checked_out_at,
                error="git executable is not available",
            )

        try:
            self._tempdir = tempfile.TemporaryDirectory(prefix="pudu_model_repo_")
            repo_path = Path(self._tempdir.name) / "repo"
            self._clone_repo(repo_path)
            commit = self._git_commit(repo_path)

            inference_literals = _read_assignments(
                repo_path / "lstm_inference_v2.py",
                {
                    "FEATURE_COLUMNS",
                    "FAILURE_LEVELS",
                    "SEVERITY_MAP",
                    "ERROR_CATEGORY_MAP",
                    "ERROR_CATEGORY_LABELS",
                },
            )
            model_literals = _read_assignments(
                repo_path / "src" / "lstm_models_v2.py",
                {"SEVERITY_LABELS", "SEVERITY_LABELS_TR", "FUTURE_WINDOW_HOURS"},
            )
            config_literals = _read_assignments(repo_path / "src" / "config.py", {"LSTM_V2_CONFIG"})

            self._error_category_map = dict(inference_literals.get("ERROR_CATEGORY_MAP") or {})
            future_window = _future_window(config_literals, model_literals)
            metrics = _parse_readme_metrics(repo_path / "README.md")
            weights_available = _has_v2_weights(repo_path)
            engine_available, engine_error = self._try_prepare_engine(repo_path, weights_available)

            return RuntimeSnapshot(
                status="ready",
                repo_url=self.repo_url,
                repo_ref=self.repo_ref,
                repo_path=str(repo_path),
                git_commit=commit,
                checked_out_at=checked_out_at,
                weights_available=weights_available,
                engine_available=engine_available,
                engine_error=engine_error,
                future_window_hours=future_window,
                feature_columns=list(inference_literals.get("FEATURE_COLUMNS") or []),
                failure_levels=sorted(inference_literals.get("FAILURE_LEVELS") or []),
                severity_map=dict(inference_literals.get("SEVERITY_MAP") or {}),
                severity_labels=_int_keyed(model_literals.get("SEVERITY_LABELS")),
                severity_labels_tr=_int_keyed(model_literals.get("SEVERITY_LABELS_TR")),
                error_category_labels=_int_keyed(inference_literals.get("ERROR_CATEGORY_LABELS")),
                metrics=metrics,
            )
        except Exception as exc:
            log.warning("Model repository could not be prepared: %s", exc)
            return RuntimeSnapshot(
                status="unavailable",
                repo_url=self.repo_url,
                repo_ref=self.repo_ref,
                checked_out_at=checked_out_at,
                error=str(exc),
            )

    def _clone_repo(self, repo_path: Path) -> None:
        cmd = ["git", "clone", "--depth", "1"]
        if self.repo_ref:
            cmd += ["--branch", self.repo_ref]
        cmd += [self.repo_url, str(repo_path)]
        result = subprocess.run(cmd, text=True, capture_output=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout or "git clone failed").strip())

    @staticmethod
    def _git_commit(repo_path: Path) -> str | None:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    def _try_prepare_engine(self, repo_path: Path, weights_available: bool) -> tuple[bool, str | None]:
        if not weights_available:
            return False, "V2 model weights are not present in the cloned repository"
        try:
            import importlib.util
            import sys

            module_path = repo_path / "lstm_inference_v2.py"
            spec = importlib.util.spec_from_file_location("pudu_lstm_inference_v2_runtime", module_path)
            if spec is None or spec.loader is None:
                return False, "Could not load lstm_inference_v2.py"
            module = importlib.util.module_from_spec(spec)
            sys.path.insert(0, str(repo_path))
            try:
                spec.loader.exec_module(module)
                self._engine = module.LSTMInferenceV2(model_dir=repo_path / "models" / "lstm_v2")
            finally:
                try:
                    sys.path.remove(str(repo_path))
                except ValueError:
                    pass
            return True, None
        except Exception as exc:
            self._engine = None
            return False, str(exc)


def _read_assignments(path: Path, names: set[str]) -> dict[str, Any]:
    if not path.exists():
        return {}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in names:
                try:
                    values[target.id] = ast.literal_eval(node.value)
                except (ValueError, SyntaxError):
                    continue
    return values


def _int_keyed(value: Any) -> dict[int, str]:
    if not isinstance(value, dict):
        return {}
    out: dict[int, str] = {}
    for key, item in value.items():
        try:
            out[int(key)] = str(item)
        except (TypeError, ValueError):
            continue
    return out


def _future_window(config_literals: dict[str, Any], model_literals: dict[str, Any]) -> int | None:
    config = config_literals.get("LSTM_V2_CONFIG")
    if isinstance(config, dict) and config.get("future_window") is not None:
        return int(config["future_window"])
    if model_literals.get("FUTURE_WINDOW_HOURS") is not None:
        return int(model_literals["FUTURE_WINDOW_HOURS"])
    return None


def _has_v2_weights(repo_path: Path) -> bool:
    model_dir = repo_path / "models" / "lstm_v2"
    required = [
        model_dir / "lstm_v2_weights.weights.h5",
        model_dir / "lstm_v2_config.json",
        model_dir / "lstm_v2_scaler.pkl",
    ]
    return all(path.exists() for path in required)


def _parse_readme_metrics(path: Path) -> list[ModelHeadMetric]:
    if not path.exists():
        return []
    metrics: list[ModelHeadMetric] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped.startswith("| Head "):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) < 3:
            continue
        match = re.match(r"Head\s+(\d+)\s+[—-]\s*(.+)", cells[0])
        if not match:
            continue
        head_no = match.group(1)
        values, unit = _parse_metric_values(cells[2])
        metrics.append(
            ModelHeadMetric(
                id=f"head_{head_no}",
                name=match.group(2),
                metric=cells[1],
                result=cells[2].replace("✅", "").strip(),
                values=values,
                unit=unit,
            )
        )
    return metrics


def _parse_metric_values(raw: str) -> tuple[dict[str, float], str | None]:
    cleaned = raw.replace("✅", "").strip()
    numbers = [float(n.replace(",", ".")) for n in re.findall(r"\d+(?:[.,]\d+)?", cleaned)]
    unit = "percent" if "%" in cleaned else ("hours" if "saat" in cleaned.lower() else None)
    return {f"value_{i + 1}": value for i, value in enumerate(numbers)}, unit


MODEL_RUNTIME = PuduModelRuntime()
