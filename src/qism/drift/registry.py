"""Model and reference versioning registry, checkpoints, rollback, and adaptation event log (V0.8).

Provides:
- ModelVersion: Complete metadata record for active and candidate detector states.
- ModelRegistry: Atomic version management, candidate promotion, rollback, and checkpointing.
- AdaptationEventLog: Immutable operational audit log of all drift transitions, safety evaluations, and adaptations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
from typing import Any

import numpy as np


class ModelStatus(str, Enum):
    """Lifecycle status of a model version in the registry."""
    ACTIVE = "ACTIVE"
    CANDIDATE = "CANDIDATE"
    SUPERSEDED = "SUPERSEDED"
    REJECTED = "REJECTED"
    ROLLED_BACK = "ROLLED_BACK"


class AdaptationDecision(str, Enum):
    """Decision made by the safe adaptation gate."""
    NO_DRIFT = "NO_DRIFT"
    DRIFT_DETECTED = "DRIFT_DETECTED"
    DRIFT_PERSISTING = "DRIFT_PERSISTING"
    ADAPTATION_INITIATED = "ADAPTATION_INITIATED"
    SHADOW_VALIDATING = "SHADOW_VALIDATING"
    PROMOTED = "PROMOTED"
    REJECTED_ANOMALY = "REJECTED_ANOMALY"
    REJECTED_COOLDOWN = "REJECTED_COOLDOWN"
    REJECTED_VALIDATION_FAILED = "REJECTED_VALIDATION_FAILED"
    ROLLBACK_EXECUTED = "ROLLBACK_EXECUTED"


@dataclass
class ModelVersionRecord:
    """Record describing a registered model version."""
    version_id: str
    parent_version_id: str | None
    created_at_step: int
    created_at_time_s: float | None
    model_type: str
    threshold: float
    reference_version_id: str
    status: ModelStatus
    trigger: str
    validation_metrics: dict[str, float] = field(default_factory=dict)
    hyperparameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class AdaptationLogEntry:
    """Immutable log entry for a single adaptation evaluation or transition."""
    step_index: int
    timestamp_s: float | None
    active_model_version: str
    reference_version: str
    drift_score: float
    drift_threshold: float
    is_drift: bool
    anomaly_score: float
    anomaly_threshold: float
    operational_alarm: bool
    adaptation_state: str
    decision: AdaptationDecision
    candidate_version: str | None = None
    reason: str = ""


class ModelRegistry:
    """Manages detector versioning, promotion, rollback, and operational event logging."""

    def __init__(self, output_dir: str | Path | None = None):
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self._versions: dict[str, ModelVersionRecord] = {}
        self._model_instances: dict[str, Any] = {}
        self.active_version_id: str = "model_v001"
        self.candidate_version_id: str | None = None
        self.event_log: list[AdaptationLogEntry] = []
        self._version_counter: int = 0

    def register_initial_model(
        self,
        model: Any,
        threshold: float,
        reference_version_id: str = "ref_v001",
        step_index: int = 0,
        timestamp_s: float | None = 0.0,
        version_id: str = "model_v001",
    ) -> ModelVersionRecord:
        """Register the baseline normal model version."""
        self._version_counter = 1
        record = ModelVersionRecord(
            version_id=version_id,
            parent_version_id=None,
            created_at_step=step_index,
            created_at_time_s=timestamp_s,
            model_type=model.__class__.__name__,
            threshold=float(threshold),
            reference_version_id=reference_version_id,
            status=ModelStatus.ACTIVE,
            trigger="INITIAL_BASELINE",
            hyperparameters=getattr(model, "hyperparameters", {}),
        )
        self._versions[version_id] = record
        self._model_instances[version_id] = model
        self.active_version_id = version_id
        return record

    def register_candidate(
        self,
        candidate_model: Any,
        candidate_threshold: float,
        reference_version_id: str,
        step_index: int,
        timestamp_s: float | None,
        trigger: str = "VALIDATED_DRIFT",
    ) -> ModelVersionRecord:
        """Register a new candidate model version for shadow validation."""
        self._version_counter += 1
        cand_id = f"model_v{self._version_counter:03d}"
        record = ModelVersionRecord(
            version_id=cand_id,
            parent_version_id=self.active_version_id,
            created_at_step=step_index,
            created_at_time_s=timestamp_s,
            model_type=candidate_model.__class__.__name__,
            threshold=float(candidate_threshold),
            reference_version_id=reference_version_id,
            status=ModelStatus.CANDIDATE,
            trigger=trigger,
            hyperparameters=getattr(candidate_model, "hyperparameters", {}),
        )
        self._versions[cand_id] = record
        self._model_instances[cand_id] = candidate_model
        self.candidate_version_id = cand_id
        return record

    def promote_candidate(
        self,
        validation_metrics: dict[str, float] | None = None,
    ) -> ModelVersionRecord:
        """Promote candidate model to active and supersede previous active version."""
        if self.candidate_version_id is None or self.candidate_version_id not in self._versions:
            raise RuntimeError("No candidate model available for promotion.")

        cand_id = self.candidate_version_id
        old_active_id = self.active_version_id

        # Supersede old active
        if old_active_id in self._versions:
            self._versions[old_active_id].status = ModelStatus.SUPERSEDED

        # Promote candidate
        cand_record = self._versions[cand_id]
        cand_record.status = ModelStatus.ACTIVE
        if validation_metrics:
            cand_record.validation_metrics.update(validation_metrics)

        self.active_version_id = cand_id
        self.candidate_version_id = None
        return cand_record

    def reject_candidate(self, reason: str = "Validation failed") -> ModelVersionRecord:
        """Reject candidate model and discard from candidate evaluation."""
        if self.candidate_version_id is None or self.candidate_version_id not in self._versions:
            raise RuntimeError("No candidate model to reject.")

        cand_id = self.candidate_version_id
        record = self._versions[cand_id]
        record.status = ModelStatus.REJECTED
        record.trigger = f"REJECTED: {reason}"
        self.candidate_version_id = None
        return record

    def rollback(self, target_version_id: str | None = None) -> ModelVersionRecord:
        """Rollback active model to parent or specified previous stable version."""
        current_record = self._versions[self.active_version_id]
        target_id = target_version_id or current_record.parent_version_id

        if not target_id or target_id not in self._versions:
            # Cannot rollback beyond root
            return current_record

        # Mark current active as rolled back
        current_record.status = ModelStatus.ROLLED_BACK

        # Restore target
        target_record = self._versions[target_id]
        target_record.status = ModelStatus.ACTIVE

        self.active_version_id = target_id
        self.candidate_version_id = None
        return target_record

    def get_active_model(self) -> Any:
        """Retrieve active model instance."""
        return self._model_instances.get(self.active_version_id)

    def get_candidate_model(self) -> Any | None:
        """Retrieve candidate model instance if in shadow validation."""
        if self.candidate_version_id:
            return self._model_instances.get(self.candidate_version_id)
        return None

    def log_event(self, entry: AdaptationLogEntry) -> None:
        """Append an entry to the immutable adaptation event log."""
        self.event_log.append(entry)

    def save_checkpoint(self, path: str | Path) -> None:
        """Serialize complete registry state, versions, and log to disk."""
        chk_path = Path(path)
        chk_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "active_version_id": self.active_version_id,
            "candidate_version_id": self.candidate_version_id,
            "version_counter": self._version_counter,
            "versions": {k: asdict(v) for k, v in self._versions.items()},
            "event_log": [
                {
                    **asdict(e),
                    "decision": e.decision.value,
                }
                for e in self.event_log
            ],
        }
        with open(chk_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_checkpoint(cls, path: str | Path, model_factory: Any | None = None) -> ModelRegistry:
        """Restore registry from disk checkpoint."""
        chk_path = Path(path)
        with open(chk_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        registry = cls(output_dir=chk_path.parent)
        registry.active_version_id = data["active_version_id"]
        registry.candidate_version_id = data["candidate_version_id"]
        registry._version_counter = data["version_counter"]

        for k, v in data["versions"].items():
            record = ModelVersionRecord(
                version_id=v["version_id"],
                parent_version_id=v["parent_version_id"],
                created_at_step=v["created_at_step"],
                created_at_time_s=v["created_at_time_s"],
                model_type=v["model_type"],
                threshold=v["threshold"],
                reference_version_id=v["reference_version_id"],
                status=ModelStatus(v["status"]),
                trigger=v["trigger"],
                validation_metrics=v.get("validation_metrics", {}),
                hyperparameters=v.get("hyperparameters", {}),
            )
            registry._versions[k] = record

        for item in data.get("event_log", []):
            entry = AdaptationLogEntry(
                step_index=item["step_index"],
                timestamp_s=item["timestamp_s"],
                active_model_version=item["active_model_version"],
                reference_version=item["reference_version"],
                drift_score=item["drift_score"],
                drift_threshold=item["drift_threshold"],
                is_drift=item["is_drift"],
                anomaly_score=item["anomaly_score"],
                anomaly_threshold=item["anomaly_threshold"],
                operational_alarm=item["operational_alarm"],
                adaptation_state=item["adaptation_state"],
                decision=AdaptationDecision(item["decision"]),
                candidate_version=item.get("candidate_version"),
                reason=item.get("reason", ""),
            )
            registry.event_log.append(entry)

        return registry
