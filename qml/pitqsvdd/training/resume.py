"""Resume resolution logic and execution guardrails for PI-TQSVDD."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from qml.pitqsvdd.training.checkpointing import CheckpointManager


def resolve_resume_target(
    checkpoint_mgr: CheckpointManager,
    resume_mode: Literal["auto", "none", "explicit"] = "auto",
    resume_path: str | Path | None = None,
    overwrite: bool = False,
) -> Path | None:
    """Resolve whether to resume, start fresh, or fail on overwrite conflict.

    Rules:
    - mode="auto": If checkpoint exists, return it. If not, start fresh.
    - mode="none": Always start fresh. If checkpoint exists and overwrite=False, raise FileExistsError.
    - mode="explicit": Load explicitly specified checkpoint path.
    """
    if resume_mode == "explicit":
        if resume_path is None:
            raise ValueError("resume_mode='explicit' requested but no resume_path was provided.")
        p = Path(resume_path)
        if not p.exists():
            raise FileNotFoundError(f"Explicit resume checkpoint does not exist: {p}")
        return p

    elif resume_mode == "auto":
        if checkpoint_mgr.has_checkpoint() and not overwrite:
            return checkpoint_mgr.latest_path
        return None

    elif resume_mode == "none":
        if checkpoint_mgr.has_checkpoint() and not overwrite:
            raise FileExistsError(
                f"Existing run found at {checkpoint_mgr.run_dir}. "
                f"Specify --overwrite to start fresh or use --resume auto."
            )
        return None

    else:
        raise ValueError(f"Unknown resume_mode: '{resume_mode}'")
