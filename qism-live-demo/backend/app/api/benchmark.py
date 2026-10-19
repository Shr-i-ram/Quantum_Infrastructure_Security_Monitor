"""Benchmark leaderboard and research figures API router."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import pandas as pd

from ..config import FIGURES_DIR, TABLES_DIR

router = APIRouter(prefix="/benchmark", tags=["Benchmark"])


@router.get("/leaderboard")
def get_leaderboard() -> dict[str, Any]:
    """Retrieve authoritative research benchmark metrics across all evaluated models."""
    csv_path = TABLES_DIR / "aggregate_metrics.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="aggregate_metrics.csv not found")

    df = pd.read_csv(csv_path)
    # Replace NaNs with None for clean JSON serialization
    records = df.where(pd.notnull(df), None).to_dict(orient="records")

    return {
        "leaderboard": records,
        "total_models": len(records),
        "source": "QISM Authoritative Multi-Model Benchmark (Phase XI)",
        "flagship_model": "PITQSVDD_6Q_Angle",
    }


@router.get("/per-attack")
def get_per_attack_metrics() -> dict[str, Any]:
    """Retrieve breakdown of detector performance per individual attack type."""
    csv_path = TABLES_DIR / "per_attack_metrics.csv"
    if not csv_path.exists():
        return {"metrics": [], "count": 0}

    df = pd.read_csv(csv_path)
    records = df.where(pd.notnull(df), None).to_dict(orient="records")
    return {
        "metrics": records,
        "count": len(records),
    }


@router.get("/figures")
def list_figures() -> dict[str, Any]:
    """Retrieve list of publication-grade figures with captions and relative paths."""
    figures = [
        {
            "id": "01_overall_auroc_comparison",
            "title": "Figure 1: Overall Test AUROC Across All Evaluated Models",
            "category": "Leaderboard Comparison",
            "caption": "Comparison of test AUROC across all 24 models including classical ML, quantum models, and PI-TQSVDD configurations.",
            "file": "comparisons/01_overall_auroc_comparison.png",
        },
        {
            "id": "02_recall_at_1pct_fpr_comparison",
            "title": "Figure 2: Event Recall at Calibrated 1% False Positive Rate",
            "category": "Operational Reliability",
            "caption": "Detection sensitivity when models are strictly calibrated to produce ≤1% false alarms under normal link conditions.",
            "file": "comparisons/02_recall_at_1pct_fpr_comparison.png",
        },
        {
            "id": "03_performance_vs_compute_frontier",
            "title": "Figure 3: Pareto Frontier: AUROC vs Inference Latency",
            "category": "Compute Efficiency",
            "caption": "Trade-off between detection accuracy and computational cost, demonstrating quantum advantage in compact state representation.",
            "file": "comparisons/03_performance_vs_compute_frontier.png",
        },
        {
            "id": "04_qubit_count_sweep",
            "title": "Figure 4: PI-TQSVDD Qubit Count Sweep (2Q, 4Q, 6Q, 8Q)",
            "category": "Ablation Study",
            "caption": "Ablation over quantum register size demonstrating peak performance at 4-6 qubits before barren plateau attenuation at 8 qubits.",
            "file": "pitqsvdd/04_qubit_count_sweep.png",
        },
        {
            "id": "05_encoding_comparison",
            "title": "Figure 5: Quantum State Encoding Scheme Comparison",
            "category": "Ablation Study",
            "caption": "Evaluation of Angle, Amplitude, and Data-Reuploading encodings on quantum Hilbert space separation.",
            "file": "pitqsvdd/05_encoding_comparison.png",
        },
        {
            "id": "06_temporal_ablation_comparison",
            "title": "Figure 6: Temporal Loss Regularization Ablation (Static vs Temporal)",
            "category": "Ablation Study",
            "caption": "Impact of temporal physics-informed regularizer on anomaly detection delay and continuous attack tracking.",
            "file": "pitqsvdd/06_temporal_ablation_comparison.png",
        },
        {
            "id": "07_generalization_robustness",
            "title": "Figure 7: Out-of-Distribution Generalization Robustness",
            "category": "Generalization & Stress",
            "caption": "Robustness of trained anomaly detectors under unobserved thermal drift, fiber aging, and novel attack onset variations.",
            "file": "generalization/07_generalization_robustness.png",
        },
    ]
    return {
        "figures": figures,
        "total": len(figures),
    }


@router.get("/figures/{category}/{filename}")
def get_figure_image(category: str, filename: str):
    """Serve a high-resolution publication figure image."""
    img_path = (FIGURES_DIR / category / filename).resolve()
    if not img_path.exists() or not str(img_path).startswith(str(FIGURES_DIR.resolve())):
        raise HTTPException(status_code=404, detail="Figure file not found.")

    return FileResponse(str(img_path), media_type="image/png")
