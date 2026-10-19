"""Configuration settings for the QISM Live Demonstration Backend."""

from pathlib import Path
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATASET_ROOT = REPO_ROOT / "data" / "qism_research_dataset"
RESULTS_ROOT = REPO_ROOT / "results" / "qism_model_benchmark"
CHECKPOINTS_DIR = RESULTS_ROOT / "checkpoints"
TABLES_DIR = RESULTS_ROOT / "tables"
FIGURES_DIR = RESULTS_ROOT / "figures"
PREPROCESSORS_DIR = RESULTS_ROOT / "preprocessors"
DB_PATH = REPO_ROOT / "qism-live-demo" / "qism_demo.db"

class Settings(BaseModel):
    app_name: str = "QISM Live Demonstration Console"
    version: str = "1.0.0"
    host: str = "127.0.0.1"
    port: int = 8000
    default_model_id: str = "PITQSVDD_6Q_Angle"
    default_qubits: int = 6
    window_update_interval_s: float = 0.5  # 2 Hz telemetry stream for responsive UI
    consecutive_anomalies_for_alert: int = 1
    alert_cooldown_windows: int = 3

settings = Settings()
