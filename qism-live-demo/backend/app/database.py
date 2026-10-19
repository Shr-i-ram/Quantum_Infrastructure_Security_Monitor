"""SQLite Database persistence for sessions, incidents, telemetry snapshots, and audit trail."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import time
from typing import Any

from .config import DB_PATH


def get_db_connection() -> sqlite3.Connection:
    """Return an active SQLite connection with row factory enabled."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create database tables if they do not exist."""
    conn = get_db_connection()
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                mode TEXT NOT NULL,  -- 'live_simulation' or 'recorded_replay'
                status TEXT NOT NULL, -- 'running', 'paused', 'stopped', 'completed'
                scenario_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                windows_processed INTEGER DEFAULT 0,
                incidents_detected INTEGER DEFAULT 0
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                incident_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                detection_timestamp REAL NOT NULL,
                simulated_time_ms REAL NOT NULL,
                model_id TEXT NOT NULL,
                anomaly_score REAL NOT NULL,
                threshold REAL NOT NULL,
                severity TEXT NOT NULL, -- 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
                status TEXT NOT NULL,   -- 'NEW', 'ACKNOWLEDGED', 'INVESTIGATING', 'RESOLVED'
                ground_truth_attack_active INTEGER NOT NULL,
                ground_truth_attack_type TEXT,
                detection_latency_windows INTEGER,
                evidence_snapshot TEXT NOT NULL, -- JSON blob of 76 features
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS telemetry_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                window_index INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                qber REAL,
                sifted_rate REAL,
                signal_gain REAL,
                decoy_gain REAL,
                detector_imbalance REAL,
                anomaly_score REAL,
                threshold REAL,
                is_anomaly INTEGER,
                ground_truth_attack_active INTEGER,
                ground_truth_attack_type TEXT,
                features_json TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                session_id TEXT,
                event_type TEXT NOT NULL,
                message TEXT NOT NULL,
                metadata_json TEXT
            )
        """)
    conn.close()


def save_session(session: dict[str, Any]) -> None:
    """Upsert a session record."""
    conn = get_db_connection()
    with conn:
        conn.execute("""
            INSERT INTO sessions (
                session_id, mode, status, scenario_id, model_id, created_at, updated_at, windows_processed, incidents_detected
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                status=excluded.status,
                updated_at=excluded.updated_at,
                windows_processed=excluded.windows_processed,
                incidents_detected=excluded.incidents_detected
        """, (
            session["session_id"],
            session["mode"],
            session["status"],
            session["scenario_id"],
            session["model_id"],
            session["created_at"],
            session.get("updated_at", time.time()),
            session.get("windows_processed", 0),
            session.get("incidents_detected", 0)
        ))
    conn.close()


def save_incident(incident: dict[str, Any]) -> None:
    """Insert or update an incident record."""
    conn = get_db_connection()
    evidence_str = (
        incident["evidence_snapshot"]
        if isinstance(incident["evidence_snapshot"], str)
        else json.dumps(incident["evidence_snapshot"])
    )
    with conn:
        conn.execute("""
            INSERT INTO incidents (
                incident_id, session_id, detection_timestamp, simulated_time_ms, model_id,
                anomaly_score, threshold, severity, status, ground_truth_attack_active,
                ground_truth_attack_type, detection_latency_windows, evidence_snapshot
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(incident_id) DO UPDATE SET
                status=excluded.status
        """, (
            incident["incident_id"],
            incident["session_id"],
            incident["detection_timestamp"],
            incident["simulated_time_ms"],
            incident["model_id"],
            float(incident["anomaly_score"]),
            float(incident["threshold"]),
            incident["severity"],
            incident["status"],
            int(incident.get("ground_truth_attack_active", 0)),
            incident.get("ground_truth_attack_type", "unknown"),
            incident.get("detection_latency_windows", 0),
            evidence_str,
        ))
    conn.close()


def save_telemetry_point(point: dict[str, Any]) -> None:
    """Persist a single telemetry stream point."""
    conn = get_db_connection()
    feat_str = (
        point.get("features_json")
        if isinstance(point.get("features_json"), str)
        else json.dumps(point.get("features", {}))
    )
    with conn:
        conn.execute("""
            INSERT INTO telemetry_history (
                session_id, window_index, timestamp, qber, sifted_rate, signal_gain,
                decoy_gain, detector_imbalance, anomaly_score, threshold, is_anomaly,
                ground_truth_attack_active, ground_truth_attack_type, features_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            point["session_id"],
            point["window_index"],
            point["timestamp"],
            point.get("qber"),
            point.get("sifted_rate"),
            point.get("signal_gain"),
            point.get("decoy_gain"),
            point.get("detector_imbalance"),
            point.get("anomaly_score"),
            point.get("threshold"),
            1 if point.get("is_anomaly") else 0,
            1 if point.get("ground_truth_attack_active") else 0,
            point.get("ground_truth_attack_type", "none"),
            feat_str,
        ))
    conn.close()


def get_incidents(
    session_id: str | None = None,
    severity: str | None = None,
    status: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Retrieve persisted incidents matching optional filters."""
    conn = get_db_connection()
    query = "SELECT * FROM incidents WHERE 1=1"
    params: list[Any] = []
    if session_id:
        query += " AND session_id = ?"
        params.append(session_id)
    if severity:
        query += " AND severity = ?"
        params.append(severity)
    if status:
        query += " AND status = ?"
        params.append(status)
    query += " ORDER BY detection_timestamp DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    conn.close()

    res = []
    for r in rows:
        d = dict(r)
        try:
            d["evidence_snapshot"] = json.loads(d["evidence_snapshot"])
        except Exception:
            pass
        res.append(d)
    return res


def update_incident_status(incident_id: str, new_status: str) -> bool:
    """Update workflow status of an incident."""
    conn = get_db_connection()
    with conn:
        cur = conn.execute(
            "UPDATE incidents SET status = ? WHERE incident_id = ?",
            (new_status, incident_id),
        )
        updated = cur.rowcount > 0
    conn.close()
    return updated


def get_recent_telemetry(session_id: str, limit: int = 100) -> list[dict[str, Any]]:
    """Fetch recent telemetry for a given session."""
    conn = get_db_connection()
    rows = conn.execute("""
        SELECT * FROM telemetry_history
        WHERE session_id = ?
        ORDER BY window_index DESC
        LIMIT ?
    """, (session_id, limit)).fetchall()
    conn.close()
    return [dict(r) for r in reversed(rows)]


def log_audit(event_type: str, message: str, session_id: str | None = None, metadata: dict[str, Any] | None = None) -> None:
    """Record an audit trail event."""
    conn = get_db_connection()
    meta_str = json.dumps(metadata or {})
    with conn:
        conn.execute(
            "INSERT INTO audit_logs (timestamp, session_id, event_type, message, metadata_json) VALUES (?, ?, ?, ?, ?)",
            (time.time(), session_id, event_type, message, meta_str),
        )
    conn.close()
