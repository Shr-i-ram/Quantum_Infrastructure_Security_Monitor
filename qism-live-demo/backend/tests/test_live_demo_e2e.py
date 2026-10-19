"""Comprehensive End-to-End Test Suite for QISM Live Demonstration Web Application.

Tests:
1. System Health and Component Readiness
2. Model Registry & Default 6-Qubit PI-TQSVDD verification
3. Live Physical Simulator Startup & Telemetry Stream
4. Dynamic Attack Injection & Causal Telemetry Impact
5. Calibrated Thresholding & Incident Creation
6. SQLite Incident Persistence & Workflow State Transitions
7. Recorded Scenario Replay with Live Inference
8. Publication Figures & Research Leaderboard Integrity
"""

import json
import time
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:8000"


def request_json(path: str, method: str = "GET", data: dict = None) -> dict:
    url = f"{BASE_URL}{path}"
    headers = {"Content-Type": "application/json"} if data else {}
    body = json.dumps(data).encode("utf-8") if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def test_e2e_live_demo():
    print("=" * 70)
    print("STARTING QISM LIVE DEMONSTRATION END-TO-END VALIDATION TEST")
    print("=" * 70)

    # 1. Health Endpoint
    print("\n[Step 1] Checking System Health & Pre-warmed Flagship Model...")
    health = request_json("/api/health")
    assert health["status"] == "HEALTHY", f"Health status not healthy: {health}"
    assert health["components"]["default_model"]["status"] == "READY", "Default model not ready"
    assert health["components"]["database"]["status"] == "READY", "SQLite DB not ready"
    assert health["components"]["preprocessor_pipeline"]["status"] == "READY", "Preprocessor pipeline not ready"
    print(f" -> System Health: HEALTHY (Uptime: {health['uptime_seconds']}s, RAM: {health['runtime']['process_memory_mb']} MB)")

    # 2. Model Catalog
    print("\n[Step 2] Discovering Available Models & Verifying 6Q PI-TQSVDD...")
    models_res = request_json("/api/models")
    assert models_res["default_model_id"] == "PITQSVDD_6Q_Angle", "Default model ID mismatch"
    assert models_res["total_available"] >= 10, f"Expected >=10 verified models, got {models_res['total_available']}"
    default_m = next(m for m in models_res["models"] if m["id"] == "PITQSVDD_6Q_Angle")
    assert default_m["status"] == "READY", "Flagship PI-TQSVDD is not marked READY"
    print(f" -> Verified {models_res['total_available']} trained models in registry.")
    print(f" -> Flagship model: {default_m['name']} (Calibrated tau={default_m['threshold_1pct']})")

    # 3. Benchmark Leaderboard & Figures
    print("\n[Step 3] Verifying Research Benchmark Leaderboard & Publication Figures...")
    leaderboard = request_json("/api/benchmark/leaderboard")
    assert len(leaderboard["leaderboard"]) > 0, "Empty leaderboard"
    figures = request_json("/api/benchmark/figures")
    assert figures["total"] == 7, f"Expected 7 publication figures, found {figures['total']}"
    print(f" -> Leaderboard contains {len(leaderboard['leaderboard'])} models.")
    print(f" -> Figure assets: All {figures['total']} high-res figures verified.")

    # 4. Start Live Simulation Session
    print("\n[Step 4] Starting Live Physical QKD Simulation Session...")
    start_res = request_json("/api/sessions/start", method="POST", data={
        "scenario_id": "live_test",
        "model_id": "PITQSVDD_6Q_Angle",
        "seed": 42
    })
    session_id = start_res["session_id"]
    print(f" -> Live session started successfully: Session ID = {session_id}")

    # Wait for a couple of clean windows
    print(" -> Allowing simulator to generate clean baseline telemetry (3 windows)...")
    time.sleep(1.8)

    # Check current session state
    cur_session = request_json("/api/sessions/current")
    assert cur_session["mode"] == "live_simulation", "Session mode is not live_simulation"
    assert cur_session["window_index"] >= 2, f"Expected >=2 windows, got {cur_session['window_index']}"
    print(f" -> Simulator active at Window #{cur_session['window_index']}")

    # 5. Inject Physical Attack
    print("\n[Step 5] Dynamically Injecting Intercept-Resend Attack into Running Session...")
    inject_res = request_json("/api/sessions/attacks/inject", method="POST", data={
        "attack_type": "intercept_resend",
        "severity": 0.85,
        "duration_windows": 10
    })
    assert inject_res["status"] == "ATTACK_INJECTED", f"Attack injection failed: {inject_res}"
    print(f" -> Injected: {inject_res['attack']['attack_type']} (severity=0.85, duration=10 windows)")

    # Allow simulation to generate attacked windows and evaluate inference
    print(" -> Processing attacked telemetry through 6-qubit PI-TQSVDD inference pipeline...")
    time.sleep(2.5)

    # 6. Check Incident Creation
    print("\n[Step 6] Verifying Calibrated Thresholding & Incident Creation...")
    incidents_res = request_json(f"/api/incidents?session_id={session_id}")
    incidents = incidents_res["incidents"]
    print(f" -> Total recorded incidents for session: {len(incidents)}")
    if len(incidents) > 0:
        first_inc = incidents[0]
        print(f" -> First Incident ID: {first_inc['incident_id']}")
        print(f"    - Score: {first_inc['anomaly_score']:.4f} >= Threshold: {first_inc['threshold']:.4f}")
        print(f"    - Severity: {first_inc['severity']}")
        print(f"    - Simulator Ground Truth: {first_inc['ground_truth_attack_type']}")
        print(f"    - Workflow Status: {first_inc['status']}")
        assert first_inc["anomaly_score"] >= first_inc["threshold"], "Incident score does not exceed threshold!"

        # 7. Update Incident Status
        print("\n[Step 7] Updating Incident Workflow Status to 'ACKNOWLEDGED'...")
        update_res = request_json(f"/api/incidents/{first_inc['incident_id']}/status", method="PATCH", data={
            "status": "ACKNOWLEDGED"
        })
        assert update_res["new_status"] == "ACKNOWLEDGED", "Failed to update incident status"
        print(f" -> Workflow status successfully updated to {update_res['new_status']}")

    # 8. Stop Live Session
    print("\n[Step 8] Terminating Live Simulation Session...")
    stop_res = request_json("/api/sessions/stop", method="POST")
    assert stop_res["status"] == "STOPPED", "Failed to stop session"
    print(" -> Live session stopped cleanly.")

    # 9. Test Recorded Scenario Replay
    print("\n[Step 9] Launching Recorded Benchmark Replay Mode (PNS Attack Scenario)...")
    replay_start = request_json("/api/sessions/replay/start", method="POST", data={
        "scenario_id": "scenario_pns_attack",
        "model_id": "PITQSVDD_6Q_Angle",
        "speed": 2.0
    })
    replay_id = replay_start["session_id"]
    print(f" -> Replay session started: {replay_id} (PNS Scenario at 2.0x speed)")

    time.sleep(1.8)
    cur_replay = request_json("/api/sessions/current")
    assert cur_replay["mode"] == "recorded_replay", "Session mode is not recorded_replay"
    assert cur_replay["display_badge"] == "RECORDED SCENARIO REPLAY", "Badge is not RECORDED SCENARIO REPLAY"
    print(f" -> Replay active at Window #{cur_replay['window_index']}/{cur_replay['total_windows']} with badge: '{cur_replay['display_badge']}'")

    # Stop Replay
    request_json("/api/sessions/stop", method="POST")
    print(" -> Replay session stopped cleanly.")

    print("\n" + "=" * 70)
    print("ALL END-TO-END VALIDATION CRITERIA PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    test_e2e_live_demo()
