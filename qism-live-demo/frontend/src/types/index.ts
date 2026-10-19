export interface TelemetryPacket {
  type: string;
  mode: "live_simulation" | "recorded_replay";
  session_id: string;
  window_index: number;
  total_windows?: number;
  timestamp: number;
  simulated_time_ms: number;
  qber: number;
  sifted_rate: number;
  signal_gain: number;
  decoy_gain: number;
  detector_imbalance: number;
  anomaly_score: number;
  threshold: number;
  is_anomaly: boolean;
  inference_latency_ms: number;
  model_id: string;
  model_components?: Record<string, number>;
  ground_truth_attack_active: boolean;
  ground_truth_attack_type: string;
  active_attack_operator?: Record<string, any> | null;
  incident?: Incident | null;
  display_badge: string;
  scenario_id: string;
}

export interface Incident {
  incident_id: string;
  session_id: string;
  detection_timestamp: number;
  simulated_time_ms: number;
  model_id: string;
  anomaly_score: number;
  threshold: number;
  severity: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  status: "NEW" | "ACKNOWLEDGED" | "INVESTIGATING" | "RESOLVED";
  ground_truth_attack_active: number;
  ground_truth_attack_type: string;
  detection_latency_windows: number;
  evidence_snapshot: Record<string, any>;
}

export interface ModelInfo {
  id: string;
  name: string;
  family: string;
  paradigm: string;
  qubits?: number;
  encoding?: string;
  circuit_depth?: number;
  is_default: boolean;
  path: string;
  status: "READY" | "MISSING";
  threshold_1pct: number;
  benchmark_auroc: number;
  description: string;
}

export interface AttackConfig {
  id: string;
  name: string;
  target: string;
  mechanism: string;
  default_severity: number;
  expected_signature: string;
  parameters: Array<{
    name: string;
    label: string;
    min: number;
    max: number;
    default: number;
  }>;
}

export interface ReplayScenario {
  id: string;
  name: string;
  attack_type: string;
  description: string;
  path: string;
  total_windows: number;
  normal_windows: string;
  attack_windows: string;
  provenance: string;
  recommended_detector: string;
}

export interface SystemHealth {
  status: "HEALTHY" | "DEGRADED" | "ERROR";
  app_name: string;
  version: string;
  uptime_seconds: number;
  active_session: {
    mode: string;
    session_id: string | null;
    simulator_status: string;
    replay_status: string;
    active_model_id: string;
  };
  components: {
    database: { status: string; path: string };
    default_model: {
      model_id: string;
      name: string;
      status: string;
      quantum_backend: string;
      circuit_depth: number;
    };
    preprocessor_pipeline: { status: string; path: string };
    dataset_repository: { status: string; path: string };
  };
  runtime: {
    python_version: string;
    os: string;
    cpu_percent: number;
    process_memory_mb: number;
    system_memory_used_pct: number;
  };
}

export interface BenchmarkRecord {
  model_id: string;
  model_name: string;
  paradigm: string;
  qubits: number | null;
  encoding: string | null;
  test_auroc: number;
  test_auprc: number;
  recall_at_1pct_fpr: number;
  mean_latency_ms: number;
  training_time_s: number;
  status: string;
}

export interface PublicationFigure {
  id: string;
  title: string;
  category: string;
  caption: string;
  file: string;
}
