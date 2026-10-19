import {
  AttackConfig,
  BenchmarkRecord,
  Incident,
  ModelInfo,
  PublicationFigure,
  ReplayScenario,
  SystemHealth,
} from "../types";

const API_BASE = window.location.origin.includes(":5173")
  ? "http://127.0.0.1:8000/api"
  : "/api";

export const WS_URL = window.location.origin.includes(":5173")
  ? "ws://127.0.0.1:8000/ws/telemetry"
  : `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}/ws/telemetry`;

export async function fetchHealth(): Promise<SystemHealth> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error("Health check failed");
  return res.json();
}

export async function fetchModels(): Promise<{
  models: ModelInfo[];
  default_model_id: string;
  active_model_id: string;
  total_available: number;
}> {
  const res = await fetch(`${API_BASE}/models`);
  if (!res.ok) throw new Error("Failed to load models");
  return res.json();
}

export async function selectModel(modelId: string): Promise<any> {
  const res = await fetch(`${API_BASE}/models/select`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_id: modelId }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Failed to select model");
  }
  return res.json();
}

export async function compareModels(modelIds: string[]): Promise<any> {
  const res = await fetch(`${API_BASE}/models/compare`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_ids: modelIds }),
  });
  if (!res.ok) throw new Error("Failed to compare models");
  return res.json();
}

export async function fetchSupportedAttacks(): Promise<{ attacks: AttackConfig[] }> {
  const res = await fetch(`${API_BASE}/scenarios/attacks`);
  if (!res.ok) throw new Error("Failed to load attacks");
  return res.json();
}

export async function fetchRecordedReplays(): Promise<{ replays: ReplayScenario[] }> {
  const res = await fetch(`${API_BASE}/scenarios/replays`);
  if (!res.ok) throw new Error("Failed to load replays");
  return res.json();
}

export async function fetchCurrentSession(): Promise<any> {
  const res = await fetch(`${API_BASE}/sessions/current`);
  if (!res.ok) throw new Error("Failed to fetch session");
  return res.json();
}

export async function startSimulation(scenarioId = "live_default", modelId = "PITQSVDD_6Q_Angle"): Promise<any> {
  const res = await fetch(`${API_BASE}/sessions/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scenario_id: scenarioId, model_id: modelId, seed: 42 }),
  });
  if (!res.ok) throw new Error("Failed to start simulation");
  return res.json();
}

export async function startReplay(scenarioId: string, modelId = "PITQSVDD_6Q_Angle", speed = 1.0): Promise<any> {
  const res = await fetch(`${API_BASE}/sessions/replay/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scenario_id: scenarioId, model_id: modelId, speed, live_inference: true }),
  });
  if (!res.ok) throw new Error("Failed to start replay");
  return res.json();
}

export async function pauseSession(): Promise<any> {
  const res = await fetch(`${API_BASE}/sessions/pause`, { method: "POST" });
  return res.json();
}

export async function resumeSession(): Promise<any> {
  const res = await fetch(`${API_BASE}/sessions/resume`, { method: "POST" });
  return res.json();
}

export async function stopSession(): Promise<any> {
  const res = await fetch(`${API_BASE}/sessions/stop`, { method: "POST" });
  return res.json();
}

export async function resetSession(): Promise<any> {
  const res = await fetch(`${API_BASE}/sessions/reset`, { method: "POST" });
  return res.json();
}

export async function injectAttack(
  attackType: string,
  severity = 0.8,
  durationWindows = 15,
  params: Record<string, any> = {}
): Promise<any> {
  const res = await fetch(`${API_BASE}/sessions/attacks/inject`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      attack_type: attackType,
      severity,
      duration_windows: durationWindows,
      parameters: params,
    }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Failed to inject attack");
  }
  return res.json();
}

export async function clearAttacks(): Promise<any> {
  const res = await fetch(`${API_BASE}/sessions/attacks/clear`, { method: "POST" });
  return res.json();
}

export async function fetchIncidents(params?: {
  sessionId?: string;
  severity?: string;
  status?: string;
}): Promise<{ incidents: Incident[]; count: number }> {
  const q = new URLSearchParams();
  if (params?.sessionId) q.set("session_id", params.sessionId);
  if (params?.severity) q.set("severity", params.severity);
  if (params?.status) q.set("status", params.status);
  const res = await fetch(`${API_BASE}/incidents?${q.toString()}`);
  if (!res.ok) throw new Error("Failed to load incidents");
  return res.json();
}

export async function updateIncidentStatus(incidentId: string, status: string): Promise<any> {
  const res = await fetch(`${API_BASE}/incidents/${incidentId}/status`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ status }),
  });
  if (!res.ok) throw new Error("Failed to update incident status");
  return res.json();
}

export async function fetchTelemetrySchema(): Promise<any> {
  const res = await fetch(`${API_BASE}/telemetry/schema`);
  if (!res.ok) throw new Error("Failed to load telemetry schema");
  return res.json();
}

export async function fetchBenchmarkLeaderboard(): Promise<{
  leaderboard: BenchmarkRecord[];
  total_models: number;
  source: string;
  flagship_model: string;
}> {
  const res = await fetch(`${API_BASE}/benchmark/leaderboard`);
  if (!res.ok) throw new Error("Failed to load leaderboard");
  return res.json();
}

export async function fetchFigures(): Promise<{ figures: PublicationFigure[]; total: number }> {
  const res = await fetch(`${API_BASE}/benchmark/figures`);
  if (!res.ok) throw new Error("Failed to load figures");
  return res.json();
}

export function getFigureImageUrl(fileRelPath: string): string {
  return `${API_BASE}/benchmark/figures/${fileRelPath}`;
}
