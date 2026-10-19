import React, { useState } from "react";
import {
  Play,
  Pause,
  Square,
  RotateCcw,
  ShieldAlert,
  Zap,
  Radio,
  Clock,
  Cpu,
  Layers,
  AlertTriangle,
  CheckCircle2,
  FileText,
  Sliders,
} from "lucide-react";
import { AnomalyScoreChart } from "../components/AnomalyScoreChart";
import { LiveTelemetryChart } from "../components/LiveTelemetryChart";
import {
  clearAttacks,
  injectAttack,
  pauseSession,
  resumeSession,
  startReplay,
  startSimulation,
  stopSession,
  resetSession,
} from "../services/api";
import { AttackConfig, Incident, ReplayScenario, TelemetryPacket } from "../types";

interface LiveMonitorProps {
  currentPacket: TelemetryPacket | null;
  history: TelemetryPacket[];
  activeIncidents: Incident[];
  supportedAttacks: AttackConfig[];
  recordedReplays: ReplayScenario[];
  activeModelName: string;
  activeModelId: string;
  onNavigateToIncidents: () => void;
}

export const LiveMonitor: React.FC<LiveMonitorProps> = ({
  currentPacket,
  history,
  activeIncidents,
  supportedAttacks,
  recordedReplays,
  activeModelName,
  activeModelId,
  onNavigateToIncidents,
}) => {
  const [selectedAttackId, setSelectedAttackId] = useState<string>("intercept_resend");
  const [attackSeverity, setAttackSeverity] = useState<number>(0.8);
  const [attackDuration, setAttackDuration] = useState<number>(15);

  const [controlMode, setControlMode] = useState<"live" | "replay">("live");
  const [selectedReplayId, setSelectedReplayId] = useState<string>("scenario_pns_attack");
  const [replaySpeed, setReplaySpeed] = useState<number>(1.0);

  const [actionLoading, setActionLoading] = useState<boolean>(false);
  const [feedbackMsg, setFeedbackMsg] = useState<{ type: "success" | "error"; text: string } | null>(null);

  const isRunning = currentPacket?.mode === "live_simulation" || currentPacket?.mode === "recorded_replay";
  const isAttackActive = !!currentPacket?.ground_truth_attack_active;
  const isAnomaly = !!currentPacket?.is_anomaly;

  const showFeedback = (type: "success" | "error", text: string) => {
    setFeedbackMsg({ type, text });
    setTimeout(() => setFeedbackMsg(null), 4000);
  };

  // Lifecycle handlers
  const handleStart = async () => {
    setActionLoading(true);
    try {
      if (controlMode === "live") {
        await startSimulation("live_default", activeModelId);
        showFeedback("success", "Started live physical QKD simulation.");
      } else {
        await startReplay(selectedReplayId, activeModelId, replaySpeed);
        showFeedback("success", "Started recorded scenario replay with live inference.");
      }
    } catch (e: any) {
      showFeedback("error", e.message || "Failed to start session.");
    } finally {
      setActionLoading(false);
    }
  };

  const handlePause = async () => {
    try {
      await pauseSession();
      showFeedback("success", "Session paused.");
    } catch (e: any) {
      showFeedback("error", e.message);
    }
  };

  const handleResume = async () => {
    try {
      await resumeSession();
      showFeedback("success", "Session resumed.");
    } catch (e: any) {
      showFeedback("error", e.message);
    }
  };

  const handleStop = async () => {
    try {
      await stopSession();
      showFeedback("success", "Session terminated.");
    } catch (e: any) {
      showFeedback("error", e.message);
    }
  };

  const handleReset = async () => {
    try {
      await resetSession();
      showFeedback("success", "Simulator state reset to clean initial conditions.");
    } catch (e: any) {
      showFeedback("error", e.message);
    }
  };

  const handleInjectAttack = async () => {
    setActionLoading(true);
    try {
      await injectAttack(selectedAttackId, attackSeverity, attackDuration);
      showFeedback("success", `Physical attack '${selectedAttackId}' injected successfully.`);
    } catch (e: any) {
      showFeedback("error", e.message || "Attack injection failed.");
    } finally {
      setActionLoading(false);
    }
  };

  const handleClearAttack = async () => {
    try {
      await clearAttacks();
      showFeedback("success", "Active attacks cleared. Restoring clean channel.");
    } catch (e: any) {
      showFeedback("error", e.message);
    }
  };

  const selectedAttackInfo = supportedAttacks.find((a) => a.id === selectedAttackId);

  return (
    <div className="space-y-4">
      {/* Active Incident Alert Banner */}
      {isAnomaly && (
        <div className="p-3.5 rounded-lg bg-red-950/80 border border-red-500/70 text-red-200 flex items-center justify-between indicator-alert shadow-lg shadow-red-950/50">
          <div className="flex items-center space-x-3">
            <ShieldAlert className="w-6 h-6 text-red-400 flex-shrink-0" />
            <div>
              <div className="font-bold text-sm tracking-wide flex items-center space-x-2">
                <span>SECURITY INCIDENT DETECTED</span>
                <span className="text-xs px-2 py-0.5 rounded bg-red-900/80 text-red-100 font-mono">
                  SCORE: {currentPacket?.anomaly_score.toFixed(4)} &ge; τ: {currentPacket?.threshold.toFixed(4)}
                </span>
              </div>
              <p className="text-xs text-red-300/90 mt-0.5">
                Detector {activeModelName} crossed target 1% FPR decision threshold at Window #{currentPacket?.window_index}.
              </p>
            </div>
          </div>
          <button
            onClick={onNavigateToIncidents}
            className="px-3 py-1.5 rounded bg-red-800 hover:bg-red-700 text-xs font-semibold text-white transition-colors"
          >
            Investigate Incident →
          </button>
        </div>
      )}

      {/* Operational Feedback Toast */}
      {feedbackMsg && (
        <div
          className={`p-2.5 rounded text-xs font-medium flex items-center justify-between ${
            feedbackMsg.type === "success"
              ? "bg-cyan-950/80 border border-cyan-500/50 text-cyan-200"
              : "bg-red-950/80 border border-red-500/50 text-red-200"
          }`}
        >
          <span>{feedbackMsg.text}</span>
          <button onClick={() => setFeedbackMsg(null)} className="text-slate-400 hover:text-white">✕</button>
        </div>
      )}

      {/* Primary Operational Metric Strip */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        <div className="qism-panel p-3">
          <div className="flex items-center space-x-1.5 text-slate-400 text-xs mb-1">
            <Clock className="w-3.5 h-3.5 text-cyan-400" />
            <span>Simulated Time</span>
          </div>
          <div className="text-lg font-bold font-mono text-slate-100">
            {currentPacket ? `${(currentPacket.simulated_time_ms / 1000).toFixed(1)} s` : "0.0 s"}
          </div>
          <div className="text-[10px] text-slate-500 mt-0.5">
            Window #{currentPacket?.window_index ?? 0}
          </div>
        </div>

        <div className="qism-panel p-3">
          <div className="flex items-center space-x-1.5 text-slate-400 text-xs mb-1">
            <Layers className="w-3.5 h-3.5 text-cyan-400" />
            <span>Telemetry Windows</span>
          </div>
          <div className="text-lg font-bold font-mono text-cyan-300">
            {history.length}
          </div>
          <div className="text-[10px] text-slate-500 mt-0.5">
            2,000 pulses/win
          </div>
        </div>

        <div className="qism-panel p-3">
          <div className="flex items-center space-x-1.5 text-slate-400 text-xs mb-1">
            <Zap className="w-3.5 h-3.5 text-cyan-400" />
            <span>Anomaly Score</span>
          </div>
          <div
            className={`text-lg font-bold font-mono ${
              isAnomaly ? "text-red-400" : "text-cyan-300"
            }`}
          >
            {currentPacket ? currentPacket.anomaly_score.toFixed(4) : "0.0000"}
          </div>
          <div className="text-[10px] text-slate-500 mt-0.5">
            Target τ: {currentPacket?.threshold.toFixed(4) ?? "0.0452"}
          </div>
        </div>

        <div className="qism-panel p-3">
          <div className="flex items-center space-x-1.5 text-slate-400 text-xs mb-1">
            <Cpu className="w-3.5 h-3.5 text-cyan-400" />
            <span>Inference Latency</span>
          </div>
          <div className="text-lg font-bold font-mono text-slate-100">
            {currentPacket ? `${currentPacket.inference_latency_ms.toFixed(1)} ms` : "0.0 ms"}
          </div>
          <div className="text-[10px] text-slate-500 mt-0.5">
            PennyLane lightning
          </div>
        </div>

        <div className="qism-panel p-3">
          <div className="flex items-center space-x-1.5 text-slate-400 text-xs mb-1">
            <Radio className="w-3.5 h-3.5 text-cyan-400" />
            <span>Active QBER</span>
          </div>
          <div
            className={`text-lg font-bold font-mono ${
              (currentPacket?.qber ?? 0) > 0.05 ? "text-amber-400" : "text-emerald-400"
            }`}
          >
            {currentPacket ? `${(currentPacket.qber * 100).toFixed(2)}%` : "0.00%"}
          </div>
          <div className="text-[10px] text-slate-500 mt-0.5">
            Nominal &le; 3.0%
          </div>
        </div>

        <div className="qism-panel p-3">
          <div className="flex items-center space-x-1.5 text-slate-400 text-xs mb-1">
            <ShieldAlert className="w-3.5 h-3.5 text-red-400" />
            <span>Recorded Incidents</span>
          </div>
          <div className="text-lg font-bold font-mono text-red-300">
            {activeIncidents.length}
          </div>
          <div className="text-[10px] text-slate-500 mt-0.5">
            SQLite Persisted
          </div>
        </div>
      </div>

      {/* Main Monitoring Deck */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        {/* Left 8 Cols: Charts */}
        <div className="lg:col-span-8 space-y-4">
          <AnomalyScoreChart
            history={history}
            currentScore={currentPacket?.anomaly_score ?? 0}
            threshold={currentPacket?.threshold ?? 0.0452}
            isAnomaly={isAnomaly}
            modelName={activeModelName}
          />

          <LiveTelemetryChart
            history={history}
            currentQBER={currentPacket?.qber ?? 0}
            currentSiftedRate={currentPacket?.sifted_rate ?? 0}
            currentSignalGain={currentPacket?.signal_gain ?? 0}
            currentDecoyGain={currentPacket?.decoy_gain ?? 0}
          />
        </div>

        {/* Right 4 Cols: Simulator Console & Attack Injector */}
        <div className="lg:col-span-4 space-y-4">
          {/* Console Control Card */}
          <div className="qism-panel p-4">
            <div className="flex items-center justify-between mb-3 pb-2 border-b border-slate-800">
              <h3 className="text-sm font-semibold text-slate-100 flex items-center space-x-2">
                <Sliders className="w-4 h-4 text-cyan-400" />
                <span>Simulation Controller</span>
              </h3>
              <div className="flex items-center space-x-1 bg-slate-900 p-0.5 rounded text-xs">
                <button
                  onClick={() => setControlMode("live")}
                  className={`px-2 py-0.5 rounded font-medium ${
                    controlMode === "live"
                      ? "bg-cyan-950 text-cyan-300 border border-cyan-500/40"
                      : "text-slate-400 hover:text-white"
                  }`}
                >
                  Live Simulator
                </button>
                <button
                  onClick={() => setControlMode("replay")}
                  className={`px-2 py-0.5 rounded font-medium ${
                    controlMode === "replay"
                      ? "bg-amber-950 text-amber-300 border border-amber-500/40"
                      : "text-slate-400 hover:text-white"
                  }`}
                >
                  Replay Mode
                </button>
              </div>
            </div>

            {/* Replay Scenario Picker */}
            {controlMode === "replay" && (
              <div className="mb-3 space-y-2 bg-slate-900/60 p-2.5 rounded border border-slate-800">
                <label className="text-xs font-medium text-slate-300 block">
                  Select Verified Benchmark Dataset Run:
                </label>
                <select
                  value={selectedReplayId}
                  onChange={(e) => setSelectedReplayId(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-700 rounded px-2.5 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-cyan-500"
                >
                  {recordedReplays.map((r) => (
                    <option key={r.id} value={r.id}>
                      {r.name}
                    </option>
                  ))}
                </select>
                <div className="flex items-center justify-between text-[11px] text-slate-400 pt-1">
                  <span>Speed:</span>
                  <div className="space-x-1">
                    {[0.5, 1.0, 2.0].map((s) => (
                      <button
                        key={s}
                        onClick={() => setReplaySpeed(s)}
                        className={`px-2 py-0.5 rounded ${
                          replaySpeed === s
                            ? "bg-amber-900/80 text-amber-200 font-bold"
                            : "bg-slate-800 text-slate-400 hover:text-white"
                        }`}
                      >
                        {s}x
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Action Buttons */}
            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={handleStart}
                disabled={isRunning || actionLoading}
                className="flex items-center justify-center space-x-1.5 py-2 px-3 rounded bg-cyan-600 hover:bg-cyan-500 disabled:opacity-40 disabled:cursor-not-allowed text-xs font-semibold text-white transition-colors"
              >
                <Play className="w-3.5 h-3.5 fill-current" />
                <span>Start Session</span>
              </button>

              <button
                onClick={handlePause}
                disabled={!isRunning}
                className="flex items-center justify-center space-x-1.5 py-2 px-3 rounded bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-xs font-semibold text-slate-200 transition-colors"
              >
                <Pause className="w-3.5 h-3.5 fill-current" />
                <span>Pause</span>
              </button>

              <button
                onClick={handleResume}
                className="flex items-center justify-center space-x-1.5 py-2 px-3 rounded bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-slate-200 transition-colors"
              >
                <Play className="w-3.5 h-3.5" />
                <span>Resume</span>
              </button>

              <button
                onClick={handleStop}
                disabled={!isRunning}
                className="flex items-center justify-center space-x-1.5 py-2 px-3 rounded bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-xs font-semibold text-slate-200 transition-colors"
              >
                <Square className="w-3.5 h-3.5 fill-current" />
                <span>Stop</span>
              </button>
            </div>

            <button
              onClick={handleReset}
              className="w-full mt-2 py-1.5 rounded bg-slate-900 border border-slate-700/80 hover:border-slate-500 text-slate-300 text-xs font-medium flex items-center justify-center space-x-1.5 transition-colors"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Reset Simulator Physics</span>
            </button>
          </div>

          {/* Dynamic Attack Injector */}
          <div className="qism-panel p-4">
            <div className="flex items-center justify-between mb-3 pb-2 border-b border-slate-800">
              <h3 className="text-sm font-semibold text-slate-100 flex items-center space-x-2">
                <ShieldAlert className="w-4 h-4 text-red-400" />
                <span>Physical Attack Injection</span>
              </h3>
              {isAttackActive && (
                <span className="text-[10px] px-2 py-0.5 rounded bg-red-950 border border-red-500/50 text-red-300 font-bold indicator-alert">
                  INJECTED
                </span>
              )}
            </div>

            <div className="space-y-3">
              <div>
                <label className="text-xs font-medium text-slate-300 block mb-1">
                  Attack Mechanism:
                </label>
                <select
                  value={selectedAttackId}
                  onChange={(e) => setSelectedAttackId(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-700 rounded px-2.5 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-red-500"
                >
                  {supportedAttacks.map((a) => (
                    <option key={a.id} value={a.id}>
                      {a.name} ({a.target})
                    </option>
                  ))}
                </select>
              </div>

              {selectedAttackInfo && (
                <div className="text-[11px] bg-slate-900/60 p-2 rounded border border-slate-800 space-y-1">
                  <p className="text-slate-400 leading-relaxed">
                    {selectedAttackInfo.mechanism}
                  </p>
                  <p className="text-cyan-400/90 font-mono text-[10px]">
                    Signature: {selectedAttackInfo.expected_signature}
                  </p>
                </div>
              )}

              <div>
                <div className="flex justify-between text-xs text-slate-300 mb-1">
                  <span>Attack Severity:</span>
                  <span className="font-mono font-bold text-red-400">
                    {(attackSeverity * 100).toFixed(0)}%
                  </span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.05"
                  value={attackSeverity}
                  onChange={(e) => setAttackSeverity(parseFloat(e.target.value))}
                  className="w-full accent-red-500 bg-slate-800 rounded h-1.5"
                />
              </div>

              <div>
                <div className="flex justify-between text-xs text-slate-300 mb-1">
                  <span>Duration (Windows):</span>
                  <span className="font-mono text-slate-300">{attackDuration} wins</span>
                </div>
                <input
                  type="range"
                  min="5"
                  max="30"
                  step="5"
                  value={attackDuration}
                  onChange={(e) => setAttackDuration(parseInt(e.target.value, 10))}
                  className="w-full accent-cyan-500 bg-slate-800 rounded h-1.5"
                />
              </div>

              <div className="grid grid-cols-2 gap-2 pt-1">
                <button
                  onClick={handleInjectAttack}
                  disabled={!isRunning || actionLoading}
                  className="py-2 px-3 rounded bg-red-600 hover:bg-red-500 disabled:opacity-40 disabled:cursor-not-allowed text-xs font-semibold text-white transition-colors flex items-center justify-center space-x-1"
                >
                  <Zap className="w-3.5 h-3.5" />
                  <span>Inject Attack</span>
                </button>

                <button
                  onClick={handleClearAttack}
                  disabled={!isRunning}
                  className="py-2 px-3 rounded bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-xs font-medium text-slate-300 transition-colors"
                >
                  Clear Attack
                </button>
              </div>
            </div>
          </div>

          {/* Research Separation Card: Ground Truth vs Model Prediction */}
          <div className="qism-panel p-3.5 text-xs space-y-2">
            <div className="font-semibold text-slate-300 text-[11px] uppercase tracking-wider flex items-center space-x-1.5">
              <FileText className="w-3.5 h-3.5 text-slate-400" />
              <span>Scientific Ground Truth Verification</span>
            </div>

            <div className="space-y-1.5 font-mono text-[11px]">
              <div className="flex justify-between bg-slate-900/60 p-1.5 rounded">
                <span className="text-slate-400">Simulator Ground Truth:</span>
                <span className={`font-bold ${isAttackActive ? "text-red-400" : "text-emerald-400"}`}>
                  {isAttackActive ? `${currentPacket?.ground_truth_attack_type.toUpperCase()}` : "CLEAN FIBER LINK"}
                </span>
              </div>

              <div className="flex justify-between bg-slate-900/60 p-1.5 rounded">
                <span className="text-slate-400">Detector Decision:</span>
                <span className={`font-bold ${isAnomaly ? "text-red-400" : "text-emerald-400"}`}>
                  {isAnomaly ? "ANOMALOUS (Alert)" : "NORMAL (Clean)"}
                </span>
              </div>

              <div className="flex justify-between bg-slate-900/60 p-1.5 rounded">
                <span className="text-slate-400">Scientific Classification:</span>
                <span className="font-bold text-slate-200">
                  {isAttackActive && isAnomaly && <span className="text-emerald-400">TRUE POSITIVE (Detection OK)</span>}
                  {isAttackActive && !isAnomaly && <span className="text-amber-400">FALSE NEGATIVE (Under Threshold)</span>}
                  {!isAttackActive && !isAnomaly && <span className="text-slate-300">TRUE NEGATIVE (Normal Link)</span>}
                  {!isAttackActive && isAnomaly && <span className="text-red-400">FALSE ALARM (Exceeded 1% FPR)</span>}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
