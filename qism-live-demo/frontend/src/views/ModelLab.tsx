import React, { useState } from "react";
import { Cpu, CheckCircle, ShieldCheck, Zap, Layers, RefreshCw, AlertTriangle } from "lucide-react";
import { compareModels, selectModel } from "../services/api";
import { ModelInfo, TelemetryPacket } from "../types";

interface ModelLabProps {
  models: ModelInfo[];
  activeModelId: string;
  currentPacket: TelemetryPacket | null;
  onModelSelected: (modelId: string) => void;
}

export const ModelLab: React.FC<ModelLabProps> = ({
  models,
  activeModelId,
  currentPacket,
  onModelSelected,
}) => {
  const [selectedForComparison, setSelectedForComparison] = useState<string[]>([
    "PITQSVDD_6Q_Angle",
    "PITQSVDD_4Q_Angle_Ref",
    "ClassicalTemporalAutoencoder",
    "IsolationForest_PCA6",
  ]);

  const [comparisonResults, setComparisonResults] = useState<any[] | null>(null);
  const [comparing, setComparing] = useState<boolean>(false);
  const [switchingModel, setSwitchingModel] = useState<string | null>(null);

  const handleSelectModel = async (modelId: string) => {
    setSwitchingModel(modelId);
    try {
      await selectModel(modelId);
      onModelSelected(modelId);
    } catch (e) {
      console.error(e);
    } finally {
      setSwitchingModel(null);
    }
  };

  const handleToggleCompare = (modelId: string) => {
    if (selectedForComparison.includes(modelId)) {
      setSelectedForComparison(selectedForComparison.filter((id) => id !== modelId));
    } else {
      if (selectedForComparison.length >= 6) return;
      setSelectedForComparison([...selectedForComparison, modelId]);
    }
  };

  const handleRunComparison = async () => {
    setComparing(true);
    try {
      const res = await compareModels(selectedForComparison);
      setComparisonResults(res.comparisons);
    } catch (e) {
      console.error(e);
    } finally {
      setComparing(false);
    }
  };

  return (
    <div className="space-y-5">
      {/* Header bar */}
      <div className="qism-panel p-4 flex flex-col md:flex-row md:items-center justify-between gap-3">
        <div>
          <h2 className="text-base font-bold text-slate-100 flex items-center space-x-2">
            <Cpu className="w-5 h-5 text-cyan-400" />
            <span>Model Lab & Comparative Inference Console</span>
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Default flagship detector: <span className="font-semibold text-cyan-300">6-qubit PI-TQSVDD</span>. Compare real online inference across classical, quantum, and baseline checkpoints.
          </p>
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={handleRunComparison}
            disabled={comparing || selectedForComparison.length === 0}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded bg-cyan-600 hover:bg-cyan-500 disabled:opacity-40 text-xs font-semibold text-white transition-colors"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${comparing ? "animate-spin" : ""}`} />
            <span>Run Multi-Model Comparison ({selectedForComparison.length})</span>
          </button>
        </div>
      </div>

      {/* Side-by-Side Comparison Output */}
      {comparisonResults && (
        <div className="qism-panel p-4 space-y-3 bg-slate-900/90 border border-cyan-500/40">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-bold uppercase tracking-wider text-cyan-300 flex items-center space-x-2">
              <Zap className="w-4 h-4 text-cyan-400" />
              <span>Live Comparative Inference Results (Identical Telemetry Window)</span>
            </h3>
            <span className="text-[11px] text-slate-400">
              Causal window L=3 · Same physical features
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
            {comparisonResults.map((r) => {
              const mInfo = models.find((m) => m.id === r.model_id);
              const isAlert = r.anomaly_score >= r.threshold;
              return (
                <div
                  key={r.model_id}
                  className={`p-3 rounded border text-xs space-y-2 ${
                    isAlert
                      ? "bg-red-950/40 border-red-500/50"
                      : "bg-slate-950/60 border-slate-800"
                  }`}
                >
                  <div className="flex justify-between items-start">
                    <span className="font-bold text-slate-200 truncate" title={mInfo?.name || r.model_id}>
                      {mInfo?.name || r.model_id}
                    </span>
                    <span
                      className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                        isAlert
                          ? "bg-red-900 text-red-200"
                          : "bg-emerald-900/70 text-emerald-200"
                      }`}
                    >
                      {isAlert ? "ANOMALY" : "NORMAL"}
                    </span>
                  </div>

                  <div className="font-mono space-y-1">
                    <div className="flex justify-between">
                      <span className="text-slate-400">Score:</span>
                      <span className={`font-bold ${isAlert ? "text-red-400" : "text-cyan-300"}`}>
                        {r.anomaly_score.toFixed(4)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Threshold:</span>
                      <span className="text-slate-300">{r.threshold.toFixed(4)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Latency:</span>
                      <span className="text-slate-300">{r.latency_ms.toFixed(1)} ms</span>
                    </div>
                  </div>

                  {r.components && Object.keys(r.components).length > 0 && (
                    <div className="pt-1.5 border-t border-slate-800 text-[10px] text-slate-400 font-mono">
                      {Object.entries(r.components).map(([k, v]) => (
                        <div key={k} className="flex justify-between">
                          <span className="truncate pr-1">{k}:</span>
                          <span className="text-slate-300 font-semibold">{Number(v).toFixed(4)}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Model Inventory Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {models.map((model) => {
          const isActive = activeModelId === model.id;
          const isFlagship = model.id === "PITQSVDD_6Q_Angle";
          const isComparing = selectedForComparison.includes(model.id);

          return (
            <div
              key={model.id}
              className={`qism-panel p-4 flex flex-col justify-between transition-all ${
                isActive
                  ? "border-cyan-500 shadow-md shadow-cyan-950/60 bg-slate-900/90"
                  : "hover:border-slate-700"
              }`}
            >
              <div>
                {/* Badges bar */}
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-1.5">
                    {isFlagship && (
                      <span className="px-2 py-0.5 rounded bg-cyan-950 border border-cyan-500 text-cyan-300 text-[10px] font-bold">
                        FLAGSHIP DEFAULT
                      </span>
                    )}
                    <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-300 text-[10px] uppercase font-mono">
                      {model.family.replace("_", " ")}
                    </span>
                  </div>

                  <span
                    className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                      model.status === "READY"
                        ? "bg-emerald-950/80 text-emerald-300 border border-emerald-500/40"
                        : "bg-red-950 text-red-400 border border-red-500/40"
                    }`}
                  >
                    {model.status}
                  </span>
                </div>

                {/* Model Title */}
                <h3 className="text-sm font-bold text-slate-100 mb-1">
                  {model.name}
                </h3>
                <p className="text-xs text-slate-400 leading-relaxed mb-3">
                  {model.description}
                </p>

                {/* Technical Specs */}
                <div className="bg-slate-950/60 p-2.5 rounded border border-slate-800 text-xs font-mono space-y-1 mb-3">
                  {model.qubits && (
                    <div className="flex justify-between">
                      <span className="text-slate-500">Qubits:</span>
                      <span className="text-cyan-300 font-bold">{model.qubits} Qubits</span>
                    </div>
                  )}
                  {model.encoding && (
                    <div className="flex justify-between">
                      <span className="text-slate-500">Encoding:</span>
                      <span className="text-slate-300 capitalize">{model.encoding}</span>
                    </div>
                  )}
                  <div className="flex justify-between">
                    <span className="text-slate-500">Calibrated Threshold:</span>
                    <span className="text-slate-200 font-bold">{model.threshold_1pct.toFixed(4)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Benchmark AUROC:</span>
                    <span className="text-emerald-400 font-bold">
                      {(model.benchmark_auroc * 100).toFixed(2)}%
                    </span>
                  </div>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="pt-2 border-t border-slate-800 flex items-center justify-between gap-2">
                <label className="flex items-center space-x-1.5 text-xs text-slate-400 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={isComparing}
                    onChange={() => handleToggleCompare(model.id)}
                    className="accent-cyan-500"
                  />
                  <span>Compare</span>
                </label>

                <button
                  onClick={() => handleSelectModel(model.id)}
                  disabled={isActive || switchingModel === model.id || model.status !== "READY"}
                  className={`px-3 py-1.5 rounded text-xs font-semibold transition-colors ${
                    isActive
                      ? "bg-cyan-950 text-cyan-300 border border-cyan-500/50 cursor-default"
                      : "bg-slate-800 hover:bg-slate-700 text-slate-200"
                  }`}
                >
                  {isActive ? "Active Detector" : switchingModel === model.id ? "Loading..." : "Activate"}
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
