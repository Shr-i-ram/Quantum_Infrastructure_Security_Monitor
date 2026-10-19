import React, { useState } from "react";
import { Database, Search, Download, LineChart, HelpCircle } from "lucide-react";
import { TelemetryPacket } from "../types";

interface TelemetryExplorerProps {
  currentPacket: TelemetryPacket | null;
  history: TelemetryPacket[];
  schema: {
    total_features: number;
    features: string[];
    categories: Record<string, string[]>;
    primary_signals: Array<{ id: string; label: string; unit: string; nominal: string }>;
  } | null;
}

export const TelemetryExplorer: React.FC<TelemetryExplorerProps> = ({
  currentPacket,
  history,
  schema,
}) => {
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [selectedCategory, setSelectedCategory] = useState<string>("all");
  const [selectedFeature, setSelectedFeature] = useState<string>("QBER");

  const categories = schema?.categories ? Object.keys(schema.categories) : [];

  const currentFeatures = (currentPacket as any)?.features || {};

  // Filter features
  const allFeatures = schema?.features || [];
  const filteredFeatures = allFeatures.filter((f) => {
    if (selectedCategory !== "all" && schema?.categories[selectedCategory]) {
      if (!schema.categories[selectedCategory].includes(f)) return false;
    }
    if (searchQuery) {
      return f.toLowerCase().includes(searchQuery.toLowerCase());
    }
    return true;
  });

  // Feature timeseries history
  const width = 640;
  const height = 180;
  const padding = { top: 20, right: 25, bottom: 25, left: 55 };
  const plotW = width - padding.left - padding.right;
  const plotH = height - padding.top - padding.bottom;

  const featureValues = history.map((h) => {
    const fMap = (h as any).features || {};
    if (fMap[selectedFeature] !== undefined) return Number(fMap[selectedFeature]);
    // check direct packet field
    if (selectedFeature.toLowerCase() === "qber") return h.qber;
    if (selectedFeature.toLowerCase().includes("sifted")) return h.sifted_rate;
    if (selectedFeature.toLowerCase().includes("gain_signal")) return h.signal_gain;
    if (selectedFeature.toLowerCase().includes("gain_decoy")) return h.decoy_gain;
    return 0.0;
  });

  const maxVal = Math.max(1e-4, ...featureValues.map((v) => Math.abs(v))) * 1.25;
  const minVal = Math.min(0, ...featureValues);

  const getX = (idx: number) => padding.left + (idx / Math.max(history.length - 1, 1)) * plotW;
  const getY = (val: number) => {
    const range = maxVal - minVal;
    if (range === 0) return padding.top + plotH / 2;
    return padding.top + plotH - ((val - minVal) / range) * plotH;
  };

  const featurePath = featureValues
    .map((v, i) => `${i === 0 ? "M" : "L"} ${getX(i).toFixed(1)} ${getY(v).toFixed(1)}`)
    .join(" ");

  const exportCSV = () => {
    if (history.length === 0) return;
    const headers = ["window_index", "simulated_time_ms", "qber", "sifted_rate", "anomaly_score", "threshold", "ground_truth_attack_type"];
    const rows = history.map((h) => [
      h.window_index,
      h.simulated_time_ms,
      h.qber,
      h.sifted_rate,
      h.anomaly_score,
      h.threshold,
      h.ground_truth_attack_type,
    ]);
    const csvContent = "data:text/csv;charset=utf-8," + [headers.join(","), ...rows.map((e) => e.join(","))].join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `qism_telemetry_export_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="space-y-4">
      {/* Header bar */}
      <div className="qism-panel p-4 flex flex-col md:flex-row md:items-center justify-between gap-3">
        <div>
          <h2 className="text-base font-bold text-slate-100 flex items-center space-x-2">
            <Database className="w-5 h-5 text-cyan-400" />
            <span>76-Feature Continuous Telemetry Explorer</span>
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Full physical observable telemetry dictionary extracted from Bob SPAD measurement clicks and environmental sensors.
          </p>
        </div>

        <div className="flex items-center space-x-3">
          <button
            onClick={exportCSV}
            disabled={history.length === 0}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-xs font-semibold text-slate-200 transition-colors"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Export CSV</span>
          </button>
        </div>
      </div>

      {/* Selected Feature Timeseries Preview */}
      <div className="qism-panel p-4">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center space-x-2">
            <LineChart className="w-4 h-4 text-cyan-400" />
            <h3 className="text-xs font-semibold text-slate-200 uppercase tracking-wider">
              Feature Trend: <span className="font-mono text-cyan-300 font-bold">{selectedFeature}</span>
            </h3>
          </div>
          <span className="text-xs text-slate-400 font-mono">
            Current: {featureValues.length > 0 ? featureValues[featureValues.length - 1].toFixed(5) : "N/A"}
          </span>
        </div>

        <div className="bg-slate-950/70 rounded border border-slate-800 p-2">
          <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-32 select-none">
            {/* Grid */}
            {[0, 0.5, 1.0].map((frac, idx) => {
              const y = padding.top + plotH * (1 - frac);
              const val = minVal + frac * (maxVal - minVal);
              return (
                <g key={idx}>
                  <line
                    x1={padding.left}
                    y1={y}
                    x2={width - padding.right}
                    y2={y}
                    stroke="#1e293b"
                    strokeDasharray="2,3"
                  />
                  <text
                    x={padding.left - 8}
                    y={y + 3.5}
                    textAnchor="end"
                    className="fill-slate-500 text-[10px] font-mono"
                  >
                    {val.toFixed(4)}
                  </text>
                </g>
              );
            })}

            {featurePath && (
              <path
                d={featurePath}
                fill="none"
                stroke="#06b6d4"
                strokeWidth="2"
                strokeLinecap="round"
              />
            )}
          </svg>
        </div>
      </div>

      {/* Feature Catalog & Categories */}
      <div className="qism-panel p-4 space-y-3">
        {/* Category Tabs */}
        <div className="flex flex-wrap items-center gap-1.5 pb-2 border-b border-slate-800 text-xs">
          <button
            onClick={() => setSelectedCategory("all")}
            className={`px-3 py-1 rounded font-medium transition-colors ${
              selectedCategory === "all"
                ? "bg-cyan-950 text-cyan-300 border border-cyan-500/50"
                : "bg-slate-900 text-slate-400 hover:text-white"
            }`}
          >
            All Features ({allFeatures.length})
          </button>
          {categories.map((cat) => (
            <button
              key={cat}
              onClick={() => setSelectedCategory(cat)}
              className={`px-3 py-1 rounded font-medium capitalize transition-colors ${
                selectedCategory === cat
                  ? "bg-cyan-950 text-cyan-300 border border-cyan-500/50"
                  : "bg-slate-900 text-slate-400 hover:text-white"
              }`}
            >
              {cat.replace("_", " ")} ({schema?.categories[cat]?.length ?? 0})
            </button>
          ))}
        </div>

        {/* Search input */}
        <div className="relative">
          <Search className="w-3.5 h-3.5 text-slate-400 absolute left-2.5 top-2.5" />
          <input
            type="text"
            placeholder="Search features (e.g. qber, dark_count, dead_time, gain)..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-slate-950 border border-slate-700 rounded pl-8 pr-3 py-1.5 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500"
          />
        </div>

        {/* Feature Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2 max-h-96 overflow-y-auto pr-1">
          {filteredFeatures.map((feat) => {
            const isSelected = selectedFeature === feat;
            const liveVal = currentFeatures[feat] !== undefined ? currentFeatures[feat] : null;

            return (
              <div
                key={feat}
                onClick={() => setSelectedFeature(feat)}
                className={`p-2.5 rounded border text-xs cursor-pointer transition-all ${
                  isSelected
                    ? "bg-cyan-950/70 border-cyan-500/70 shadow-sm"
                    : "bg-slate-900/50 border-slate-800 hover:border-slate-700"
                }`}
              >
                <div className="font-mono text-slate-300 font-medium truncate mb-1" title={feat}>
                  {feat}
                </div>
                <div className="flex justify-between items-baseline font-mono">
                  <span className="text-[10px] text-slate-500">Live Value:</span>
                  <span className="font-bold text-cyan-300 text-[11px]">
                    {liveVal !== null ? (typeof liveVal === "number" ? liveVal.toFixed(5) : String(liveVal)) : "N/A"}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
