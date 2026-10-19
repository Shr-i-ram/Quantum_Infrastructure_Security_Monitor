import React, { useState } from "react";
import { AlertTriangle, ShieldCheck, HelpCircle } from "lucide-react";
import { TelemetryPacket } from "../types";

interface AnomalyScoreChartProps {
  history: TelemetryPacket[];
  currentScore: number;
  threshold: number;
  isAnomaly: boolean;
  modelName: string;
}

export const AnomalyScoreChart: React.FC<AnomalyScoreChartProps> = ({
  history,
  currentScore,
  threshold,
  isAnomaly,
  modelName,
}) => {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  const width = 680;
  const height = 240;
  const padding = { top: 25, right: 30, bottom: 35, left: 55 };

  // Determine vertical scale (dynamic maximum based on history and threshold)
  const maxScore = Math.max(
    0.15,
    threshold * 2.2,
    ...history.map((h) => h.anomaly_score)
  );

  const plotW = width - padding.left - padding.right;
  const plotH = height - padding.top - padding.bottom;

  const pointsCount = Math.max(history.length, 2);

  const getX = (idx: number) => padding.left + (idx / (pointsCount - 1)) * plotW;
  const getY = (val: number) => padding.top + plotH - (Math.min(val, maxScore) / maxScore) * plotH;

  // Build SVG path for anomaly score
  const scorePath = history
    .map((h, i) => `${i === 0 ? "M" : "L"} ${getX(i).toFixed(1)} ${getY(h.anomaly_score).toFixed(1)}`)
    .join(" ");

  const areaPath =
    history.length > 0
      ? `${scorePath} L ${getX(history.length - 1).toFixed(1)} ${(padding.top + plotH).toFixed(1)} L ${getX(0).toFixed(1)} ${(padding.top + plotH).toFixed(1)} Z`
      : "";

  const thresholdY = getY(threshold);

  const hoveredItem = hoveredIndex !== null && hoveredIndex < history.length ? history[hoveredIndex] : null;

  return (
    <div className="qism-panel p-4 flex flex-col justify-between">
      {/* Header bar */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center space-x-2">
          <div className="w-2.5 h-2.5 rounded-full bg-cyan-400" />
          <h3 className="text-sm font-semibold text-slate-100">
            Detector Anomaly Score vs Decision Threshold
          </h3>
          <span className="text-xs px-2 py-0.5 rounded bg-slate-800 text-slate-400 font-mono">
            {modelName}
          </span>
        </div>

        {/* Current State Pill */}
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-1.5 text-xs text-slate-400">
            <span className="w-3 h-0.5 bg-red-400 border-dashed" />
            <span>Threshold (τ): <span className="font-mono text-red-300 font-semibold">{threshold.toFixed(4)}</span></span>
          </div>

          <div
            className={`flex items-center space-x-1.5 px-2.5 py-0.5 rounded text-xs font-semibold ${
              isAnomaly
                ? "bg-red-950/80 text-red-300 border border-red-500/60 indicator-alert"
                : "bg-emerald-950/60 text-emerald-300 border border-emerald-500/40"
            }`}
          >
            {isAnomaly ? <AlertTriangle className="w-3.5 h-3.5" /> : <ShieldCheck className="w-3.5 h-3.5" />}
            <span>{isAnomaly ? "ANOMALY CONFIRMED" : "NORMAL OPERATIONAL REGIME"}</span>
          </div>
        </div>
      </div>

      {/* SVG Plot */}
      <div className="relative w-full overflow-hidden bg-slate-950/60 rounded border border-slate-800/80">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-48 select-none"
          onMouseLeave={() => setHoveredIndex(null)}
        >
          <defs>
            <linearGradient id="scoreGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.4" />
              <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.0" />
            </linearGradient>
            <linearGradient id="alertGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#ef4444" stopOpacity="0.25" />
              <stop offset="100%" stopColor="#ef4444" stopOpacity="0.02" />
            </linearGradient>
          </defs>

          {/* Grid lines */}
          {[0, 0.25, 0.5, 0.75, 1.0].map((frac, idx) => {
            const y = padding.top + plotH * (1 - frac);
            const val = frac * maxScore;
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
                  {val.toFixed(3)}
                </text>
              </g>
            );
          })}

          {/* Ground truth attack interval shading */}
          {history.map((h, i) => {
            if (!h.ground_truth_attack_active) return null;
            const x = getX(i);
            const stepW = plotW / Math.max(history.length, 1);
            return (
              <rect
                key={i}
                x={x - stepW / 2}
                y={padding.top}
                width={stepW}
                height={plotH}
                fill="url(#alertGradient)"
              />
            );
          })}

          {/* Area fill under anomaly score curve */}
          {areaPath && <path d={areaPath} fill="url(#scoreGradient)" />}

          {/* Calibrated Threshold Line (τ) */}
          <line
            x1={padding.left}
            y1={thresholdY}
            x2={width - padding.right}
            y2={thresholdY}
            stroke="#f87171"
            strokeWidth="1.5"
            strokeDasharray="4,3"
          />

          {/* Score line */}
          {scorePath && (
            <path
              d={scorePath}
              fill="none"
              stroke="#22d3ee"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          )}

          {/* Data points & anomaly highlights */}
          {history.map((h, i) => {
            const cx = getX(i);
            const cy = getY(h.anomaly_score);
            const isAnomPoint = h.anomaly_score >= threshold;
            return (
              <g key={i}>
                <circle
                  cx={cx}
                  cy={cy}
                  r={isAnomPoint ? 4 : 2}
                  fill={isAnomPoint ? "#ef4444" : "#06b6d4"}
                  stroke={isAnomPoint ? "#fecaca" : "#083344"}
                  strokeWidth={isAnomPoint ? 1.5 : 0.5}
                  className="cursor-pointer hover:r-5 transition-all"
                  onMouseEnter={() => setHoveredIndex(i)}
                />
              </g>
            );
          })}

          {/* Hover Crosshair & Details */}
          {hoveredItem && hoveredIndex !== null && (
            <g>
              <line
                x1={getX(hoveredIndex)}
                y1={padding.top}
                x2={getX(hoveredIndex)}
                y2={padding.top + plotH}
                stroke="#64748b"
                strokeDasharray="2,2"
              />
              <circle
                cx={getX(hoveredIndex)}
                cy={getY(hoveredItem.anomaly_score)}
                r={5}
                fill="#f8fafc"
                stroke="#06b6d4"
                strokeWidth="2"
              />
            </g>
          )}

          {/* X Axis baseline */}
          <line
            x1={padding.left}
            y1={padding.top + plotH}
            x2={width - padding.right}
            y2={padding.top + plotH}
            stroke="#334155"
          />
          <text
            x={width / 2}
            y={height - 8}
            textAnchor="middle"
            className="fill-slate-500 text-[10px]"
          >
            Temporal Telemetry Sliding Windows (Causal Online Sequence, L=3)
          </text>
        </svg>

        {/* Hover Tooltip Card */}
        {hoveredItem && hoveredIndex !== null && (
          <div
            className="absolute top-2 right-4 bg-slate-900/95 border border-slate-700 p-2.5 rounded text-xs shadow-xl backdrop-blur font-mono pointer-events-none"
          >
            <div className="text-slate-400 font-semibold mb-1">
              Window #{hoveredItem.window_index}
            </div>
            <div className="flex justify-between space-x-4">
              <span className="text-slate-400">Anomaly Score:</span>
              <span className={`font-bold ${hoveredItem.anomaly_score >= threshold ? "text-red-400" : "text-cyan-300"}`}>
                {hoveredItem.anomaly_score.toFixed(4)}
              </span>
            </div>
            <div className="flex justify-between space-x-4">
              <span className="text-slate-400">Calibrated Threshold:</span>
              <span className="text-slate-300">{threshold.toFixed(4)}</span>
            </div>
            <div className="flex justify-between space-x-4">
              <span className="text-slate-400">Decision:</span>
              <span className={hoveredItem.anomaly_score >= threshold ? "text-red-400 font-bold" : "text-emerald-400"}>
                {hoveredItem.anomaly_score >= threshold ? "ALERT (Anomaly)" : "NORMAL"}
              </span>
            </div>
            <div className="flex justify-between space-x-4">
              <span className="text-slate-400">Simulator Ground Truth:</span>
              <span className="text-slate-300">
                {hoveredItem.ground_truth_attack_active ? hoveredItem.ground_truth_attack_type.toUpperCase() : "CLEAN"}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Semantic Guidance Footer */}
      <div className="mt-2.5 pt-2 border-t border-slate-800/80 flex items-center justify-between text-[11px] text-slate-500">
        <div className="flex items-center space-x-1.5">
          <HelpCircle className="w-3.5 h-3.5 text-slate-400" />
          <span>Decision Semantics: Score $\ge$ Calibrated Threshold (1% FPR target) declares security incident.</span>
        </div>
        <div className="font-mono">
          Current Score: <span className="font-bold text-slate-200">{currentScore.toFixed(4)}</span>
        </div>
      </div>
    </div>
  );
};
