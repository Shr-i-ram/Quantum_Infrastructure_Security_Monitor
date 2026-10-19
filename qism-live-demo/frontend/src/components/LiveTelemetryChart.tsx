import React, { useState } from "react";
import { LineChart, Sliders } from "lucide-react";
import { TelemetryPacket } from "../types";

interface LiveTelemetryChartProps {
  history: TelemetryPacket[];
  currentQBER: number;
  currentSiftedRate: number;
  currentSignalGain: number;
  currentDecoyGain: number;
}

export const LiveTelemetryChart: React.FC<LiveTelemetryChartProps> = ({
  history,
  currentQBER,
  currentSiftedRate,
  currentSignalGain,
  currentDecoyGain,
}) => {
  const [activeSignal, setActiveSignal] = useState<"qber" | "gains" | "sifted">("qber");

  const width = 680;
  const height = 210;
  const padding = { top: 20, right: 30, bottom: 30, left: 55 };

  const plotW = width - padding.left - padding.right;
  const plotH = height - padding.top - padding.bottom;
  const pointsCount = Math.max(history.length, 2);

  const getX = (idx: number) => padding.left + (idx / (pointsCount - 1)) * plotW;

  // Compute scale based on active signal
  let maxVal = 0.3; // Default 30% for QBER
  let unit = "%";
  if (activeSignal === "qber") {
    maxVal = Math.max(0.15, ...history.map((h) => h.qber * 1.2));
    unit = "%";
  } else if (activeSignal === "sifted") {
    maxVal = Math.max(0.01, ...history.map((h) => h.sifted_rate * 1.3));
    unit = " /pulse";
  } else {
    maxVal = Math.max(0.015, ...history.map((h) => Math.max(h.signal_gain, h.decoy_gain) * 1.3));
    unit = "";
  }

  const getY = (val: number) => padding.top + plotH - (Math.min(val, maxVal) / maxVal) * plotH;

  // Signal paths
  const qberPath = history
    .map((h, i) => `${i === 0 ? "M" : "L"} ${getX(i).toFixed(1)} ${getY(h.qber).toFixed(1)}`)
    .join(" ");

  const siftedPath = history
    .map((h, i) => `${i === 0 ? "M" : "L"} ${getX(i).toFixed(1)} ${getY(h.sifted_rate).toFixed(1)}`)
    .join(" ");

  const sigGainPath = history
    .map((h, i) => `${i === 0 ? "M" : "L"} ${getX(i).toFixed(1)} ${getY(h.signal_gain).toFixed(1)}`)
    .join(" ");

  const decGainPath = history
    .map((h, i) => `${i === 0 ? "M" : "L"} ${getX(i).toFixed(1)} ${getY(h.decoy_gain).toFixed(1)}`)
    .join(" ");

  // QBER 11% Abort limit
  const abortLimitY = activeSignal === "qber" ? getY(0.11) : null;

  return (
    <div className="qism-panel p-4 flex flex-col justify-between">
      {/* Header with signal selectors */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center space-x-2">
          <LineChart className="w-4 h-4 text-cyan-400" />
          <h3 className="text-sm font-semibold text-slate-100">
            Physical QKD Channel & Detector Telemetry
          </h3>
        </div>

        {/* Signal Tab Selector */}
        <div className="flex items-center space-x-1 bg-slate-900 p-0.5 rounded border border-slate-800 text-xs">
          <button
            onClick={() => setActiveSignal("qber")}
            className={`px-2.5 py-1 rounded font-medium transition-colors ${
              activeSignal === "qber"
                ? "bg-cyan-950 text-cyan-300 border border-cyan-500/40"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            QBER (Error Rate)
          </button>
          <button
            onClick={() => setActiveSignal("gains")}
            className={`px-2.5 py-1 rounded font-medium transition-colors ${
              activeSignal === "gains"
                ? "bg-cyan-950 text-cyan-300 border border-cyan-500/40"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            Decoy-State Gains (Q_μ, Q_ν)
          </button>
          <button
            onClick={() => setActiveSignal("sifted")}
            className={`px-2.5 py-1 rounded font-medium transition-colors ${
              activeSignal === "sifted"
                ? "bg-cyan-950 text-cyan-300 border border-cyan-500/40"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            Sifted Key Rate
          </button>
        </div>
      </div>

      {/* SVG Canvas */}
      <div className="relative w-full overflow-hidden bg-slate-950/60 rounded border border-slate-800/80">
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-40 select-none">
          {/* Grid lines */}
          {[0, 0.25, 0.5, 0.75, 1.0].map((frac, idx) => {
            const y = padding.top + plotH * (1 - frac);
            const val = frac * maxVal;
            const displayVal = activeSignal === "qber" ? `${(val * 100).toFixed(1)}%` : val.toFixed(4);
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
                  {displayVal}
                </text>
              </g>
            );
          })}

          {/* QBER 11% Theoretical BB84 Security Threshold Line */}
          {activeSignal === "qber" && abortLimitY !== null && abortLimitY >= padding.top && (
            <g>
              <line
                x1={padding.left}
                y1={abortLimitY}
                x2={width - padding.right}
                y2={abortLimitY}
                stroke="#f59e0b"
                strokeWidth="1"
                strokeDasharray="3,3"
              />
              <text
                x={width - padding.right - 5}
                y={abortLimitY - 4}
                textAnchor="end"
                className="fill-amber-400 text-[9px] font-mono"
              >
                11.0% BB84 Abort Threshold
              </text>
            </g>
          )}

          {/* Signal traces */}
          {activeSignal === "qber" && qberPath && (
            <path
              d={qberPath}
              fill="none"
              stroke="#06b6d4"
              strokeWidth="2"
              strokeLinecap="round"
            />
          )}

          {activeSignal === "sifted" && siftedPath && (
            <path
              d={siftedPath}
              fill="none"
              stroke="#10b981"
              strokeWidth="2"
              strokeLinecap="round"
            />
          )}

          {activeSignal === "gains" && (
            <>
              {sigGainPath && (
                <path
                  d={sigGainPath}
                  fill="none"
                  stroke="#38bdf8"
                  strokeWidth="2"
                  strokeLinecap="round"
                />
              )}
              {decGainPath && (
                <path
                  d={decGainPath}
                  fill="none"
                  stroke="#a855f7"
                  strokeWidth="2"
                  strokeLinecap="round"
                />
              )}
            </>
          )}

          {/* Data point markers */}
          {history.map((h, i) => {
            const cx = getX(i);
            const val =
              activeSignal === "qber"
                ? h.qber
                : activeSignal === "sifted"
                ? h.sifted_rate
                : h.signal_gain;
            const cy = getY(val);
            return (
              <circle
                key={i}
                cx={cx}
                cy={cy}
                r={2}
                fill="#06b6d4"
                className="hover:r-4 transition-all"
              />
            );
          })}

          {/* X Axis baseline */}
          <line
            x1={padding.left}
            y1={padding.top + plotH}
            x2={width - padding.right}
            y2={padding.top + plotH}
            stroke="#334155"
          />
        </svg>
      </div>

      {/* Live Readout Values strip */}
      <div className="mt-2.5 pt-2 border-t border-slate-800/80 grid grid-cols-4 gap-2 text-xs">
        <div>
          <span className="text-slate-500 block text-[10px]">Quantum Bit Error Rate:</span>
          <span className={`font-mono font-bold ${currentQBER > 0.05 ? "text-amber-400" : "text-cyan-300"}`}>
            {(currentQBER * 100).toFixed(2)}%
          </span>
        </div>
        <div>
          <span className="text-slate-500 block text-[10px]">Sifted Key Rate:</span>
          <span className="font-mono font-bold text-emerald-400">
            {(currentSiftedRate * 1000).toFixed(2)} kEv/s
          </span>
        </div>
        <div>
          <span className="text-slate-500 block text-[10px]">Signal Gain (Q_μ):</span>
          <span className="font-mono font-bold text-sky-400">
            {currentSignalGain.toFixed(4)}
          </span>
        </div>
        <div>
          <span className="text-slate-500 block text-[10px]">Decoy Gain (Q_ν):</span>
          <span className="font-mono font-bold text-purple-400">
            {currentDecoyGain.toFixed(4)}
          </span>
        </div>
      </div>
    </div>
  );
};
