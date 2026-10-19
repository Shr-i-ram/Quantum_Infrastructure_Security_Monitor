import React from "react";
import { Activity, ShieldCheck, ShieldAlert, Cpu, Radio, Maximize2, Minimize2 } from "lucide-react";
import { SystemHealth } from "../types";

interface HeaderProps {
  health: SystemHealth | null;
  activeModelName: string;
  mode: "live_simulation" | "recorded_replay" | "none";
  activeAttackType: string | null;
  presentationMode: boolean;
  onTogglePresentation: () => void;
  isConnected: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  health,
  activeModelName,
  mode,
  activeAttackType,
  presentationMode,
  onTogglePresentation,
  isConnected,
}) => {
  return (
    <header className="h-16 border-b border-slate-800 bg-slate-950/80 px-6 flex items-center justify-between backdrop-blur sticky top-0 z-40">
      {/* Brand & Identity */}
      <div className="flex items-center space-x-4">
        <div className="w-9 h-9 rounded-lg bg-cyan-950 border border-cyan-500/40 flex items-center justify-center text-cyan-400">
          <Activity className="w-5 h-5" />
        </div>
        <div>
          <div className="flex items-center space-x-2">
            <h1 className="text-base font-bold tracking-wider text-slate-100 uppercase">
              QISM
            </h1>
            <span className="text-xs px-2 py-0.5 rounded-full font-medium bg-slate-800 text-slate-300 border border-slate-700">
              v1.0 Demo Console
            </span>
          </div>
          <p className="text-xs text-slate-400 hidden sm:block">
            Quantum Infrastructure Security Monitor · Anomaly-Based ML QKD Attack Prevention
          </p>
        </div>
      </div>

      {/* Central Environment Badge */}
      <div className="flex items-center space-x-3">
        {mode === "live_simulation" && (
          <div className="flex items-center space-x-2 px-3 py-1 rounded-md bg-cyan-950/60 border border-cyan-500/50 text-cyan-300 text-xs font-semibold tracking-wide">
            <span className="w-2 h-2 rounded-full bg-cyan-400 indicator-pulse" />
            <span>SIMULATION ENVIRONMENT</span>
          </div>
        )}

        {mode === "recorded_replay" && (
          <div className="flex items-center space-x-2 px-3 py-1 rounded-md bg-amber-950/60 border border-amber-500/50 text-amber-300 text-xs font-semibold tracking-wide">
            <span className="w-2 h-2 rounded-full bg-amber-400 indicator-pulse" />
            <span>RECORDED SCENARIO REPLAY</span>
          </div>
        )}

        {mode === "none" && (
          <div className="flex items-center space-x-2 px-3 py-1 rounded-md bg-slate-900 border border-slate-700 text-slate-400 text-xs">
            <span className="w-2 h-2 rounded-full bg-slate-500" />
            <span>SYSTEM STANDBY</span>
          </div>
        )}

        {activeAttackType && activeAttackType !== "none" && (
          <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded-md bg-red-950/80 border border-red-500/60 text-red-300 text-xs font-bold indicator-alert">
            <ShieldAlert className="w-3.5 h-3.5" />
            <span>ATTACK ACTIVE: {activeAttackType.toUpperCase().replace("_", " ")}</span>
          </div>
        )}
      </div>

      {/* Status Indicators & Presentation Controls */}
      <div className="flex items-center space-x-4">
        {/* Active Model Indicator */}
        <div className="hidden lg:flex items-center space-x-2 px-3 py-1 rounded bg-slate-900 border border-slate-800 text-xs">
          <Cpu className="w-3.5 h-3.5 text-cyan-400" />
          <span className="text-slate-400">Detector:</span>
          <span className="font-semibold text-cyan-300">{activeModelName}</span>
        </div>

        {/* Backend Connection State */}
        <div className="flex items-center space-x-2 text-xs">
          <span
            className={`w-2 h-2 rounded-full ${
              isConnected ? "bg-emerald-400" : "bg-red-500"
            }`}
          />
          <span className="text-slate-400 hidden sm:inline">
            {isConnected ? "Live Stream 2Hz" : "Disconnected"}
          </span>
        </div>

        {/* Presentation Mode Toggle */}
        <button
          onClick={onTogglePresentation}
          className="p-1.5 rounded bg-slate-900 border border-slate-700 text-slate-300 hover:text-cyan-300 hover:border-cyan-500 transition-colors"
          title={presentationMode ? "Exit Presentation Mode" : "Enter Presentation Mode"}
        >
          {presentationMode ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
        </button>
      </div>
    </header>
  );
};
