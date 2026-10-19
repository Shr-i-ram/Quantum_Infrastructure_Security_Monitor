import React from "react";
import {
  Activity,
  ShieldAlert,
  Database,
  Cpu,
  BarChart3,
  HeartPulse,
} from "lucide-react";

export type NavTab =
  | "live"
  | "incidents"
  | "telemetry"
  | "models"
  | "evaluation"
  | "health";

interface SidebarProps {
  currentTab: NavTab;
  onSelectTab: (tab: NavTab) => void;
  incidentCount: number;
  collapsed: boolean;
  onToggleCollapse: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentTab,
  onSelectTab,
  incidentCount,
  collapsed,
}) => {
  const navItems = [
    { id: "live", label: "Live Monitor", icon: Activity, badge: null },
    {
      id: "incidents",
      label: "Incidents",
      icon: ShieldAlert,
      badge: incidentCount > 0 ? incidentCount : null,
      badgeColor: "bg-red-500/20 text-red-400 border border-red-500/40",
    },
    { id: "telemetry", label: "Telemetry Explorer", icon: Database, badge: "76 feat" },
    { id: "models", label: "Model Lab", icon: Cpu, badge: "13 avail" },
    { id: "evaluation", label: "Evaluation Results", icon: BarChart3, badge: "Phase XI" },
    { id: "health", label: "System Health", icon: HeartPulse, badge: null },
  ];

  return (
    <aside
      className={`border-r border-slate-800 bg-slate-950 flex flex-col justify-between transition-all duration-300 ${
        collapsed ? "w-16" : "w-64"
      }`}
    >
      <div className="py-4">
        <div className="px-4 mb-3">
          {!collapsed && (
            <p className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
              Operations Navigation
            </p>
          )}
        </div>
        <nav className="space-y-1 px-2">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = currentTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onSelectTab(item.id as NavTab)}
                className={`w-full flex items-center px-3 py-2.5 rounded-lg text-sm font-medium transition-all ${
                  isActive
                    ? "bg-cyan-950/60 text-cyan-300 border border-cyan-500/40 shadow-sm shadow-cyan-950"
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-900"
                }`}
                title={collapsed ? item.label : undefined}
              >
                <Icon
                  className={`w-5 h-5 flex-shrink-0 ${
                    isActive ? "text-cyan-400" : "text-slate-500"
                  }`}
                />
                {!collapsed && (
                  <span className="ml-3 flex-1 text-left">{item.label}</span>
                )}
                {!collapsed && item.badge && (
                  <span
                    className={`ml-2 px-1.5 py-0.5 text-[10px] rounded font-semibold ${
                      item.badgeColor || "bg-slate-800 text-slate-400 border border-slate-700"
                    }`}
                  >
                    {item.badge}
                  </span>
                )}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Footer Info in Sidebar */}
      {!collapsed && (
        <div className="p-4 border-t border-slate-800/80 bg-slate-900/40 text-[11px] text-slate-500 space-y-1">
          <div className="flex justify-between">
            <span>Protocol:</span>
            <span className="text-slate-300 font-mono">BB84 Decoy-State</span>
          </div>
          <div className="flex justify-between">
            <span>Encoding:</span>
            <span className="text-slate-300 font-mono">Phase / Polarization</span>
          </div>
          <div className="flex justify-between">
            <span>QML Engine:</span>
            <span className="text-slate-300 font-mono">PennyLane lightning</span>
          </div>
        </div>
      )}
    </aside>
  );
};
