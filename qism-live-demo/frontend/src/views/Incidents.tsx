import React, { useState } from "react";
import {
  ShieldAlert,
  Search,
  Filter,
  CheckCircle,
  Eye,
  Clock,
  Cpu,
  Layers,
  FileCode,
  AlertOctagon,
  ArrowRight,
} from "lucide-react";
import { updateIncidentStatus } from "../services/api";
import { Incident } from "../types";

interface IncidentsProps {
  incidents: Incident[];
  onRefresh: () => void;
}

export const Incidents: React.FC<IncidentsProps> = ({ incidents, onRefresh }) => {
  const [selectedIncident, setSelectedIncident] = useState<Incident | null>(null);
  const [statusFilter, setStatusFilter] = useState<string>("ALL");
  const [severityFilter, setSeverityFilter] = useState<string>("ALL");
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [updating, setUpdating] = useState<boolean>(false);

  const filtered = incidents.filter((inc) => {
    if (statusFilter !== "ALL" && inc.status !== statusFilter) return false;
    if (severityFilter !== "ALL" && inc.severity !== severityFilter) return false;
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      return (
        inc.incident_id.toLowerCase().includes(q) ||
        inc.model_id.toLowerCase().includes(q) ||
        inc.ground_truth_attack_type.toLowerCase().includes(q)
      );
    }
    return true;
  });

  const handleStatusUpdate = async (incidentId: string, newStatus: string) => {
    setUpdating(true);
    try {
      await updateIncidentStatus(incidentId, newStatus);
      if (selectedIncident && selectedIncident.incident_id === incidentId) {
        setSelectedIncident({ ...selectedIncident, status: newStatus as any });
      }
      onRefresh();
    } catch (e) {
      console.error(e);
    } finally {
      setUpdating(false);
    }
  };

  const getSeverityBadge = (severity: string) => {
    switch (severity) {
      case "CRITICAL":
        return "bg-red-950/80 text-red-300 border-red-500/60";
      case "HIGH":
        return "bg-orange-950/80 text-orange-300 border-orange-500/60";
      case "MEDIUM":
        return "bg-amber-950/80 text-amber-300 border-amber-500/60";
      default:
        return "bg-slate-800 text-slate-300 border-slate-700";
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "NEW":
        return "bg-cyan-950/80 text-cyan-300 border-cyan-500/50";
      case "ACKNOWLEDGED":
        return "bg-indigo-950/80 text-indigo-300 border-indigo-500/50";
      case "INVESTIGATING":
        return "bg-amber-950/80 text-amber-300 border-amber-500/50";
      case "RESOLVED":
        return "bg-emerald-950/80 text-emerald-300 border-emerald-500/50";
      default:
        return "bg-slate-800 text-slate-400";
    }
  };

  return (
    <div className="space-y-4">
      {/* Header & Filter Bar */}
      <div className="qism-panel p-4 flex flex-col md:flex-row md:items-center justify-between gap-3">
        <div>
          <h2 className="text-base font-bold text-slate-100 flex items-center space-x-2">
            <ShieldAlert className="w-5 h-5 text-red-400" />
            <span>Recorded Security Incident Registry</span>
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Persistent security alerts logged when model anomaly scores exceed calibrated thresholds. Backed by SQLite.
          </p>
        </div>

        {/* Filters */}
        <div className="flex flex-wrap items-center gap-2">
          {/* Search box */}
          <div className="relative">
            <Search className="w-3.5 h-3.5 text-slate-400 absolute left-2.5 top-2.5" />
            <input
              type="text"
              placeholder="Search incident, model, attack..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="bg-slate-950 border border-slate-700 rounded pl-8 pr-3 py-1.5 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500 w-52"
            />
          </div>

          {/* Severity filter */}
          <select
            value={severityFilter}
            onChange={(e) => setSeverityFilter(e.target.value)}
            className="bg-slate-950 border border-slate-700 rounded px-2.5 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-cyan-500"
          >
            <option value="ALL">All Severities</option>
            <option value="CRITICAL">Critical</option>
            <option value="HIGH">High</option>
            <option value="MEDIUM">Medium</option>
            <option value="LOW">Low</option>
          </select>

          {/* Status filter */}
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="bg-slate-950 border border-slate-700 rounded px-2.5 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-cyan-500"
          >
            <option value="ALL">All Statuses</option>
            <option value="NEW">New</option>
            <option value="ACKNOWLEDGED">Acknowledged</option>
            <option value="INVESTIGATING">Investigating</option>
            <option value="RESOLVED">Resolved</option>
          </select>

          <button
            onClick={onRefresh}
            className="px-2.5 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-xs text-slate-300 font-medium"
          >
            Refresh
          </button>
        </div>
      </div>

      {/* Main Table + Investigation Panel Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        {/* Incident Table */}
        <div className={selectedIncident ? "lg:col-span-7" : "lg:col-span-12"}>
          <div className="qism-panel overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-900/80 border-b border-slate-800 text-slate-400 font-semibold uppercase tracking-wider">
                  <tr>
                    <th className="py-2.5 px-3">Incident ID</th>
                    <th className="py-2.5 px-3">Timestamp</th>
                    <th className="py-2.5 px-3">Model</th>
                    <th className="py-2.5 px-3">Severity</th>
                    <th className="py-2.5 px-3">Score / Threshold</th>
                    <th className="py-2.5 px-3">Ground Truth</th>
                    <th className="py-2.5 px-3">Status</th>
                    <th className="py-2.5 px-3 text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono">
                  {filtered.length === 0 ? (
                    <tr>
                      <td colSpan={8} className="py-8 text-center text-slate-500 font-sans text-xs">
                        No incidents matching current filters. Run simulation and inject an attack to trigger alerts.
                      </td>
                    </tr>
                  ) : (
                    filtered.map((inc) => (
                      <tr
                        key={inc.incident_id}
                        onClick={() => setSelectedIncident(inc)}
                        className={`hover:bg-slate-900/60 cursor-pointer transition-colors ${
                          selectedIncident?.incident_id === inc.incident_id ? "bg-slate-900/90 border-l-2 border-cyan-500" : ""
                        }`}
                      >
                        <td className="py-2.5 px-3 text-cyan-300 font-semibold">{inc.incident_id}</td>
                        <td className="py-2.5 px-3 text-slate-400">
                          {new Date(inc.detection_timestamp * 1000).toLocaleTimeString()}
                        </td>
                        <td className="py-2.5 px-3 text-slate-300 font-sans text-[11px] truncate max-w-[120px]">
                          {inc.model_id}
                        </td>
                        <td className="py-2.5 px-3 font-sans">
                          <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${getSeverityBadge(inc.severity)}`}>
                            {inc.severity}
                          </span>
                        </td>
                        <td className="py-2.5 px-3">
                          <span className="text-red-400 font-bold">{inc.anomaly_score.toFixed(4)}</span>
                          <span className="text-slate-500 text-[10px]"> / {inc.threshold.toFixed(4)}</span>
                        </td>
                        <td className="py-2.5 px-3 font-sans text-[11px]">
                          <span className={inc.ground_truth_attack_active ? "text-red-400 font-semibold" : "text-emerald-400"}>
                            {inc.ground_truth_attack_active ? inc.ground_truth_attack_type.toUpperCase() : "CLEAN"}
                          </span>
                        </td>
                        <td className="py-2.5 px-3 font-sans">
                          <span className={`px-2 py-0.5 rounded text-[10px] font-semibold border ${getStatusBadge(inc.status)}`}>
                            {inc.status}
                          </span>
                        </td>
                        <td className="py-2.5 px-3 text-right">
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              setSelectedIncident(inc);
                            }}
                            className="p-1 rounded hover:bg-slate-800 text-slate-400 hover:text-cyan-300"
                            title="Inspect Evidence"
                          >
                            <Eye className="w-3.5 h-3.5" />
                          </button>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Evidence Investigation Drawer / Panel */}
        {selectedIncident && (
          <div className="lg:col-span-5 space-y-3">
            <div className="qism-panel p-4">
              <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                <div>
                  <h3 className="text-sm font-bold text-slate-100 flex items-center space-x-2">
                    <AlertOctagon className="w-4 h-4 text-red-400" />
                    <span>Evidence Dossier: {selectedIncident.incident_id}</span>
                  </h3>
                  <p className="text-[11px] text-slate-400 font-mono mt-0.5">
                    Session: {selectedIncident.session_id}
                  </p>
                </div>
                <button
                  onClick={() => setSelectedIncident(null)}
                  className="text-slate-400 hover:text-white text-xs px-2 py-1 rounded bg-slate-900 border border-slate-800"
                >
                  Close
                </button>
              </div>

              {/* Status Workflow Actions */}
              <div className="py-3 border-b border-slate-800 flex items-center justify-between">
                <span className="text-xs text-slate-400">Workflow Action:</span>
                <div className="flex items-center space-x-1.5">
                  {(["ACKNOWLEDGED", "INVESTIGATING", "RESOLVED"] as const).map((s) => (
                    <button
                      key={s}
                      disabled={updating || selectedIncident.status === s}
                      onClick={() => handleStatusUpdate(selectedIncident.incident_id, s)}
                      className={`px-2 py-1 rounded text-[10px] font-semibold transition-colors ${
                        selectedIncident.status === s
                          ? "bg-cyan-900/60 text-cyan-200 border border-cyan-500/50"
                          : "bg-slate-900 border border-slate-700 text-slate-300 hover:bg-slate-800"
                      }`}
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </div>

              {/* Technical Inspection Cards */}
              <div className="py-3 space-y-3 text-xs">
                <div className="grid grid-cols-2 gap-2 font-mono">
                  <div className="bg-slate-900/60 p-2.5 rounded border border-slate-800">
                    <span className="text-slate-500 block text-[10px]">Anomaly Score:</span>
                    <span className="text-base font-bold text-red-400">
                      {selectedIncident.anomaly_score.toFixed(4)}
                    </span>
                    <span className="text-[10px] text-slate-400 block mt-0.5">
                      Threshold τ: {selectedIncident.threshold.toFixed(4)}
                    </span>
                  </div>

                  <div className="bg-slate-900/60 p-2.5 rounded border border-slate-800">
                    <span className="text-slate-500 block text-[10px]">Detection Latency:</span>
                    <span className="text-base font-bold text-cyan-300">
                      {selectedIncident.detection_latency_windows} wins
                    </span>
                    <span className="text-[10px] text-slate-400 block mt-0.5">
                      Since attack injection
                    </span>
                  </div>
                </div>

                {/* Ground truth vs Model claim */}
                <div className="bg-slate-900/70 p-3 rounded border border-slate-800 space-y-1.5">
                  <div className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">
                    Scientific Evaluation Separation
                  </div>
                  <div className="flex justify-between font-mono">
                    <span className="text-slate-400">Simulator Ground Truth:</span>
                    <span className="font-bold text-red-400">
                      {selectedIncident.ground_truth_attack_active
                        ? selectedIncident.ground_truth_attack_type.toUpperCase()
                        : "CLEAN LINK"}
                    </span>
                  </div>
                  <div className="flex justify-between font-mono">
                    <span className="text-slate-400">Detector Model:</span>
                    <span className="text-slate-200">{selectedIncident.model_id}</span>
                  </div>
                  <p className="text-[10px] text-slate-500 leading-tight pt-1">
                    Note: Unsupervised detector scores deviation from clean state space without predicting specific attack classes.
                  </p>
                </div>

                {/* Key Evidence Features Snapshot */}
                <div>
                  <div className="text-[11px] font-semibold text-slate-300 mb-1.5 flex items-center space-x-1.5">
                    <FileCode className="w-3.5 h-3.5 text-cyan-400" />
                    <span>Telemetry Evidence Snapshot</span>
                  </div>
                  <div className="bg-slate-950 p-2.5 rounded border border-slate-800 font-mono text-[11px] max-h-56 overflow-y-auto">
                    <pre className="text-slate-300 whitespace-pre-wrap">
                      {JSON.stringify(selectedIncident.evidence_snapshot, null, 2)}
                    </pre>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
