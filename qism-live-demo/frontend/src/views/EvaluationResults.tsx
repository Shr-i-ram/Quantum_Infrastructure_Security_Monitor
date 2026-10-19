import React, { useState } from "react";
import { BarChart3, Image, Award, ExternalLink, X, Info } from "lucide-react";
import { getFigureImageUrl } from "../services/api";
import { BenchmarkRecord, PublicationFigure } from "../types";

interface EvaluationResultsProps {
  leaderboard: BenchmarkRecord[];
  figures: PublicationFigure[];
}

export const EvaluationResults: React.FC<EvaluationResultsProps> = ({
  leaderboard,
  figures,
}) => {
  const [selectedFigure, setSelectedFigure] = useState<PublicationFigure | null>(null);
  const [activeTab, setActiveTab] = useState<"leaderboard" | "figures">("leaderboard");

  return (
    <div className="space-y-4">
      {/* Header bar */}
      <div className="qism-panel p-4 flex flex-col md:flex-row md:items-center justify-between gap-3">
        <div>
          <h2 className="text-base font-bold text-slate-100 flex items-center space-x-2">
            <Award className="w-5 h-5 text-amber-400" />
            <span>Authoritative Evaluation Results & Publication Figures</span>
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Phase XI benchmark leaderboard and high-resolution publication assets. Evaluated across 24 classical and quantum models.
          </p>
        </div>

        <div className="flex items-center space-x-1 bg-slate-900 p-0.5 rounded border border-slate-800 text-xs">
          <button
            onClick={() => setActiveTab("leaderboard")}
            className={`px-3 py-1.5 rounded font-medium transition-colors ${
              activeTab === "leaderboard"
                ? "bg-cyan-950 text-cyan-300 border border-cyan-500/40"
                : "text-slate-400 hover:text-white"
            }`}
          >
            Benchmark Leaderboard
          </button>
          <button
            onClick={() => setActiveTab("figures")}
            className={`px-3 py-1.5 rounded font-medium transition-colors ${
              activeTab === "figures"
                ? "bg-cyan-950 text-cyan-300 border border-cyan-500/40"
                : "text-slate-400 hover:text-white"
            }`}
          >
            Publication Figures ({figures.length})
          </button>
        </div>
      </div>

      {/* Tab 1: Leaderboard Table */}
      {activeTab === "leaderboard" && (
        <div className="qism-panel overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-900/90 border-b border-slate-800 text-slate-400 font-semibold uppercase tracking-wider">
                <tr>
                  <th className="py-2.5 px-3">Model Name</th>
                  <th className="py-2.5 px-3">Family / Paradigm</th>
                  <th className="py-2.5 px-3">Qubits</th>
                  <th className="py-2.5 px-3">Encoding</th>
                  <th className="py-2.5 px-3 text-right">Test AUROC</th>
                  <th className="py-2.5 px-3 text-right">Test AUPRC</th>
                  <th className="py-2.5 px-3 text-right">Recall @ 1% FPR</th>
                  <th className="py-2.5 px-3 text-right">Latency (ms)</th>
                  <th className="py-2.5 px-3 text-right">Training (s)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono">
                {leaderboard.map((row) => {
                  const isFlagship = row.model_id === "PITQSVDD_6Q_Angle";
                  return (
                    <tr
                      key={row.model_id}
                      className={`hover:bg-slate-900/60 transition-colors ${
                        isFlagship ? "bg-cyan-950/40 border-l-2 border-cyan-500 font-bold" : ""
                      }`}
                    >
                      <td className="py-2.5 px-3 font-sans">
                        <div className="flex items-center space-x-2">
                          <span className={isFlagship ? "text-cyan-300 font-bold" : "text-slate-200"}>
                            {row.model_name || row.model_id}
                          </span>
                          {isFlagship && (
                            <span className="px-1.5 py-0.2 rounded bg-cyan-950 border border-cyan-500/60 text-cyan-300 text-[9px]">
                              FLAGSHIP
                            </span>
                          )}
                        </div>
                      </td>
                      <td className="py-2.5 px-3 font-sans text-slate-400 text-[11px] capitalize">
                        {(row.paradigm || "unsupervised").replace(/_/g, " ")}
                      </td>
                      <td className="py-2.5 px-3 text-slate-300">
                        {row.qubits ? `${row.qubits}Q` : "—"}
                      </td>
                      <td className="py-2.5 px-3 text-slate-400 capitalize">
                        {row.encoding || "—"}
                      </td>
                      <td className="py-2.5 px-3 text-right text-emerald-400 font-bold">
                        {row.test_auroc ? (row.test_auroc * 100).toFixed(2) + "%" : "—"}
                      </td>
                      <td className="py-2.5 px-3 text-right text-cyan-300">
                        {row.test_auprc ? (row.test_auprc * 100).toFixed(2) + "%" : "—"}
                      </td>
                      <td className="py-2.5 px-3 text-right text-amber-300 font-bold">
                        {row.recall_at_1pct_fpr ? (row.recall_at_1pct_fpr * 100).toFixed(2) + "%" : "—"}
                      </td>
                      <td className="py-2.5 px-3 text-right text-slate-300">
                        {row.mean_latency_ms ? row.mean_latency_ms.toFixed(1) : "—"}
                      </td>
                      <td className="py-2.5 px-3 text-right text-slate-400">
                        {row.training_time_s ? row.training_time_s.toFixed(1) + "s" : "—"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Tab 2: Publication Figure Gallery */}
      {activeTab === "figures" && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {figures.map((fig) => (
            <div
              key={fig.id}
              onClick={() => setSelectedFigure(fig)}
              className="qism-panel p-3 cursor-pointer hover:border-cyan-500/70 transition-all group flex flex-col justify-between"
            >
              <div>
                <div className="aspect-[16/10] bg-slate-950 rounded overflow-hidden border border-slate-800 mb-2 relative">
                  <img
                    src={getFigureImageUrl(fig.file)}
                    alt={fig.title}
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                    loading="lazy"
                  />
                  <span className="absolute top-2 left-2 px-2 py-0.5 rounded bg-slate-950/80 backdrop-blur border border-slate-800 text-[10px] text-cyan-300 font-semibold">
                    {fig.category}
                  </span>
                </div>
                <h4 className="text-xs font-bold text-slate-200 line-clamp-1 mb-1">
                  {fig.title}
                </h4>
                <p className="text-[11px] text-slate-400 line-clamp-2 leading-tight">
                  {fig.caption}
                </p>
              </div>

              <div className="pt-2 mt-2 border-t border-slate-800/80 flex items-center justify-between text-[10px] text-slate-500">
                <span>Click to expand high-res</span>
                <ExternalLink className="w-3 h-3 text-cyan-400" />
              </div>
            </div>
          ))}
        </div>
      )}

      {/* High-Resolution Figure Modal */}
      {selectedFigure && (
        <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-md flex items-center justify-center p-4">
          <div className="qism-panel max-w-4xl w-full max-h-[90vh] overflow-y-auto p-5 border border-cyan-500/40 shadow-2xl space-y-4">
            <div className="flex items-center justify-between pb-2 border-b border-slate-800">
              <div>
                <span className="text-[10px] px-2 py-0.5 rounded bg-cyan-950 border border-cyan-500/50 text-cyan-300 font-bold uppercase">
                  {selectedFigure.category}
                </span>
                <h3 className="text-sm font-bold text-slate-100 mt-1">
                  {selectedFigure.title}
                </h3>
              </div>
              <button
                onClick={() => setSelectedFigure(null)}
                className="p-1 rounded bg-slate-900 border border-slate-800 text-slate-400 hover:text-white"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="bg-slate-950 rounded p-2 border border-slate-800 flex items-center justify-center">
              <img
                src={getFigureImageUrl(selectedFigure.file)}
                alt={selectedFigure.title}
                className="max-h-[60vh] object-contain rounded"
              />
            </div>

            <div className="bg-slate-900/60 p-3 rounded border border-slate-800 text-xs text-slate-300 leading-relaxed">
              <span className="font-bold text-cyan-400 block mb-0.5">Scientific Context:</span>
              {selectedFigure.caption}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
