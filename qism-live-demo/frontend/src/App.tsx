import React, { useEffect, useRef, useState } from "react";
import { Header } from "./components/Header";
import { NavTab, Sidebar } from "./components/Sidebar";
import {
  WS_URL,
  fetchBenchmarkLeaderboard,
  fetchFigures,
  fetchHealth,
  fetchIncidents,
  fetchModels,
  fetchRecordedReplays,
  fetchSupportedAttacks,
  fetchTelemetrySchema,
} from "./services/api";
import {
  AttackConfig,
  BenchmarkRecord,
  Incident,
  ModelInfo,
  PublicationFigure,
  ReplayScenario,
  SystemHealth,
  TelemetryPacket,
} from "./types";
import { EvaluationResults } from "./views/EvaluationResults";
import { Incidents } from "./views/Incidents";
import { LiveMonitor } from "./views/LiveMonitor";
import { ModelLab } from "./views/ModelLab";
import { SystemHealthView } from "./views/SystemHealth";
import { TelemetryExplorer } from "./views/TelemetryExplorer";

export const App: React.FC = () => {
  const [currentTab, setCurrentTab] = useState<NavTab>("live");
  const [presentationMode, setPresentationMode] = useState<boolean>(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState<boolean>(false);

  // Real-time telemetry state
  const [currentPacket, setCurrentPacket] = useState<TelemetryPacket | null>(null);
  const [history, setHistory] = useState<TelemetryPacket[]>([]);
  const [activeIncidents, setActiveIncidents] = useState<Incident[]>([]);
  const [isConnected, setIsConnected] = useState<boolean>(false);

  // Static & catalog state
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [activeModelId, setActiveModelId] = useState<string>("PITQSVDD_6Q_Angle");
  const [supportedAttacks, setSupportedAttacks] = useState<AttackConfig[]>([]);
  const [recordedReplays, setRecordedReplays] = useState<ReplayScenario[]>([]);
  const [schema, setSchema] = useState<any>(null);
  const [leaderboard, setLeaderboard] = useState<BenchmarkRecord[]>([]);
  const [figures, setFigures] = useState<PublicationFigure[]>([]);
  const [health, setHealth] = useState<SystemHealth | null>(null);

  const wsRef = useRef<WebSocket | null>(null);

  // Initial data loading
  const loadInitialData = async () => {
    try {
      const [hData, mData, aData, rData, iData, sData, lData, fData] = await Promise.all([
        fetchHealth().catch(() => null),
        fetchModels().catch(() => ({ models: [], active_model_id: "PITQSVDD_6Q_Angle" })),
        fetchSupportedAttacks().catch(() => ({ attacks: [] })),
        fetchRecordedReplays().catch(() => ({ replays: [] })),
        fetchIncidents().catch(() => ({ incidents: [] })),
        fetchTelemetrySchema().catch(() => null),
        fetchBenchmarkLeaderboard().catch(() => ({ leaderboard: [] })),
        fetchFigures().catch(() => ({ figures: [] })),
      ]);

      if (hData) setHealth(hData);
      if (mData.models.length > 0) {
        setModels(mData.models);
        setActiveModelId(mData.active_model_id || "PITQSVDD_6Q_Angle");
      }
      setSupportedAttacks(aData.attacks);
      setRecordedReplays(rData.replays);
      setActiveIncidents(iData.incidents);
      setSchema(sData);
      setLeaderboard(lData.leaderboard);
      setFigures(fData.figures);
    } catch (e) {
      console.error("Error loading initial QISM console data:", e);
    }
  };

  useEffect(() => {
    loadInitialData();
  }, []);

  // Real-time WebSocket connection
  useEffect(() => {
    let reconnectTimeout: any;

    const connectWebSocket = () => {
      try {
        const ws = new WebSocket(WS_URL);
        wsRef.current = ws;

        ws.onopen = () => {
          setIsConnected(true);
          console.log("Connected to QISM Telemetry Stream WebSocket");
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            if (data.type === "telemetry_update") {
              const packet = data as TelemetryPacket;
              setCurrentPacket(packet);

              setHistory((prev) => {
                const next = [...prev, packet];
                return next.length > 50 ? next.slice(-50) : next;
              });

              if (packet.incident) {
                setActiveIncidents((prev) => {
                  const exists = prev.some((i) => i.incident_id === packet.incident?.incident_id);
                  return exists ? prev : [packet.incident!, ...prev];
                });
              }
            }
          } catch (err) {
            console.error("Failed to parse WebSocket packet:", err);
          }
        };

        ws.onclose = () => {
          setIsConnected(false);
          reconnectTimeout = setTimeout(connectWebSocket, 2000);
        };

        ws.onerror = () => {
          setIsConnected(false);
          ws.close();
        };
      } catch (e) {
        console.error("WebSocket setup error:", e);
        reconnectTimeout = setTimeout(connectWebSocket, 3000);
      }
    };

    connectWebSocket();

    return () => {
      clearTimeout(reconnectTimeout);
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  const activeModelObj = models.find((m) => m.id === activeModelId);
  const activeModelName = activeModelObj ? activeModelObj.name : "PI-TQSVDD (6 Qubits)";

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans select-none">
      {/* Top Application Header */}
      <Header
        health={health}
        activeModelName={activeModelName}
        mode={currentPacket?.mode || "none"}
        activeAttackType={currentPacket?.ground_truth_attack_type || null}
        presentationMode={presentationMode}
        onTogglePresentation={() => setPresentationMode(!presentationMode)}
        isConnected={isConnected}
      />

      <div className="flex-1 flex overflow-hidden">
        {/* Navigation Sidebar */}
        {!presentationMode && (
          <Sidebar
            currentTab={currentTab}
            onSelectTab={setCurrentTab}
            incidentCount={activeIncidents.length}
            collapsed={sidebarCollapsed}
            onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
          />
        )}

        {/* Main Content Area */}
        <main className="flex-1 overflow-y-auto p-4 md:p-6 bg-[#070d18]">
          <div className="max-w-7xl mx-auto">
            {currentTab === "live" && (
              <LiveMonitor
                currentPacket={currentPacket}
                history={history}
                activeIncidents={activeIncidents}
                supportedAttacks={supportedAttacks}
                recordedReplays={recordedReplays}
                activeModelName={activeModelName}
                activeModelId={activeModelId}
                onNavigateToIncidents={() => setCurrentTab("incidents")}
              />
            )}

            {currentTab === "incidents" && (
              <Incidents
                incidents={activeIncidents}
                onRefresh={async () => {
                  const res = await fetchIncidents();
                  setActiveIncidents(res.incidents);
                }}
              />
            )}

            {currentTab === "telemetry" && (
              <TelemetryExplorer
                currentPacket={currentPacket}
                history={history}
                schema={schema}
              />
            )}

            {currentTab === "models" && (
              <ModelLab
                models={models}
                activeModelId={activeModelId}
                currentPacket={currentPacket}
                onModelSelected={(mId) => {
                  setActiveModelId(mId);
                  loadInitialData();
                }}
              />
            )}

            {currentTab === "evaluation" && (
              <EvaluationResults
                leaderboard={leaderboard}
                figures={figures}
              />
            )}

            {currentTab === "health" && (
              <SystemHealthView
                health={health}
                onRefresh={async () => {
                  const h = await fetchHealth();
                  setHealth(h);
                }}
              />
            )}
          </div>
        </main>
      </div>
    </div>
  );
};

export default App;
