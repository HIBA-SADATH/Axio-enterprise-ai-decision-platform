
import { useCallback, useEffect, useState } from "react";
import RiskRecords from "../components/RiskRecords";
import RiskAlerts from "../components/RiskAlerts";
import BusinessRisk from "../components/BusinessRisk";
import "./RiskDashboard.css";

import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

const API_URL = "http://127.0.0.1:8000/api/risk/summary/";

const RISK_LEVELS = ["Low", "Medium", "High", "Critical"];

const RISK_COLORS = {
  Low: "#22c55e",
  Medium: "#eab308",
  High: "#f97316",
  Critical: "#ef4444",
};

const formatNumber = (value) =>
  Number(value ?? 0).toLocaleString("en-IN");

const formatPercent = (value) =>
  `${Number(value ?? 0).toFixed(1)}%`;

function RiskDashboard() {
  const [activePage, setActivePage] = useState("Overview");
  const [summary, setSummary] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState(null);

  const loadSummary = useCallback(async () => {
    setLoading(true);
    setError("");

    try {
      const response = await fetch(API_URL);

      if (!response.ok) {
        throw new Error("Unable to load risk summary.");
      }

      const data = await response.json();
      setSummary(data);
      setLastUpdated(new Date());
    } catch {
      setError(
        "Could not connect to Django. Check that the backend server is running."
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadSummary();
  }, [loadSummary]);

  const riskCounts = summary?.risk_counts ?? {};
  const totalRecords = Number(summary?.total_records ?? 0);
  const normalRecords = Number(summary?.normal_records ?? 0);
  const anomalyRecords = Number(summary?.anomaly_records ?? 0);
  const highCount = Number(riskCounts.High ?? 0);
  const criticalCount = Number(riskCounts.Critical ?? 0);
  const priorityCount = highCount + criticalCount;

  const anomalyRate = totalRecords
    ? (anomalyRecords / totalRecords) * 100
    : 0;

  const priorityRate = totalRecords
    ? (priorityCount / totalRecords) * 100
    : 0;

  const riskChartData = RISK_LEVELS.map((name) => ({
    name,
    value: Number(riskCounts[name] ?? 0),
  }));

  const riskCards = [
    {
      title: "Total Records",
      value: totalRecords,
      detail: "Transactions analyzed",
      color: "blue",
      icon: "▤",
    },
    {
      title: "Normal Records",
      value: normalRecords,
      detail: "No detected anomaly",
      color: "green",
      icon: "✓",
    },
    {
      title: "Anomalies Detected",
      value: anomalyRecords,
      detail: `${formatPercent(anomalyRate)} of all records`,
      color: "orange",
      icon: "⌁",
    },
    {
      title: "Critical Risks",
      value: criticalCount,
      detail: "Requires priority review",
      color: "red",
      icon: "!",
    },
  ];

  const pageTitles = {
    Overview: "Risk Intelligence Dashboard",
    Records: "Risk Records",
    Alerts: "Risk Alerts",
    "Business Risk": "Business Risk",
  };

  const pageDescriptions = {
    Overview:
      "Monitor unusual transactions and business risk in one place.",
    Records:
      "Search, filter, and explore analyzed transaction records.",
    Alerts:
      "Review high-priority transactions that need attention.",
    "Business Risk":
      "Assess business profitability and sub-category risk.",
  };

  return (
    <div className="dashboard">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">R</div>
          <div>
            <h2>RiskIQ</h2>
            <p className="sidebar-label">INTELLIGENCE PLATFORM</p>
          </div>
        </div>

        <p className="nav-heading">WORKSPACE</p>

        <nav>
          <button
            className={`nav-item ${
              activePage === "Overview" ? "active" : ""
            }`}
            onClick={() => setActivePage("Overview")}
          >
            <span>◈</span> Overview
          </button>

          <button
            className={`nav-item ${
              activePage === "Records" ? "active" : ""
            }`}
            onClick={() => setActivePage("Records")}
          >
            <span>▤</span> Risk Records
          </button>

          <button
            className={`nav-item ${
              activePage === "Alerts" ? "active" : ""
            }`}
            onClick={() => setActivePage("Alerts")}
          >
            <span>⚠</span> Risk Alerts
            {priorityCount > 0 && (
              <span className="nav-count">{priorityCount}</span>
            )}
          </button>

          <button
            className={`nav-item ${
              activePage === "Business Risk" ? "active" : ""
            }`}
            onClick={() => setActivePage("Business Risk")}
          >
            <span>▥</span> Business Risk
          </button>
        </nav>

        <div className="sidebar-spacer" />

        <div className="engine-card">
          <span className="status-dot" />
          <div>
            <strong>Risk Engine</strong>
            <small>
              {error ? "Connection issue" : "AI model configured"}
            </small>
          </div>
          <span className="engine-status">
            {error ? "!" : "●"}
          </span>
        </div>

        <p className="sidebar-bottom">
          ENTERPRISE RISK INTELLIGENCE
        </p>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <p className="eyebrow">
              ENTERPRISE AI / RISK ANALYTICS
            </p>
            <h1>{pageTitles[activePage]}</h1>
            <p className="subtitle">
              {pageDescriptions[activePage]}
            </p>
          </div>

          <div className="topbar-actions">
            <span className="live-badge">
              <span className="status-dot" />
              {error ? "BACKEND OFFLINE" : "RISK MONITOR"}
            </span>

            <button
              className="refresh-button"
              onClick={loadSummary}
              disabled={loading}
              title="Refresh dashboard data"
            >
              <span className={loading ? "refresh-spin" : ""}>
                ↻
              </span>
              {loading ? "Loading" : "Refresh"}
            </button>
          </div>
        </header>

        {error && activePage !== "Business Risk" && (
          <div className="error-message" role="alert">
            <span>⚠</span>
            <div>
              <strong>Unable to load dashboard data</strong>
              <p>{error}</p>
            </div>
            <button onClick={loadSummary}>Try again</button>
          </div>
        )}

        {activePage === "Overview" && (
          <>
            <section className="overview-welcome">
              <div>
                <span className="welcome-label">OVERVIEW</span>
                <h2>Your business risk at a glance</h2>
                <p>
                  AI-assisted transaction analysis and risk
                  prioritization.
                </p>
              </div>
              <div className="model-pill">
                <span className="model-pulse" />
                Isolation Forest
                <small>Detection model</small>
              </div>
            </section>

            <section className="stats-grid">
              {riskCards.map((card) => (
                <article
                  className={`stat-card stat-${card.color}`}
                  key={card.title}
                >
                  <div className="stat-top">
                    <span className="stat-title">{card.title}</span>
                    <span className={`stat-icon ${card.color}`}>
                      {card.icon}
                    </span>
                  </div>

                  <h2>
                    {summary ? formatNumber(card.value) : "—"}
                  </h2>

                  <div className="stat-bottom">
                    <span className="stat-indicator" />
                    <span>{card.detail}</span>
                  </div>
                </article>
              ))}
            </section>

            <section className="content-grid">
              <article className="panel risk-panel">
                <div className="panel-heading">
                  <div>
                    <h2>Risk Distribution</h2>
                    <p>
                      Breakdown of records by assigned risk level
                    </p>
                  </div>
                  <span className="panel-icon">◉</span>
                </div>

                {loading && !summary ? (
                  <div className="loading-state">
                    <span className="loading-spinner" />
                    Loading risk distribution...
                  </div>
                ) : summary ? (
                  <>
                    <div className="chart-wrap">
                      <div className="chart-center">
                        <strong>{formatNumber(totalRecords)}</strong>
                        <span>Total records</span>
                      </div>

                      <ResponsiveContainer width="100%" height={270}>
                        <PieChart>
                          <Pie
                            data={riskChartData}
                            dataKey="value"
                            nameKey="name"
                            cx="50%"
                            cy="50%"
                            innerRadius={76}
                            outerRadius={105}
                            paddingAngle={3}
                            stroke="none"
                          >
                            {riskChartData.map((entry) => (
                              <Cell
                                key={entry.name}
                                fill={RISK_COLORS[entry.name]}
                              />
                            ))}
                          </Pie>
                          <Tooltip
                            formatter={(value, name) => [
                              `${formatNumber(value)} records`,
                              `${name} Risk`,
                            ]}
                            contentStyle={{
                              background: "#171e2d",
                              border: "1px solid #303b50",
                              borderRadius: "10px",
                              color: "#ffffff",
                            }}
                          />
                          <Legend
                            verticalAlign="bottom"
                            iconType="circle"
                            formatter={(value) => (
                              <span className="chart-legend-label">
                                {value}
                              </span>
                            )}
                          />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>

                    <div className="risk-list">
                      {RISK_LEVELS.map((level) => {
                        const count = Number(
                          riskCounts[level] ?? 0
                        );
                        const percentage = totalRecords
                          ? (count / totalRecords) * 100
                          : 0;

                        return (
                          <div className="risk-row" key={level}>
                            <div className="risk-row-label">
                              <span
                                className="risk-dot"
                                style={{
                                  backgroundColor:
                                    RISK_COLORS[level],
                                }}
                              />
                              <span>{level} Risk</span>
                              <strong>{formatNumber(count)}</strong>
                              <span className="risk-percentage">
                                {formatPercent(percentage)}
                              </span>
                            </div>

                            <div className="progress-track">
                              <div
                                className="progress-fill"
                                style={{
                                  width: `${Math.min(
                                    percentage,
                                    100
                                  )}%`,
                                  backgroundColor:
                                    RISK_COLORS[level],
                                }}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </>
                ) : (
                  <p className="loading">
                    No summary data available.
                  </p>
                )}
              </article>

              <article className="panel alerts-panel">
                <div className="panel-heading">
                  <div>
                    <h2>Priority Risk Overview</h2>
                    <p>High and critical risk records</p>
                  </div>
                  <span className="alert-icon">!</span>
                </div>

                <div className="priority-total">
                  <span>Priority records</span>
                  <strong>
                    {summary ? formatNumber(priorityCount) : "—"}
                  </strong>
                  <small>
                    {summary
                      ? `${formatPercent(priorityRate)} of all records`
                      : "Waiting for data"}
                  </small>
                </div>

                <div className="priority-breakdown">
                  <div className="priority-row">
                    <span className="priority-dot high-dot" />
                    <span>High Risk</span>
                    <strong>
                      {summary ? formatNumber(highCount) : "—"}
                    </strong>
                  </div>
                  <div className="priority-row">
                    <span className="priority-dot critical-dot" />
                    <span>Critical Risk</span>
                    <strong>
                      {summary
                        ? formatNumber(criticalCount)
                        : "—"}
                    </strong>
                  </div>
                </div>

                <div className="alert-note">
                  <span>⚠</span>
                  <p>
                    Review these transactions first. The assigned
                    risk levels are model outputs and should be
                    verified before making business decisions.
                  </p>
                </div>

                <button
                  className="view-alerts-button"
                  onClick={() => setActivePage("Alerts")}
                >
                  View risk alerts <span>→</span>
                </button>
              </article>
            </section>

            <section className="panel detection-panel">
              <div className="panel-heading">
                <div>
                  <h2>Anomaly Detection Summary</h2>
                  <p>
                    High-level indicators from the analyzed dataset
                  </p>
                </div>
                <span className="panel-icon">⌁</span>
              </div>

              <div className="detection-grid">
                <div className="detection-item">
                  <span className="detection-label">
                    Anomaly rate
                  </span>
                  <strong>
                    {summary ? formatPercent(anomalyRate) : "—"}
                  </strong>
                  <div className="mini-track">
                    <div
                      className="mini-fill orange-fill"
                      style={{
                        width: `${Math.min(anomalyRate, 100)}%`,
                      }}
                    />
                  </div>
                  <small>
                    Share of records flagged as anomalous
                  </small>
                </div>

                <div className="detection-item">
                  <span className="detection-label">
                    Normal records
                  </span>
                  <strong>
                    {summary
                      ? formatPercent(
                          totalRecords
                            ? (normalRecords / totalRecords) * 100
                            : 0
                        )
                      : "—"}
                  </strong>
                  <div className="mini-track">
                    <div
                      className="mini-fill green-fill"
                      style={{
                        width: `${
                          totalRecords
                            ? Math.min(
                                (normalRecords / totalRecords) * 100,
                                100
                              )
                            : 0
                        }%`,
                      }}
                    />
                  </div>
                  <small>
                    Records not flagged by the current model
                  </small>
                </div>

                <div className="detection-item">
                  <span className="detection-label">
                    High-priority rate
                  </span>
                  <strong>
                    {summary ? formatPercent(priorityRate) : "—"}
                  </strong>
                  <div className="mini-track">
                    <div
                      className="mini-fill red-fill"
                      style={{
                        width: `${Math.min(priorityRate, 100)}%`,
                      }}
                    />
                  </div>
                  <small>
                    High and critical records combined
                  </small>
                </div>
              </div>
            </section>

            <div className="dashboard-meta">
              <span>
                <span className="status-dot" />
                {loading ? "Updating data..." : "Dashboard status"}
                {!loading && !error && " · Connected"}
              </span>
              <span>
                {lastUpdated
                  ? `Last updated ${lastUpdated.toLocaleTimeString(
                      "en-IN"
                    )}`
                  : "Waiting for backend data"}
              </span>
            </div>
          </>
        )}

        {activePage === "Records" && <RiskRecords />}
        {activePage === "Alerts" && <RiskAlerts />}
        {activePage === "Business Risk" && <BusinessRisk />}

        <footer>
          <span>
            RiskIQ · Enterprise AI Decision Intelligence Platform
          </span>
          <span>Model: Isolation Forest</span>
        </footer>
      </main>
    </div>
  );
}




export default RiskDashboard;