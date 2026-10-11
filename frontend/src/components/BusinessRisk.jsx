import "./businessrisk.css";
import { useEffect, useState } from "react";

const API_URL = "http://127.0.0.1:8000/api/risk/business-scores/";

const formatNumber = (value) =>
  Number(value ?? 0).toLocaleString("en-IN", {
    maximumFractionDigits: 2,
  });

function BusinessRisk() {
  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    async function loadBusinessRisks() {
      try {
        const response = await fetch(API_URL);

        if (!response.ok) {
          throw new Error("Could not load business risk scores.");
        }

        const data = await response.json();
        setRecords(data.business_risk_scores ?? []);
      } catch {
        setError(
          "Unable to load business risk data. Check that Django is running."
        );
      } finally {
        setLoading(false);
      }
    }

    loadBusinessRisks();
  }, []);

  const highCount = records.filter(
    (row) => row.Business_Risk_Level === "High"
  ).length;

  const mediumCount = records.filter(
    (row) => row.Business_Risk_Level === "Medium"
  ).length;

  const lowCount = records.filter(
    (row) => row.Business_Risk_Level === "Low"
  ).length;

  return (
    <div className="business-risk-page">
      <section className="overview-welcome">
        <div>
          <span className="welcome-label">BUSINESS ANALYTICS</span>
          <h2>Business Risk Assessment</h2>
          <p>
            Review profitability and identify sub-categories that need
            business attention.
          </p>
        </div>
        <div className="model-pill">
          <span className="model-pulse" />
          Rule-based scoring
          <small>Business risk rules</small>
        </div>
      </section>

      {error && (
        <div className="error-message" role="alert">
          <span>⚠</span>
          <div>
            <strong>Unable to load business risks</strong>
            <p>{error}</p>
          </div>
        </div>
      )}

      {loading ? (
        <div className="loading-state">Loading business risk scores...</div>
      ) : !error ? (
        <>
          <section className="stats-grid">
            <article className="stat-card stat-red">
              <div className="stat-top">
                <span className="stat-title">High Risk</span>
                <span className="stat-icon red">!</span>
              </div>
              <h2>{highCount}</h2>
              <div className="stat-bottom">
                Sub-categories with negative total profit
              </div>
            </article>

            <article className="stat-card stat-orange">
              <div className="stat-top">
                <span className="stat-title">Medium Risk</span>
                <span className="stat-icon orange">!</span>
              </div>
              <h2>{mediumCount}</h2>
              <div className="stat-bottom">
                Positive profit but margin below 5%
              </div>
            </article>

            <article className="stat-card stat-green">
              <div className="stat-top">
                <span className="stat-title">Low Risk</span>
                <span className="stat-icon green">✓</span>
              </div>
              <h2>{lowCount}</h2>
              <div className="stat-bottom">
                Profit margin of at least 5%
              </div>
            </article>

            <article className="stat-card stat-blue">
              <div className="stat-top">
                <span className="stat-title">Total Sub-categories</span>
                <span className="stat-icon blue">▤</span>
              </div>
              <h2>{records.length}</h2>
              <div className="stat-bottom">
                Business groups assessed
              </div>
            </article>
          </section>

          <section className="panel">
            <div className="panel-heading">
              <div>
                <h2>Business Risk Scores</h2>
                <p>
                  Profitability, risk level, and reasons for each assessment
                </p>
              </div>
            </div>

            <div style={{ overflowX: "auto" }}>
              <table className="business-risk-table">
                <thead>
                  <tr>
                    <th>Sub-category</th>
                    <th>Total Sales</th>
                    <th>Total Profit</th>
                    <th>Profit Margin</th>
                    <th>Risk Level</th>
                    <th>Explanation</th>
                  </tr>
                </thead>
                <tbody>
                  {records.map((row) => (
                    <tr key={row["Sub-Category"]}>
                      <td>{row["Sub-Category"]}</td>
                      <td>{formatNumber(row.Total_Sales)}</td>
                      <td>{formatNumber(row.Total_Profit)}</td>
                      <td>{formatNumber(row.Profit_Margin_Percent)}%</td>
                      <td>
                        <span
                          className={`business-risk-badge ${String(
                            row.Business_Risk_Level ?? "Low"
                          ).toLowerCase()}`}
                        >
                          {row.Business_Risk_Level}
                        </span>
                      </td>
                      <td>{row.Business_Risk_Explanation}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <p className="business-risk-note">
              These risk levels use predefined rules. They indicate
              profitability concerns, not confirmed fraud or financial loss
              causes.
            </p>
          </section>
        </>
      ) : null}
    </div>
  );
}

export default BusinessRisk;
