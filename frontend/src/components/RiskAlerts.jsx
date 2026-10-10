
import { useEffect, useState } from "react";

const API_URL = "http://127.0.0.1:8000/api/risk/alerts/";

const SHAP_FEATURES = [
  { key: "SHAP_Sales", label: "Sales" },
  { key: "SHAP_Quantity", label: "Quantity" },
  { key: "SHAP_Discount", label: "Discount" },
  { key: "SHAP_Profit", label: "Profit" },
  { key: "SHAP_Profit Margin", label: "Profit Margin" },
];

function getField(record, names) {
  for (const name of names) {
    if (record[name] !== undefined && record[name] !== null) {
      return record[name];
    }
  }
  return "—";
}

function formatNumber(value) {
  if (
    value === "—" ||
    value === "" ||
    value === null ||
    value === undefined
  ) {
    return "—";
  }

  const number = Number(value);

  return Number.isFinite(number)
    ? number.toLocaleString("en-IN", {
        maximumFractionDigits: 2,
      })
    : value;
}

function formatShap(value) {
  if (
    value === null ||
    value === undefined ||
    value === "" ||
    !Number.isFinite(Number(value))
  ) {
    return "—";
  }

  const number = Number(value);

  return `${number > 0 ? "+" : ""}${number.toFixed(5)}`;
}

function getShapFeatures(alert) {
  return SHAP_FEATURES
    .map((feature) => ({
      ...feature,
      value: alert[feature.key],
    }))
    .filter(
      (feature) =>
        feature.value !== null &&
        feature.value !== undefined &&
        feature.value !== "" &&
        Number.isFinite(Number(feature.value))
    );
}

function getStrongestFeature(alert) {
  const features = getShapFeatures(alert);

  if (features.length === 0) {
    return null;
  }

  return features.reduce((strongest, current) =>
    Math.abs(Number(current.value)) >
    Math.abs(Number(strongest.value))
      ? current
      : strongest
  );
}

function RiskAlerts() {
  const [alerts, setAlerts] = useState([]);
  const [page, setPage] = useState(1);
  const [totalAlerts, setTotalAlerts] = useState(0);
  const [totalPages, setTotalPages] = useState(1);
  const [criticalCount, setCriticalCount] = useState(0);
  const [highCount, setHighCount] = useState(0);
  const [filter, setFilter] = useState("All");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  
  
  // Load High and Critical counts from the summary API.
  useEffect(() => {
    const controller = new AbortController();

    async function loadAlertSummary() {
      try {
        const response = await fetch(
          "http://127.0.0.1:8000/api/risk/summary/",
          { signal: controller.signal }
        );

        if (!response.ok) {
          throw new Error("Failed to load risk summary.");
        }

        const data = await response.json();
        console.log("Risk summary response:", data);

        const counts = data.risk_counts ?? {};

        if (!controller.signal.aborted) {
          setCriticalCount(Number(counts.Critical ?? 0));
          setHighCount(Number(counts.High ?? 0));
        }
      } catch (err) {
        if (err.name !== "AbortError") {
          console.error("Alert summary error:", err);
          setError("Unable to load the alert summary.");
        }
      }
    }

    loadAlertSummary();

    return () => controller.abort();
  }, []);


  // Fetch alerts for the selected page and risk filter.
  useEffect(() => {
    const controller = new AbortController();

    async function loadAlerts() {
      setLoading(true);
      setError("");

      try {
        const params = new URLSearchParams({
          page: String(page),
          page_size: "10",
        });

        if (filter !== "All") {
          params.set("level", filter);
        }

        const response = await fetch(`${API_URL}?${params}`, {
          signal: controller.signal,
        });

        if (!response.ok) {
          throw new Error("Failed to load alerts.");
        }

        const data = await response.json();

        setAlerts(data.results);
        setTotalAlerts(data.total_alerts);
        setTotalPages(data.total_pages);
      } catch (err) {
        if (err.name !== "AbortError") {
          setError("Unable to load risk alerts.");
        }
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      }
    }

    loadAlerts();

    return () => controller.abort();
  }, [page, filter]);

  function changeFilter(event) {
    setFilter(event.target.value);
    setPage(1);
  }

  return (
    <section className="records-panel">
      <div className="records-heading">
        <div>
          <h2>Risk Alerts</h2>
          <p>
            High and Critical risk records requiring attention.
          </p>
        </div>

        <span className="risk-badge critical">
          {totalAlerts.toLocaleString("en-IN")} alerts
        </span>
      </div>

      <div className="alert-summary">
        <div className="alert-summary-card critical-summary">
          <span>Critical Alerts</span>
          <strong>{criticalCount.toLocaleString("en-IN")}</strong>
          <small>Highest risk level</small>
        </div>

        <div className="alert-summary-card high-summary">
          <span>High-Risk Alerts</span>
          <strong>{highCount.toLocaleString("en-IN")}</strong>
          <small>Require attention</small>
        </div>
      </div>

      <div className="records-actions">
        <select value={filter} onChange={changeFilter}>
          <option value="All">All alerts</option>
          <option value="Critical">Critical only</option>
          <option value="High">High only</option>
        </select>
      </div>

      {error && (
        <div className="error-message" role="alert">
          {error}
        </div>
      )}

      {loading ? (
        <p className="loading">Loading alerts...</p>
      ) : (
        <>
          <div className="records-count">
            {totalAlerts.toLocaleString("en-IN")} matching alerts
          </div>

          <div className="table-wrapper">
            <table className="risk-table">
              <thead>
                <tr>
                  <th>Alert No.</th>
                  <th>Risk Level</th>
                  <th>Anomaly</th>
                  <th>Sales</th>
                  <th>Profit</th>
                  <th>Discount</th>
                  <th>Quantity</th>
                  <th>Why Flagged?</th>
                  <th>SHAP Explanation</th>
                  <th>Feature Contributions</th>
                </tr>
              </thead>

              <tbody>
                {alerts.map((alert, index) => {
                  const profit = getField(alert, ["Profit", "profit"]);
                  const strongest = getStrongestFeature(alert);
                  const features = getShapFeatures(alert);

                  const anomaly =
                    String(alert.Anomaly) === "1" ||
                    alert.Anomaly === true ||
                    String(alert.Anomaly).toLowerCase() === "anomaly";

                  return (
                    <tr key={`${page}-${index}`}>
                      <td>{(page - 1) * 10 + index + 1}</td>

                      <td>
                        <span
                          className={`risk-badge ${String(
                            alert["Risk Level"] ?? "Unknown"
                          ).toLowerCase()}`}
                        >
                          {alert["Risk Level"] ?? "Unknown"}
                        </span>
                      </td>

                      <td>{anomaly ? "Detected" : "Normal"}</td>

                      <td>
                        {formatNumber(
                          getField(alert, ["Sales", "sales"])
                        )}
                      </td>

                      <td
                        className={
                          profit !== "—" &&
                          Number.isFinite(Number(profit)) &&
                          Number(profit) < 0
                            ? "negative-profit"
                            : ""
                        }
                      >
                        {formatNumber(profit)}
                      </td>

                      <td>
                        {formatNumber(
                          getField(alert, ["Discount", "discount"])
                        )}
                      </td>

                      <td>
                        {formatNumber(
                          getField(alert, ["Quantity", "quantity"])
                        )}
                      </td>

                      <td>
                        {getField(alert, ["Risk Explanation"]) === "—"
                          ? "No explanation available"
                          : getField(alert, ["Risk Explanation"])}
                      </td>

                      <td>
                        {strongest ? (
                          <div className="shap-explanation">
                            <strong>{strongest.label}</strong>
                            <span>
                              Strongest contributing feature
                            </span>
                            <small>
                              SHAP value: {formatShap(strongest.value)}
                            </small>
                          </div>
                        ) : (
                          <span className="shap-unavailable">
                            Explanation not available
                          </span>
                        )}
                      </td>

                      <td>
                        {features.length > 0 ? (
                          <div className="shap-contributions">
                            {features.map((feature) => (
                              <div
                                className="shap-feature-row"
                                key={feature.key}
                              >
                                <span>{feature.label}</span>
                                <strong
                                  className={
                                    Number(feature.value) < 0
                                      ? "shap-negative"
                                      : "shap-positive"
                                  }
                                >
                                  {formatShap(feature.value)}
                                </strong>
                              </div>
                            ))}
                          </div>
                        ) : (
                          <span className="shap-unavailable">
                            No SHAP values
                          </span>
                        )}
                      </td>
                    </tr>
                  );
                })}

                {alerts.length === 0 && (
                  <tr>
                    <td colSpan="10">No alerts found.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          <p className="shap-note">
            SHAP values show how features contribute to the model output.
            They do not prove fraud or establish a business cause.
            Explanations are shown only for records with available SHAP values.
          </p>

          <div className="pagination">
            <button
              disabled={page <= 1}
              onClick={() => setPage((current) => current - 1)}
            >
              Previous
            </button>

            <span>
              Page {page} of {totalPages}
            </span>

            <button
              disabled={page >= totalPages}
              onClick={() => setPage((current) => current + 1)}
            >
              Next
            </button>
          </div>
        </>
      )}
    </section>
  );
}

export default RiskAlerts;
