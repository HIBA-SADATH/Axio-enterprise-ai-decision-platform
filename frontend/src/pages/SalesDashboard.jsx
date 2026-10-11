
import { useEffect, useState } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";
import "./SalesDashboard.css";

const money = (value) =>
  new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(Number(value) || 0);

const compactMoney = (value) =>
  `$${(Number(value) || 0).toLocaleString("en-US", {
    notation: "compact",
    maximumFractionDigits: 1,
  })}`;

const monthLabel = (value) => {
  if (!value) return "";
  const [year, month] = value.split("-");
  return new Date(Number(year), Number(month) - 1).toLocaleDateString(
    "en-US",
    { month: "short", year: "2-digit" }
  );
};

async function getApi(path) {
  const response = await fetch(path);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || `Request failed (${response.status})`);
  }
  return data;
}

function MetricCard({ label, value, note }) {
  return (
    <article className="sales-metric-card">
      <p className="sales-metric-label">{label}</p>
      <h2>{value}</h2>
      <p className="sales-metric-note">{note}</p>
    </article>
  );
}

function SalesDashboard() {
  const [summary, setSummary] = useState(null);
  const [monthly, setMonthly] = useState([]);
  const [forecast, setForecast] = useState([]);
  const [evaluation, setEvaluation] = useState(null);
  const [forecastMonths, setForecastMonths] = useState("6");
  const [testMonths, setTestMonths] = useState("6");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function loadDashboard() {
      setLoading(true);
      setError("");

      try {
        const [summaryData, monthlyData] = await Promise.all([
          getApi("/api/sales/summary/"),
          getApi("/api/sales/monthly/"),
        ]);

        if (cancelled) return;
        setSummary(summaryData);
        setMonthly(monthlyData.monthly_sales || []);

        const [forecastData, evaluationData] = await Promise.all([
          getApi(`/api/sales/forecast/?months=${forecastMonths}`),
          getApi(`/api/sales/evaluation/?test_months=${testMonths}`),
        ]);

        if (cancelled) return;
        setForecast(forecastData.forecast || []);
        setEvaluation(evaluationData);
      } catch (err) {
        if (!cancelled) {
          setError(
            `${err.message}. Make sure Django is running on port 8000 and its sales APIs are available.`
          );
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    loadDashboard();
    return () => {
      cancelled = true;
    };
  }, [forecastMonths, testMonths]);

  const forecastChart = [
    ...monthly.slice(-12).map((item) => ({
      month: item.month,
      historical: item.revenue,
    })),
    ...forecast.map((item) => ({
      month: item.month,
      forecast: item.forecast_revenue,
    })),
  ];

  const evaluationModels = evaluation?.models;

  return (
    <main className="sales-dashboard">
      <header className="sales-page-header">
        <div>
          <p className="sales-eyebrow">SALES &amp; DEMAND INTELLIGENCE</p>
          <h1>Sales Performance</h1>
          <p className="sales-subtitle">
            Explore historical revenue, category performance, and forecast
            scenarios.
          </p>
        </div>
        <span className="sales-status">
          <span className="sales-status-dot" />
          Analytics workspace
        </span>
      </header>

      {error && (
        <div className="sales-error" role="alert">
          <strong>Unable to load sales data</strong>
          <p>{error}</p>
          <button
            type="button"
            onClick={() => {
              setForecastMonths((current) => current);
              setTestMonths((current) =>
                current === "6" ? "3" : "6"
              );
            }}
          >
            Retry loading
          </button>
        </div>
      )}

      {loading && (
        <p className="sales-loading" role="status">
          Loading sales analytics…
        </p>
      )}

      {summary && (
        <>
          <section className="sales-metrics-grid" aria-label="Sales KPIs">
            <MetricCard
              label="Total revenue"
              value={money(summary.total_revenue)}
              note="Across available sales records"
            />
            <MetricCard
              label="Units sold"
              value={Number(summary.units_sold || 0).toLocaleString("en-US")}
              note={`${Number(summary.total_records || 0).toLocaleString("en-US")} sales records`}
            />
            <MetricCard
              label="Average sale"
              value={money(summary.average_sale_per_record)}
              note="Average revenue per record"
            />
            <MetricCard
              label="Product categories"
              value={(summary.sales_by_category || []).length}
              note="Categories represented in the dataset"
            />
          </section>

          <section className="sales-panel">
            <div className="sales-panel-header">
              <div>
                <h2>Monthly revenue trend</h2>
                <p>Historical revenue grouped by order month</p>
              </div>
              <span className="sales-period">
                {monthly.length
                  ? `${monthLabel(monthly[0].month)} – ${monthLabel(monthly[monthly.length - 1].month)}`
                  : "No data"}
              </span>
            </div>
            <div className="sales-chart">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={monthly}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis
                    dataKey="month"
                    tickFormatter={monthLabel}
                    minTickGap={28}
                  />
                  <YAxis tickFormatter={compactMoney} width={75} />
                  <Tooltip
                    labelFormatter={monthLabel}
                    formatter={(value) => [money(value), "Revenue"]}
                  />
                  <Line
                    type="monotone"
                    dataKey="revenue"
                    name="Revenue"
                    stroke="#3978f6"
                    strokeWidth={2.5}
                    dot={false}
                    activeDot={{ r: 5 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </section>

          <section className="sales-panel">
            <div className="sales-panel-header">
              <div>
                <h2>Revenue by category</h2>
                <p>Compare total sales across product categories</p>
              </div>
            </div>
            <div className="sales-chart sales-category-chart">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={summary.sales_by_category || []}
                  layout="vertical"
                  margin={{ left: 16, right: 28 }}
                >
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                  <XAxis type="number" tickFormatter={compactMoney} />
                  <YAxis
                    type="category"
                    dataKey="category"
                    width={100}
                  />
                  <Tooltip formatter={(value) => money(value)} />
                  <Bar
                    dataKey="revenue"
                    name="Revenue"
                    fill="#3978f6"
                    radius={[0, 5, 5, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </section>
        </>
      )}

      <section className="sales-panel">
        <div className="sales-panel-header sales-controls-header">
          <div>
            <h2>Revenue forecast</h2>
            <p>
              Historical data and forecast scenario from Exponential
              Smoothing
            </p>
          </div>
          <label className="sales-select-label">
            Forecast horizon
            <select
              value={forecastMonths}
              onChange={(event) => setForecastMonths(event.target.value)}
            >
              <option value="3">3 months</option>
              <option value="6">6 months</option>
              <option value="9">9 months</option>
              <option value="12">12 months</option>
            </select>
          </label>
        </div>

        {forecast.length > 0 && (
          <>
            <div className="sales-chart">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={forecastChart}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="month" tickFormatter={monthLabel} />
                  <YAxis tickFormatter={compactMoney} width={75} />
                  <Tooltip
                    labelFormatter={monthLabel}
                    formatter={(value, name) => [money(value), name]}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="historical"
                    name="Historical revenue"
                    stroke="#64748b"
                    strokeWidth={2}
                    dot={false}
                    connectNulls={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="forecast"
                    name="Forecast revenue"
                    stroke="#0d9b76"
                    strokeWidth={3}
                    strokeDasharray="6 4"
                    connectNulls={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="sales-forecast-grid">
              {forecast.map((item) => (
                <article className="sales-forecast-card" key={item.month}>
                  <span>{monthLabel(item.month)}</span>
                  <strong>{money(item.forecast_revenue)}</strong>
                </article>
              ))}
            </div>
            <p className="sales-data-note">
              Demonstration forecast based on historical data ending December
              2017. It is not a current business prediction.
            </p>
          </>
        )}
      </section>

      <section className="sales-panel">
        <div className="sales-panel-header sales-controls-header">
          <div>
            <h2>Forecast model evaluation</h2>
            <p>Compare model error against a simple naive baseline</p>
          </div>
          <label className="sales-select-label">
            Test period
            <select
              value={testMonths}
              onChange={(event) => setTestMonths(event.target.value)}
            >
              <option value="3">3 months</option>
              <option value="6">6 months</option>
              <option value="9">9 months</option>
              <option value="12">12 months</option>
            </select>
          </label>
        </div>

        {evaluationModels && (
          <>
            <div className="sales-evaluation-grid">
              <article className="sales-evaluation-card">
                <span>Exponential Smoothing</span>
                <p>MAE</p>
                <strong>{money(evaluationModels.exponential_smoothing.mae)}</strong>
                <p>RMSE</p>
                <strong>{money(evaluationModels.exponential_smoothing.rmse)}</strong>
              </article>
              <article className="sales-evaluation-card">
                <span>Naive baseline</span>
                <p>MAE</p>
                <strong>{money(evaluationModels.naive_baseline.mae)}</strong>
                <p>RMSE</p>
                <strong>{money(evaluationModels.naive_baseline.rmse)}</strong>
              </article>
            </div>

            <p className="sales-data-note">
              Evaluation period: {evaluation.actual_period_start} to{" "}
              {evaluation.actual_period_end}. Lower MAE and RMSE indicate
              smaller forecast errors. This is a historical holdout test, not
              current sales performance.
            </p>

            <div className="sales-chart">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={evaluation.monthly_results || []}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="month" tickFormatter={monthLabel} />
                  <YAxis tickFormatter={compactMoney} width={75} />
                  <Tooltip
                    labelFormatter={monthLabel}
                    formatter={(value, name) => [money(value), name]}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="actual_revenue"
                    name="Actual revenue"
                    stroke="#0d9b76"
                    strokeWidth={2.5}
                  />
                  <Line
                    type="monotone"
                    dataKey="exponential_smoothing"
                    name="Exponential Smoothing"
                    stroke="#3978f6"
                    strokeWidth={2}
                    strokeDasharray="5 4"
                  />
                  <Line
                    type="monotone"
                    dataKey="naive_baseline"
                    name="Naive baseline"
                    stroke="#e59b35"
                    strokeWidth={2}
                    strokeDasharray="3 3"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </>
        )}
      </section>

      <footer className="sales-dashboard-footer">
        Sales &amp; Demand Intelligence · Historical Superstore dataset
      </footer>
    </main>
  );
}

export default SalesDashboard;
