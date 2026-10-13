import { useCallback, useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import "./CustomerDashboard.css";

const API_URL = "http://127.0.0.1:8000/api/customers/summary/";
const SEGMENT_COLORS = ["#1d5fc4", "#5b4bb0", "#1b8a5a"];
const formatNumber = (value) => Number(value ?? 0).toLocaleString("en-IN");
const formatCurrency = (value) => Number(value ?? 0).toLocaleString("en-IN", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
const formatPercent = (value) => `${Number(value ?? 0).toFixed(1)}%`;

function CustomerDashboard() {
  const [summary, setSummary] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState(null);

  const loadSummary = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const response = await fetch(API_URL);
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Unable to load customer analysis.");
      setSummary(data);
      setLastUpdated(new Date());
    } catch (requestError) {
      setError(requestError.message || "Could not connect to Django. Check that the backend server is running.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const requestTimer = window.setTimeout(loadSummary, 0);
    return () => window.clearTimeout(requestTimer);
  }, [loadSummary]);

  const statCards = [
    ["Customer accounts", summary ? formatNumber(summary.total_customers) : "—", "Historical records analysed", "◉"],
    ["Total sales", summary ? formatCurrency(summary.total_sales) : "—", "Across the customer dataset", "$"],
    ["Labeled churn accounts", summary ? formatNumber(summary.churned_customers) : "—", summary ? `${formatPercent(summary.churn_rate)} of records` : "Historical churn label", "!"],
    ["Average sales per account", summary ? formatCurrency(summary.average_sales_per_customer) : "—", "Mean historical sales", "↗"],
  ];

  return <div className="customer-dashboard">
    <header className="customer-header">
      <div><p className="eyebrow">ENTERPRISE AI / CUSTOMER ANALYTICS</p><h1>Customer Intelligence</h1><p>Customer behavior analysis and churn-label insights from the available dataset.</p></div>
      <div className="customer-header-actions"><span className="customer-status"><span className="status-dot" />{error ? "BACKEND OFFLINE" : "DATASET CONNECTED"}</span><button className="refresh-button" type="button" onClick={loadSummary} disabled={loading}>{loading ? "Loading…" : "↻ Refresh"}</button></div>
    </header>

    {error && <section className="customer-error" role="alert"><div><strong>Customer analysis could not be loaded.</strong><p>{error}</p></div><button type="button" onClick={loadSummary}>Try again</button></section>}
    <section className="customer-stats" aria-label="Customer statistics">{statCards.map(([label, value, detail, icon]) => <article className="customer-stat-card" key={label}><div className="customer-stat-top"><span>{label}</span><span className="customer-stat-icon">{icon}</span></div><strong>{value}</strong><small>{detail}</small></article>)}</section>

    {loading && !summary ? <section className="customer-loading"><span />Loading customer analytics…</section> : summary && <>
      <section className="customer-chart-grid">
        <article className="customer-panel"><div className="customer-panel-heading"><div><h2>Customer segments</h2><p>Distribution of customer accounts by segment</p></div></div><ResponsiveContainer width="100%" height={300}><PieChart><Pie data={summary.segment_breakdown} dataKey="customers" nameKey="name" innerRadius={72} outerRadius={108} paddingAngle={3}>{summary.segment_breakdown.map((entry, index) => <Cell key={entry.name} fill={SEGMENT_COLORS[index % SEGMENT_COLORS.length]} />)}</Pie><Tooltip formatter={(value) => [formatNumber(value), "Accounts"]} /><Legend /></PieChart></ResponsiveContainer></article>
        <article className="customer-panel"><div className="customer-panel-heading"><div><h2>Churn label rate by region</h2><p>Observed historical churn labels, not live model predictions</p></div></div><ResponsiveContainer width="100%" height={300}><BarChart data={summary.region_breakdown} margin={{ top: 8, right: 8, left: -18 }}><CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#dfe4ec" /><XAxis dataKey="name" tickLine={false} axisLine={false} /><YAxis tickFormatter={(value) => `${value}%`} tickLine={false} axisLine={false} /><Tooltip formatter={(value) => [formatPercent(value), "Churn label rate"]} /><Bar dataKey="churn_rate" name="Churn label rate" fill="#d03a55" radius={[5, 5, 0, 0]} /></BarChart></ResponsiveContainer></article>
      </section>
      <section className="customer-chart-grid">
        <article className="customer-panel"><div className="customer-panel-heading"><div><h2>Sales by category</h2><p>Total historical sales for each product category</p></div></div><ResponsiveContainer width="100%" height={300}><BarChart data={summary.sales_by_category} layout="vertical" margin={{ top: 5, right: 18, left: 25, bottom: 5 }}><CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#dfe4ec" /><XAxis type="number" tickFormatter={(value) => `$${Math.round(value / 1000)}k`} tickLine={false} axisLine={false} /><YAxis dataKey="name" type="category" width={105} tickLine={false} axisLine={false} /><Tooltip formatter={(value) => [formatCurrency(value), "Sales"]} /><Bar dataKey="sales" name="Sales" fill="#1d5fc4" radius={[0, 5, 5, 0]} /></BarChart></ResponsiveContainer></article>
        <article className="customer-panel"><div className="customer-panel-heading"><div><h2>Highest-risk product areas</h2><p>Sub-categories ranked by observed churn-label rate</p></div></div><div className="customer-risk-list">{summary.highest_risk_subcategories.map((item) => <div className="customer-risk-row" key={item.name}><div><strong>{item.name}</strong><span>{formatNumber(item.customers)} accounts</span></div><div className="customer-risk-rate"><strong>{formatPercent(item.churn_rate)}</strong><div><span style={{ width: `${Math.min(item.churn_rate, 100)}%` }} /></div></div></div>)}</div></article>
      </section>
      <section className="customer-panel customer-table-panel"><div className="customer-panel-heading"><div><h2>Segment performance</h2><p>Customer account count, historical sales, and observed churn labels</p></div></div><div className="customer-table-wrap"><table><thead><tr><th>Segment</th><th>Accounts</th><th>Sales</th><th>Labeled churn</th><th>Churn rate</th></tr></thead><tbody>{summary.segment_breakdown.map((item) => <tr key={item.name}><td>{item.name}</td><td>{formatNumber(item.customers)}</td><td>{formatCurrency(item.sales)}</td><td>{formatNumber(item.churned_customers)}</td><td>{formatPercent(item.churn_rate)}</td></tr>)}</tbody></table></div></section>
      <p className="customer-data-note">{summary.data_note} {lastUpdated && `Last refreshed ${lastUpdated.toLocaleTimeString("en-IN")}.`}</p>
    </>}
  </div>;
}

export default CustomerDashboard;
