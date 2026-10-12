import React, { useEffect, useState } from "react";
import "./customerDashboard.css";

const initialCustomers = [
  { id: 1, name: "Customer A", segment: "High Value", risk: "Low", spend: 24500, discount: "5%" },
  { id: 2, name: "Customer B", segment: "At Risk", risk: "High", spend: 8200, discount: "15%" },
  { id: 3, name: "Customer C", segment: "Regular", risk: "Medium", spend: 12600, discount: "10%" },
  { id: 4, name: "Customer D", segment: "High Value", risk: "Low", spend: 31200, discount: "5%" },
  { id: 5, name: "Customer E", segment: "At Risk", risk: "High", spend: 6500, discount: "20%" },
  { id: 6, name: "Customer F", segment: "Regular", risk: "Medium", spend: 14300, discount: "10%" },
];

const segmentColors = {
  "High Value": "#8b7cff",
  Regular: "#23c9b7",
  "At Risk": "#ff7185",
};

function CustomerDashboard() {
  const [customers, setCustomers] = useState(initialCustomers);
  const [search, setSearch] = useState("");
  const [segment, setSegment] = useState("All");
  const [apiStatus, setApiStatus] = useState("demo");

  // Optional API integration.
  // Set VITE_CUSTOMER_API_URL in your frontend .env file when the backend is ready.
  useEffect(() => {
    const apiUrl = import.meta.env.VITE_CUSTOMER_API_URL;

    if (!apiUrl) return;

    const controller = new AbortController();

    async function loadCustomers() {
      try {
        setApiStatus("loading");

        const response = await fetch(apiUrl, {
          signal: controller.signal,
        });

        if (!response.ok) {
          throw new Error("Customer API request failed");
        }

        const data = await response.json();
        const rows = Array.isArray(data) ? data : data.customers;

        if (!Array.isArray(rows)) {
          throw new Error("Unexpected customer API response");
        }

        setCustomers(rows);
        setApiStatus("connected");
      } catch (error) {
        if (error.name !== "AbortError") {
          console.error("Customer dashboard:", error);
          setApiStatus("on progress");
        }
      }
    }

    loadCustomers();

    return () => controller.abort();
  }, []);

  const filteredCustomers = customers.filter((customer) => {
    const matchesSearch = String(customer.name ?? "")
      .toLowerCase()
      .includes(search.toLowerCase());

    const matchesSegment =
      segment === "All" || customer.segment === segment;

    return matchesSearch && matchesSegment;
  });

  const totalSpend = customers.reduce(
    (sum, customer) => sum + (Number(customer.spend) || 0),
    0
  );

  const highRisk = customers.filter(
    (customer) => customer.risk === "High"
  ).length;

  const highValue = customers.filter(
    (customer) => customer.segment === "High Value"
  ).length;

  const segments = ["High Value", "Regular", "At Risk"];

  const formatCurrency = (amount) =>
    new Intl.NumberFormat("en-IN", {
      style: "currency",
      currency: "INR",
      maximumFractionDigits: 0,
    }).format(amount);

  return (
    <main className="customer-dashboard">
      <header className="customer-header">
        <div>
          <p className="customer-eyebrow">AXIO / CUSTOMER INTELLIGENCE</p>
          <h1>Customer Dashboard</h1>
          <p className="customer-subtitle">
            Understand customers, identify risk, and discover opportunities.
          </p>
        </div>

    
      </header>

      

      <section className="customer-kpi-grid">
        <article className="customer-kpi">
          <span className="kpi-label">Total Customers</span>
          <strong>{customers.length.toLocaleString("en-IN")}</strong>
          <span className="kpi-note">Customers in current dataset</span>
        </article>

        <article className="customer-kpi">
          <span className="kpi-label">Total Customer Spend</span>
          <strong>{formatCurrency(totalSpend)}</strong>
          <span className="kpi-note">Sum of spend in this dataset</span>
        </article>

        <article className="customer-kpi">
          <span className="kpi-label">High-Value Customers</span>
          <strong>{highValue}</strong>
          <span className="kpi-note">Customers in high-value segment</span>
        </article>

        <article className="customer-kpi">
          <span className="kpi-label">High-Risk Customers</span>
          <strong className="risk-number">{highRisk}</strong>
          <span className="kpi-note">Require attention</span>
        </article>
      </section>

      <section className="customer-panel">
        <div className="customer-section-heading">
          <div>
            <h2>Customer Segments</h2>
            <p>Distribution across the current dataset</p>
          </div>
        </div>

        <div className="customer-segment-grid">
          {segments.map((item) => {
            const count = customers.filter(
              (customer) => customer.segment === item
            ).length;

            const percentage = customers.length
              ? Math.round((count / customers.length) * 100)
              : 0;

            return (
              <article className="segment-card" key={item}>
                <div className="segment-card-top">
                  <span
                    className="segment-indicator"
                    style={{ backgroundColor: segmentColors[item] }}
                  />
                  <span>{item}</span>
                </div>

                <strong>{count}</strong>
                <div className="segment-progress">
                  <span
                    style={{
                      width: `${percentage}%`,
                      backgroundColor: segmentColors[item],
                    }}
                  />
                </div>
                <small>{percentage}% of customers</small>
              </article>
            );
          })}
        </div>
      </section>

      <section className="customer-panel">
        <div className="customer-section-heading customer-table-heading">
          <div>
            <h2>Customer Insights</h2>
            <p>Search and filter customer profiles</p>
          </div>

          <div className="customer-filters">
            <input
              type="search"
              placeholder="Search customers..."
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              aria-label="Search customers"
            />

            <select
              value={segment}
              onChange={(event) => setSegment(event.target.value)}
              aria-label="Filter customer segment"
            >
              <option value="All">All segments</option>
              {segments.map((item) => (
                <option value={item} key={item}>
                  {item}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="customer-table-wrapper">
          <table className="customer-table">
            <thead>
              <tr>
                <th>Customer</th>
                <th>Segment</th>
                <th>Risk Level</th>
                <th>Total Spend</th>
                <th>Suggested Discount</th>
              </tr>
            </thead>

            <tbody>
              {filteredCustomers.map((customer) => (
                <tr key={customer.id}>
                  <td>
                    <div className="customer-name">
                      <span className="customer-avatar">
                        {String(customer.name ?? "?").charAt(0).toUpperCase()}
                      </span>
                      {customer.name}
                    </div>
                  </td>

                  <td>
                    <span
                      className="customer-segment-badge"
                      style={{
                        color: segmentColors[customer.segment] || "#a8b2c8",
                      }}
                    >
                      {customer.segment ?? "Unknown"}
                    </span>
                  </td>

                  <td>
                    <span
                      className={`customer-risk-badge ${String(
                        customer.risk ?? "unknown"
                      ).toLowerCase()}`}
                    >
                      {customer.risk ?? "Unknown"}
                    </span>
                  </td>

                  <td>{formatCurrency(Number(customer.spend) || 0)}</td>
                  <td>{customer.discount ?? "—"}</td>
                </tr>
              ))}

              {filteredCustomers.length === 0 && (
                <tr>
                  <td colSpan="5" className="customer-empty">
                    No customers match your search.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>

      <footer className="customer-footer">
        AXIO CUSTOMER INTELLIGENCE
        <span>Insights depend on the connected dataset.</span>
      </footer>
    </main>
  );
}

export default CustomerDashboard;