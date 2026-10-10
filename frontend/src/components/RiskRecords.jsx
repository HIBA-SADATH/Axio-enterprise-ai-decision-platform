
import { useEffect, useMemo, useState } from "react";

const API_URL = "http://127.0.0.1:8000/api/risk/records/";
const PAGE_SIZE = 10;

// Read a field even if the dataset uses slightly different column names.
function getField(record, names) {
  for (const name of names) {
    if (record[name] !== undefined && record[name] !== null) {
      return record[name];
    }
  }
  return "—";
}

function formatNumber(value) {
  if (value === "—" || value === "") return "—";

  const number = Number(value);

  return Number.isFinite(number)
    ? number.toLocaleString("en-IN", {
        maximumFractionDigits: 2,
      })
    : value;
}

function RiskRecords() {
  const [allRecords, setAllRecords] = useState([]);
  const [search, setSearch] = useState("");
  const [level, setLevel] = useState("All");
  const [sortOrder, setSortOrder] = useState("default");
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [exporting, setExporting] = useState(false);

  // Load every record matching the selected risk level.
  useEffect(() => {
    const controller = new AbortController();

    async function fetchAllRecords() {
      setLoading(true);
      setError("");
      setAllRecords([]);

      try {
        const results = [];
        let currentPage = 1;
        let totalPages = 1;

        do {
          const params = new URLSearchParams({
            page: String(currentPage),
            page_size: "100",
          });

          if (level !== "All") {
            params.set("level", level);
          }

          const response = await fetch(`${API_URL}?${params}`, {
            signal: controller.signal,
          });

          if (!response.ok) {
            throw new Error("Failed to load risk records.");
          }

          const data = await response.json();

          results.push(...data.results);
          totalPages = Math.ceil(
            data.total_records / data.page_size
          );

          currentPage += 1;
        } while (currentPage <= totalPages);

        if (!controller.signal.aborted) {
          setAllRecords(results);
        }
      } catch (err) {
        if (err.name !== "AbortError") {
          setError(
            "Unable to load records. Please check that the Django server is running."
          );
        }
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      }
    }

    fetchAllRecords();

    return () => controller.abort();
  }, [level]);

  function changeLevel(event) {
    setLevel(event.target.value);
    setPage(1);
  }

  // Search across all loaded records.
  const filteredRecords = useMemo(() => {
    const query = search.toLowerCase().trim();

    const filtered = allRecords.filter((record) => {
      if (!query) return true;

      return Object.values(record).some((value) =>
        String(value ?? "").toLowerCase().includes(query)
      );
    });

    const priority = {
      Critical: 4,
      High: 3,
      Medium: 2,
      Low: 1,
    };

    if (sortOrder === "critical-first") {
      filtered.sort(
        (a, b) =>
          (priority[b["Risk Level"]] ?? 0) -
          (priority[a["Risk Level"]] ?? 0)
      );
    } else if (sortOrder === "low-first") {
      filtered.sort(
        (a, b) =>
          (priority[a["Risk Level"]] ?? 0) -
          (priority[b["Risk Level"]] ?? 0)
      );
    }

    return filtered;
  }, [allRecords, search, sortOrder]);

  // Pagination happens after searching and sorting.
  const totalPages = Math.max(
    1,
    Math.ceil(filteredRecords.length / PAGE_SIZE)
  );

  const safePage = Math.min(page, totalPages);

  const visibleRecords = filteredRecords.slice(
    (safePage - 1) * PAGE_SIZE,
    safePage * PAGE_SIZE
  );

  // Export all records matching the selected risk level.
  function exportCSV() {
    setExporting(true);
    setError("");

    try {
      if (allRecords.length === 0) {
        setError("There are no records to export.");
        return;
      }

      const columns = [
        ...new Set(allRecords.flatMap((record) => Object.keys(record))),
      ];

      const escapeCSV = (value) =>
        `"${String(value ?? "").replace(/"/g, '""')}"`;

      const csvContent = [
        columns.map(escapeCSV).join(","),
        ...allRecords.map((record) =>
          columns.map((column) => escapeCSV(record[column])).join(",")
        ),
      ].join("\r\n");

      const blob = new Blob(["\uFEFF", csvContent], {
        type: "text/csv;charset=utf-8;",
      });

      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");

      link.href = url;
      link.download = `risk_records_${level.toLowerCase()}.csv`;

      document.body.appendChild(link);
      link.click();
      link.remove();

      URL.revokeObjectURL(url);
    } catch {
      setError("CSV export failed. Please try again.");
    } finally {
      setExporting(false);
    }
  }

  return (
    <section className="records-panel">
      <div className="records-heading">
        <div>
          <h2>Risk Records</h2>
          <p>Review transactions identified by the risk engine.</p>
        </div>

        <div className="records-actions">
          <input
            type="text"
            placeholder="Search all records..."
            value={search}
            onChange={(event) => {
              setSearch(event.target.value);
              setPage(1);
            }}
          />

          <select
            value={sortOrder}
            onChange={(event) => {
              setSortOrder(event.target.value);
              setPage(1);
            }}
          >
            <option value="default">Default order</option>
            <option value="critical-first">Critical risk first</option>
            <option value="low-first">Low risk first</option>
          </select>

          <select value={level} onChange={changeLevel}>
            <option value="All">All risk levels</option>
            <option value="Low">Low</option>
            <option value="Medium">Medium</option>
            <option value="High">High</option>
            <option value="Critical">Critical</option>
          </select>

          <button
            onClick={exportCSV}
            disabled={exporting || loading}
          >
            {exporting
              ? "Exporting..."
              : loading
                ? "Loading records..."
                : "Export CSV"}
          </button>
        </div>
      </div>

      <p className="records-count">
        {loading
          ? "Loading records..."
          : `${filteredRecords.length.toLocaleString("en-IN")} matching records`}
      </p>

      {error && <div className="error-message">{error}</div>}

      {loading ? (
        <p className="loading">Loading records...</p>
      ) : (
        <>
          <div className="table-wrapper">
            <table className="risk-table">
              <thead>
                <tr>
                  <th>Record</th>
                  <th>Risk Level</th>
                  <th>Anomaly</th>
                  <th>Sales</th>
                  <th>Profit</th>
                  <th>Discount</th>
                  <th>Quantity</th>
                  <th>Risk Explanation</th>
                </tr>
              </thead>

              <tbody>
                {visibleRecords.map((record, index) => {
                  const riskLevel = getField(record, ["Risk Level"]);

                  const anomaly = getField(record, ["Anomaly"]);

                  const isAnomaly =
                    String(anomaly) === "1" ||
                    anomaly === true ||
                    String(anomaly).toLowerCase() === "true";

                  const sales = getField(record, ["Sales", "sales"]);
                  const profit = getField(record, ["Profit", "profit"]);
                  const discount = getField(record, ["Discount", "discount"]);
                  const quantity = getField(record, ["Quantity", "quantity"]);

                  return (
                    <tr key={record.id ?? `${safePage}-${index}`}>
                      <td>
                        {(safePage - 1) * PAGE_SIZE + index + 1}
                      </td>

                      <td>
                        <span
                          className={`risk-badge ${String(
                            riskLevel
                          ).toLowerCase()}`}
                        >
                          {riskLevel}
                        </span>
                      </td>

                      <td>{isAnomaly ? "Detected" : "Normal"}</td>

                      <td>{formatNumber(sales)}</td>

                      <td
                        className={
                          Number(profit) < 0 ? "negative-profit" : ""
                        }
                      >
                        {formatNumber(profit)}
                      </td>

                      <td>{formatNumber(discount)}</td>

                      <td>{formatNumber(quantity)}</td>

                      <td>
                        {getField(record, ["Risk Explanation"]) === "—"
                          ? "No explanation available"
                          : getField(record, ["Risk Explanation"])}
                      </td>
                    </tr>
                  );
                })}

                {visibleRecords.length === 0 && (
                  <tr>
                    <td colSpan="8">No matching records found.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          <div className="pagination">
            <button
              disabled={safePage <= 1}
              onClick={() => setPage((current) => current - 1)}
            >
              Previous
            </button>

            <span>
              Page {safePage} of {totalPages}
            </span>

            <button
              disabled={safePage >= totalPages}
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

export default RiskRecords;
