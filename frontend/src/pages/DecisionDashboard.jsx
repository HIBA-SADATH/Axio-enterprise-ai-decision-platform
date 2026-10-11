import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import "./DecisionDashboard.css";

const API = "http://127.0.0.1:8000/api/copilot";

async function apiRequest(path, options = {}) {
  const response = await fetch(`${API}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.error || `Request failed (${response.status})`);
  }

  return data;
}

function formatMoney(value) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  }).format(value ?? 0);
}

function DecisionDashboard() {
  const navigate = useNavigate();
  const [refreshing, setRefreshing] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(""); 
  const [analytics, setAnalytics] = useState(null);
  const [recommendations, setRecommendations] = useState(null);
  const [optimizer, setOptimizer] = useState(null);

  const [copilotQuestion, setCopilotQuestion] = useState("");
  const [copilotAnswer, setCopilotAnswer] = useState("");
  const [copilotLoading, setCopilotLoading] = useState(false);

  const [ragQuestion, setRagQuestion] = useState("");
  const [ragAnswer, setRagAnswer] = useState(null);
  const [ragStatus, setRagStatus] = useState("");
  const [ragLoading, setRagLoading] = useState(false);

  const [discountChange, setDiscountChange] = useState(-10);
  const [scenario, setScenario] = useState(null);
  const [scenarioLoading, setScenarioLoading] = useState(false);

  const [pageError, setPageError] = useState("");

  async function askCopilot(event) {
    event.preventDefault();

    const question = copilotQuestion.trim();

    if (!question) return;

    setCopilotLoading(true);
    setCopilotAnswer("");
    setPageError("");

    try {
      const result = await apiRequest("/ai/", {
        method: "POST",
        body: JSON.stringify({ question }),
      });

      setCopilotAnswer(
        result.answer || "The AI Copilot returned no answer."
      );
    } catch (error) {
      setCopilotAnswer(`Unable to get an answer: ${error.message}`);
    } finally {
      setCopilotLoading(false);
    }
  }


async function loadDashboardData() {
  setRefreshing(true);
  setPageError("");

  try {
    const results = await Promise.allSettled([
      apiRequest("/analytics/"),
      apiRequest("/recommendations/"),
      apiRequest("/optimizer/"),
    ]);

    if (results[0].status === "fulfilled") {
      setAnalytics(results[0].value);
    }

    if (results[1].status === "fulfilled") {
      setRecommendations(results[1].value);
    }

    if (results[2].status === "fulfilled") {
      setOptimizer(results[2].value);
    }

    const errors = results
      .filter((item) => item.status === "rejected")
      .map((item) => item.reason.message);

    if (errors.length) {
      setPageError(errors.join(" | "));
    } else {
      setLastUpdated(new Date().toLocaleTimeString());
    }
  } catch (error) {
    setPageError(error.message);
  } finally {
    setRefreshing(false);
  }
}
useEffect(() => {
  loadDashboardData();
}, []);

  async function buildDocumentIndex() {
    setRagLoading(true);
    setRagStatus("");
    setRagAnswer(null);

    try {
      const result = await apiRequest("/rag/build/", {
        method: "POST",
      });

      setRagStatus(
        `${result.message} Chunks indexed: ${result.chunks_indexed}`
      );
    } catch (error) {
      setRagStatus(`Indexing failed: ${error.message}`);
    } finally {
      setRagLoading(false);
    }
  }

  async function askDocument(event) {
    event.preventDefault();

    if (!ragQuestion.trim()) return;

    setRagLoading(true);
    setRagAnswer(null);

    try {
      const result = await apiRequest("/rag/ask/", {
        method: "POST",
        body: JSON.stringify({
          question: ragQuestion.trim(),
        }),
      });

      setRagAnswer(result);
    } catch (error) {
      setRagAnswer({
        answer: `Unable to answer: ${error.message}`,
        sources: [],
      });
    } finally {
      setRagLoading(false);
    }
  }

  async function runScenario(event) {
    event.preventDefault();

    setScenarioLoading(true);
    setScenario(null);

    try {
      const result = await apiRequest("/what-if/", {
        method: "POST",
        body: JSON.stringify({
          discount_change_percent: Number(discountChange),
        }),
      });

      setScenario(result);
    } catch (error) {
      setPageError(error.message);
    } finally {
      setScenarioLoading(false);
    }
  }

  const categories = analytics?.category_performance || [];
  const highestProfit = recommendations?.highest_profit_category;

  return (
    <div className="decision-page">
      <header className="di-header">
        <div>
          
          <h1>AI Decision Intelligence</h1>
          <p>
            Explore business performance, ask questions, analyse documents,
            simulate scenarios, and compare decision alternatives.
          </p>
        </div>

       
<div className="di-header-actions">

  {lastUpdated && (
  <p className="di-last-updated">
    Last updated: {lastUpdated}
  </p>
)}

  <button
    className="di-button di-secondary"
    onClick={loadDashboardData}
    disabled={refreshing}
  >
    {refreshing ? "Refreshing…" : "↻ Refresh Data"}
  </button>
</div>
 
      </header>

      {pageError && (
        <div className="di-error" role="alert">
          <strong>Some data could not be loaded:</strong> {pageError}
        </div>
      )}

      <section className="di-metrics" id="overview">
        <Metric
          label="Total sales"
          value={analytics ? formatMoney(analytics.total_sales) : "Loading…"}
        />
        <Metric
          label="Total profit"
          value={analytics ? formatMoney(analytics.total_profit) : "Loading…"}
        />
        <Metric
          label="Units sold"
          value={analytics
            ? Number(analytics.total_quantity).toLocaleString("en-US")
            : "Loading…"}
        />
        <Metric
          label="Highest-profit category"
          value={highestProfit?.category || "Loading…"}
        />
      </section>

      <section className="di-panel" id="copilot">
        <div className="di-section-heading">
          <div>
            <span className="di-eyebrow">OLLAMA · NATURAL-LANGUAGE ANALYTICS</span>
            <h2>AI Executive Copilot</h2>
            <p>
              Ask questions about the available sales and profitability data.
            </p>
          </div>
          <span className="di-icon">✳</span>
        </div>

        <form className="di-form" onSubmit={askCopilot}>
          <input
            value={copilotQuestion}
            onChange={(event) => setCopilotQuestion(event.target.value)}
            placeholder="Which category generates the highest profit?"
            aria-label="Business question"
          />
          <button className="di-button di-primary" disabled={copilotLoading}>
            {copilotLoading ? "Thinking…" : "Ask Copilot"}
          </button>
        </form>

        <div className="di-quick-prompts">
          {[
            "Which category generates the highest profit?",
            "Which subcategories are loss-making?",
            "How can the business improve profitability?",
          ].map((prompt) => (
            <button
              key={prompt}
              type="button"
              className="di-chip"
              onClick={() => setCopilotQuestion(prompt)}
            >
              {prompt}
            </button>
          ))}
        </div>

        {copilotAnswer && (
          <div className="di-answer">
            <strong>Copilot response</strong>
            <p>{copilotAnswer}</p>
          </div>
        )}
      </section>

      <section className="di-panel" id="rag">
        <div className="di-section-heading">
          <div>
            <span className="di-eyebrow">RETRIEVAL-AUGMENTED GENERATION</span>
            <h2>RAG Document Assistant</h2>
            <p>
              Retrieve relevant passages from PDF and TXT files in
              the backend documents folder.
            </p>
          </div>
          <span className="di-icon">▤</span>
        </div>

        <button
          type="button"
          className="di-button di-secondary"
          onClick={buildDocumentIndex}
          disabled={ragLoading}
        >
          {ragLoading ? "Processing…" : "Build document index"}
        </button>

        {ragStatus && <p className="di-status">{ragStatus}</p>}

        <form className="di-form di-rag-form" onSubmit={askDocument}>
          <input
            value={ragQuestion}
            onChange={(event) => setRagQuestion(event.target.value)}
            placeholder="Ask a question about your report"
            aria-label="Document question"
          />
          <button className="di-button di-primary" disabled={ragLoading}>
            Ask document
          </button>
        </form>

        {ragAnswer && (
          <div className="di-answer">
            <strong>Document answer</strong>
            <p>{ragAnswer.answer}</p>

            {ragAnswer.sources?.length > 0 && (
              <div className="di-source-list">
                Retrieved source chunks:
                {ragAnswer.sources.map((source) => (
                  <span key={source.chunk}>
                    Chunk {source.chunk} · similarity {source.similarity}
                  </span>
                ))}
              </div>
            )}
          </div>
        )}
      </section>

      <section className="di-panel" id="what-if">
        <div className="di-section-heading">
          <div>
            <span className="di-eyebrow">SCENARIO ANALYSIS</span>
            <h2>What-If Simulator</h2>
            <p>
              Test a relative change in existing discount rates using
              a simplified accounting-based scenario.
            </p>
          </div>
          <span className="di-icon">⌁</span>
        </div>

        <form className="di-scenario-form" onSubmit={runScenario}>
          <div className="di-slider-label">
            <label htmlFor="di-discount-slider">
              Relative discount change
            </label>
            <strong>{Number(discountChange) > 0 ? "+" : ""}{discountChange}%</strong>
          </div>

          <input
            id="di-discount-slider"
            type="range"
            min="-100"
            max="100"
            step="5"
            value={discountChange}
            onChange={(event) => setDiscountChange(event.target.value)}
          />

          <button
            className="di-button di-primary"
            disabled={scenarioLoading}
          >
            {scenarioLoading ? "Calculating…" : "Run scenario"}
          </button>
        </form>

        {scenario && (
          <div className="di-scenario-results">
            <Metric
              label="Baseline sales"
              value={formatMoney(scenario.baseline_sales)}
            />
            <Metric
              label="Scenario sales estimate"
              value={formatMoney(scenario.scenario_sales_estimate)}
            />
            <Metric
              label="Baseline profit"
              value={formatMoney(scenario.baseline_profit)}
            />
            <Metric
              label="Scenario profit estimate"
              value={formatMoney(scenario.scenario_profit_estimate)}
            />
            <Metric
              label="Estimated profit change"
              value={formatMoney(scenario.estimated_profit_change)}
            />
            <p className="di-disclaimer">{scenario.assumption}</p>
          </div>
        )}

        <p className="di-disclaimer">
          This model assumes pre-discount sales, quantities, and estimated
          costs remain fixed. It does not predict changes in customer demand.
        </p>
      </section>

      <section className="di-panel" id="recommendation">
        <div className="di-section-heading">
          <div>
            <span className="di-eyebrow">DATA-DRIVEN INSIGHTS</span>
            <h2>Decision Recommendations</h2>
            <p>
              Rule-based suggestions derived from historical sales,
              profit, discounts, and product subcategories.
            </p>
          </div>
          <span className="di-icon">◈</span>
        </div>

        {highestProfit && (
          <div className="di-leader-card">
            <span>Highest-profit category</span>
            <h3>{highestProfit.category}</h3>
            <p>
              Profit: {formatMoney(highestProfit.profit)}
              {" · "}
              Sales: {formatMoney(highestProfit.sales)}
            </p>
          </div>
        )}

        {recommendations?.recommendations?.length ? (
          <div className="di-recommendations">
            {recommendations.recommendations.map((item, index) => (
              <article
                className="di-recommendation"
                key={`${item.type}-${item.focus}-${index}`}
              >
                <div>
                  <span className="di-recommendation-type">{item.type}</span>
                  <h3>{item.focus}</h3>
                </div>

                <p>
                  {item.observation ||
                    (item.profit !== undefined
                      ? `Historical profit: ${formatMoney(item.profit)}.`
                      : `Historical sales: ${formatMoney(item.sales)}; profit margin: ${item.profit_margin_percent}%.`)}
                </p>

                <p>{item.recommendation}</p>
              </article>
            ))}
          </div>
        ) : (
          <p className="di-muted">Recommendations are not available yet.</p>
        )}
      </section>

      <section className="di-panel" id="optimizer">
        <div className="di-section-heading">
          <div>
            <span className="di-eyebrow">SCENARIO-BASED OPTIMIZATION</span>
            <h2>Discount Scenario Optimizer</h2>
            <p>
              Compare tested discount adjustments and rank subcategories
              by estimated profit improvement.
            </p>
          </div>
          <button
            className="di-button di-secondary"
            onClick={loadDashboardData}
          >
            Refresh results
          </button>
        </div>

        {optimizer?.ranking?.length ? (
          <>
            <div className="di-table-wrap">
              <table className="di-table">
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Subcategory</th>
                    <th>Baseline profit</th>
                    <th>Discount adjustment</th>
                    <th>Estimated profit</th>
                    <th>Profit improvement</th>
                  </tr>
                </thead>

                <tbody>
                  {optimizer.ranking.slice(0, 10).map((item, index) => (
                    <tr key={item.subcategory}>
                      <td>{index + 1}</td>
                      <td><strong>{item.subcategory}</strong></td>
                      <td>{formatMoney(item.baseline_profit)}</td>
                      <td>
                        {item.relative_discount_change_percent > 0 ? "+" : ""}
                        {item.relative_discount_change_percent}%
                      </td>
                      <td>{formatMoney(item.scenario_profit_estimate)}</td>
                      <td>{formatMoney(item.estimated_profit_improvement)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="di-disclaimer">
              {optimizer.assumptions?.join(" ")}
            </div>
          </>
        ) : (
          <p className="di-muted">
            Optimizer results are loading or unavailable.
          </p>
        )}
      </section>

      <section className="di-panel">
        <div className="di-section-heading">
          <div>
            <span className="di-eyebrow">SALES BREAKDOWN</span>
            <h2>Top Subcategories by Sales</h2>
          </div>
        </div>

        {analytics?.top_subcategories_by_sales?.length ? (
          <div className="di-table-wrap">
            <table className="di-table">
              <thead>
                <tr>
                  <th>Subcategory</th>
                  <th>Sales</th>
                  <th>Profit</th>
                </tr>
              </thead>
              <tbody>
                {analytics.top_subcategories_by_sales.map((item) => (
                  <tr key={item["Sub-Category"]}>
                    <td>{item["Sub-Category"]}</td>
                    <td>{formatMoney(item.sales)}</td>
                    <td>{formatMoney(item.profit)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="di-muted">Sales breakdown is loading or unavailable.</p>
        )}
      </section>

      <footer className="di-footer">
        AI Decision Intelligence · React + Django REST Framework + Ollama
      </footer>
    </div>
  );
}

function Metric({ label, value }) {
  return (
    <div className="di-metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export default DecisionDashboard;
