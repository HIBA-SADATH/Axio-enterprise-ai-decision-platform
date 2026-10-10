import { useNavigate } from "react-router-dom";

const modules = [
  {
    title: "Customer Intelligence",
    description: "Understand customer behavior and predict churn.",
    icon: "◎",
    color: "blue",
    path: "/customers",
    tag: "CUSTOMER ANALYTICS",
  },
  {
    title: "Sales & Demand",
    description: "Forecast sales and identify demand trends.",
    icon: "▥",
    color: "green",
    path: "/sales",
    tag: "PREDICTIVE ANALYTICS",
  },
  {
    title: "Risk & Anomaly",
    description: "Detect unusual transactions and assess business risk.",
    icon: "◇",
    color: "orange",
    path: "/risk",
    tag: "RISK MONITORING",
  },
  {
    title: "AI Decision Intelligence",
    description: "Turn business insights into actionable decisions.",
    icon: "✧",
    color: "purple",
    path: "/decisions",
    tag: "AI-POWERED INSIGHTS",
  },
];

function Dashboard() {
  const navigate = useNavigate();

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <div>
          <p className="eyebrow">ENTERPRISE INTELLIGENCE / OVERVIEW</p>
          <h1>Business Overview</h1>
          <p className="subtitle">
            Your unified workspace for smarter, data-driven decisions.
          </p>
        </div>

        <div className="system-status">
          <span className="status-dot"></span>
          Platform Online
        </div>
      </header>

      <section className="welcome-banner">
        <div>
          <p className="banner-label">WELCOME TO AXIO</p>
          <h2>Intelligence that drives decisions.</h2>
          <p>
            Explore your business through four connected intelligence modules.
          </p>
        </div>
        <div className="banner-symbol">A</div>
      </section>

      <section className="section-heading">
        <div>
          <h2>Intelligence Modules</h2>
          <p>Choose a module to explore its dashboard.</p>
        </div>
        <span className="module-count">04 MODULES</span>
      </section>

      <section className="module-grid">
        {modules.map((module) => (
          <button
            className="module-card"
            key={module.path}
            onClick={() => navigate(module.path)}
          >
            <div className="module-card-top">
              <div className={`module-icon ${module.color}`}>
                {module.icon}
              </div>
              <span className="card-arrow">↗</span>
            </div>

            <p className="module-tag">{module.tag}</p>
            <h3>{module.title}</h3>
            <p className="module-description">{module.description}</p>

            <span className="module-link">Open dashboard <span>→</span></span>
          </button>
        ))}
      </section>

      <footer className="dashboard-footer">
        AXIO <span>•</span> ENTERPRISE AI DECISION INTELLIGENCE PLATFORM
      </footer>
    </div>
  );
}

export default Dashboard;