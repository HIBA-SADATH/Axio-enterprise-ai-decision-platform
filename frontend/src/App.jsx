
import { BrowserRouter, Routes, Route, Navigate, NavLink ,useLocation} from "react-router-dom";
 
import Dashboard from "./pages/Dashboard";
import CustomerDashboard from "./pages/CustomerDashboard";
import SalesDashboard from "./pages/SalesDashboard";
import RiskDashboard from "./pages/RiskDashboard";
import DecisionDashboard from "./pages/DecisionDashboard";
 
import "./App.css";
 
const NAV_ITEMS = [
  { to: "/", label: "Overview", icon: "▦", end: true },
  { to: "/customers", label: "Customers", icon: "☺" },
  { to: "/sales", label: "Sales", icon: "↗" },
  { to: "/risk", label: "Risk", icon: "⚠" },
  { to: "/decisions", label: "Decisions", icon: "◈" },
];
 
function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-icon">✦</div>
        <div>
          <h2>AXIO</h2>
          <span>Business intelligence</span>
        </div>
      </div>
 
      <p className="menu-heading">Modules</p>
 
      <nav className="sidebar-nav" aria-label="Main navigation">
        {NAV_ITEMS.map(({ to, label, icon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) => `nav-item${isActive ? " active" : ""}`}
          >
            <span className="nav-icon" aria-hidden="true">
              {icon}
            </span>
            {label}
          </NavLink>
        ))}
      </nav>
 
      <div className="sidebar-footer">
        <span className="status-dot" />
        All systems operational
      </div>
    </aside>
  );
}

function AppLayout() {
  const location = useLocation();
  const isDecisionPage = location.pathname.startsWith("/decisions");

  return (
    <div className="app-layout">
      {isDecisionPage ? (
        <aside className="sidebar">
          <div className="brand">
            <div className="brand-icon">✦</div>
            <div>
              <h2>AXIO</h2>
              <span></span>
            </div>
          </div>

          <p className="menu-heading">AI Intelligence</p>

          <nav className="sidebar-nav" aria-label="Decision Intelligence navigation">
            <NavLink to="/" end className="nav-item">
              ← Main Dashboard
            </NavLink>

            <NavLink to="/decisions" end className="nav-item">
              Overview
            </NavLink>

            <a className="nav-item" href="/decisions#copilot">
              AI Executive Copilot
            </a>
            <a className="nav-item" href="/decisions#rag">
              RAG Document Assistant
            </a>
            <a className="nav-item" href="/decisions#what-if">
              What-If Simulator
            </a>
            <a className="nav-item" href="/decisions#recommendations">
              Decision Recommendations
            </a>
            <a className="nav-item" href="/decisions#optimizer">
              Discount Scenario Optimizer
            </a>
          </nav>

          <div className="sidebar-footer">
            <span className="status-dot" />
            AI services workspace
          </div>
        </aside>
      ) : (
        <Sidebar />
      )}

      <main className="main-content">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/customers" element={<CustomerDashboard />} />
          <Route path="/sales" element={<SalesDashboard />} />
          <Route path="/risk" element={<RiskDashboard />} />
          <Route path="/decisions" element={<DecisionDashboard />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}




 
function App() {
  return (
    <BrowserRouter>
      <AppLayout />
    </BrowserRouter>
  );
}
 
export default App;