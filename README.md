# Axio — Enterprise AI Decision Intelligence Platform

## Overview
Axio is a unified AI-powered platform that helps businesses analyze customer behavior, forecast sales, detect risks and anomalies, and support data-driven decisions.

## Modules
- **Customer Intelligence:** Customer behavior analysis and churn prediction.
- **Sales & Demand Intelligence:** Sales forecasting and demand analysis.
- **Risk & Anomaly Intelligence:** Business risk scoring and anomaly detection.
- **AI Decision Intelligence:** AI-powered insights, recommendations, and decision support.

## Technology Stack
- **Frontend:** React, JavaScript, HTML, CSS
- **Backend:** Python, Django, REST APIs
- **Data & AI:** Pandas, NumPy, Scikit-learn, and other relevant machine learning tools
- **Version Control:** Git and GitHub

## Project Structure
- `backend/` — Backend applications and REST APIs (Django)
- `frontend/` — User interface (React + Vite)
- `backend/data/` — Datasets and risk scoring data

## Getting Started: Running the Servers

To run the complete platform locally, you will need to run the **Backend server** (Django) and the **Frontend server** (React/Vite) in two separate terminal windows.

---

### Prerequisites
- **Python**: Version 3.10+ installed ([python.org](https://www.python.org/))
- **Node.js**: Version 18+ and npm installed ([nodejs.org](https://nodejs.org/))
- *(Optional)* **Ollama**: If you want to use the local LLM AI Copilot (`llama3.2`) ([ollama.com](https://ollama.com/))

---

### 1. Running the Backend Server (Django)

Open a terminal and navigate to the `backend/` directory:

```bash
cd backend
```

#### Step 1: Create a Virtual Environment

- **Windows (PowerShell / Command Prompt):**
  ```powershell
  python -m venv venv
  # or if using the Python Launcher:
  py -m venv venv
  ```
- **macOS / Linux:**
  ```bash
  python3 -m venv venv
  ```

#### Step 2: Activate the Virtual Environment

- **Windows (PowerShell):**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
  *(Note: If you encounter an execution policy error in PowerShell, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first)*

- **Windows (Command Prompt):**
  ```cmd
  venv\Scripts\activate.bat
  ```

- **macOS / Linux:**
  ```bash
  source venv/bin/activate
  ```

#### Step 3: Install Backend Dependencies

```bash
pip install -r requirements.txt
```

#### Step 4: Run Database Migrations

```bash
python manage.py migrate
```

#### Step 5: Start the Django Server

```bash
python manage.py runserver
```

The backend server will start at:  
👉 **`http://127.0.0.1:8000/`**

#### Backend API Verification Endpoints:
- **Base Health Check:** `http://127.0.0.1:8000/`
- **Customer Intelligence:** `http://127.0.0.1:8000/api/customers/summary/`
- **Sales Intelligence:** `http://127.0.0.1:8000/api/sales/summary/`
- **Risk Intelligence:** `http://127.0.0.1:8000/api/risk/summary/`
- **AI Decision Analytics:** `http://127.0.0.1:8000/api/copilot/analytics/`

---

### 2. Running the Frontend Server (React + Vite)

Open a **new** terminal window and navigate to the `frontend/` directory:

```bash
cd frontend
```

#### Step 1: Install Dependencies

```bash
npm install
```

#### Step 2: Start the Development Server

```bash
npm run dev
```

The frontend application will start at:  
👉 **`http://localhost:5173/`**

Open `http://localhost:5173/` in your browser to access the Axio dashboard.

---

### 3. Optional: Running AI Copilot & RAG (Ollama)

For the AI Executive Copilot and RAG Document Assistant in the Decisions module:

1. Install and start [Ollama](https://ollama.com/).
2. Pull the required model:
   ```bash
   ollama pull llama3.2
   ```
3. Ensure the Ollama service is active (`ollama serve`).

---

## Quick Reference Summary

| Service | Directory | Command | Default URL |
| :--- | :--- | :--- | :--- |
| **Backend** (Django) | `backend/` | `python manage.py runserver` | `http://127.0.0.1:8000/` |
| **Frontend** (Vite) | `frontend/` | `npm run dev` | `http://localhost:5173/` |
| **AI LLM** (Ollama, Optional) | System | `ollama serve` | `http://localhost:11434/` |

---

## Team Project
This project is developed collaboratively by four members, with each member contributing a separate intelligence module.

## Status
Development and integration in progress.