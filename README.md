# DRL Cache Research Platform

An end-to-end research environment for experimenting with deep-reinforcement-learning-driven caching policies. The repository bundles a FastAPI backend for request simulation and telemetry collection, a PyTorch DQN agent implementation, and a Vite/React dashboard for visualising cache behaviour in real time.

## Repository Layout

```
backend/             # FastAPI service, SQLAlchemy models, simulation endpoints
frontend/cache-frontend/
  src/               # React dashboard for launching runs & viewing stats
rl_agent/            # DQN agent, training scripts, replay buffer utilities
simulator/           # Helpers for generating synthetic request traces
scripts/             # Convenience scripts for managing experiments
```

## Features

- **Policy experimentation pipeline** – switch between TTL, DRL, and hybrid cache policies while recording hit ratio, latency, and staleness metrics.
- **Model lifecycle tooling** – train agents offline, register models, and hot-load weights directly into the service.
- **Interactive dashboard** – launch simulations, stream KPIs, and inspect historical outcomes via live charts.

## Prerequisites

- Python 3.11+
- Node.js 18+ (for the Vite dashboard)
- PostgreSQL 14+ (or any Postgres-compatible database)

## Backend Setup

1. **Create a virtual environment and install dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Provision a PostgreSQL database**

   ```sql
   CREATE DATABASE drl_cache;
   CREATE USER drl_user WITH PASSWORD 'your-password';
   GRANT ALL PRIVILEGES ON DATABASE drl_cache TO drl_user;
   ```

3. **Configure environment variables**

   Create `backend/.env` with at least:

   ```env
   DATABASE_URL=postgresql://drl_user:your-password@localhost:5432/drl_cache
   # Optional overrides
   MODEL_PATH=../rl_agent/model.pt
   SIM_DIR=../simulator
   FRONTEND_ORIGIN=http://localhost:5173
   ```

4. **Run the API server**

   ```bash
   uvicorn backend.main:app --reload
   ```

   By default the service listens on `http://localhost:8000` and exposes interactive docs at `/docs`.

## Frontend Setup

1. Install dependencies (first run only):

   ```bash
   cd frontend/cache-frontend
   npm install
   ```

2. Start the development server:

   ```bash
   npm run dev
   ```

   Vite serves the dashboard at `http://localhost:5173` and proxies API requests to the backend.

## Training and Managing Agents

- **Offline training** – launch a DQN training run against the synthetic environment:

  ```bash
  python -m rl_agent.train --episodes 500 --out rl_agent/model.pt
  ```

  The resulting weights can be referenced by `MODEL_PATH` for live simulations.

- **Evaluating / comparing runs** – additional utilities live under `rl_agent/` (e.g., `evaluate.py`, `compare_results.py`) and `simulator/` for building trace-driven workloads.

## Helpful Scripts

- `scripts/` contains ad-hoc helpers for orchestrating simulations or seeding experiments.
- Use `simulator/make_csv.py` to synthesise workloads that mimic production distributions.

## Contributing

Contributions that improve policy benchmarks, expand visualisations, or tighten operational tooling are welcome. Please open an issue with ideas or bug reports before submitting a pull request.

## License

This repository is provided for research and experimentation purposes. 

