# ⚡ IndustrialPilot — Autonomous Factory Incident Response Agent

> **Hackathon Entry — Global AI Hackathon Series with Qwen Cloud | Track 4: Autopilot Agent**

IndustrialPilot is an autonomous AI agent that monitors industrial sensor alerts across 8 factory machines, diagnoses root causes using **Qwen AI**, and either takes a real corrective action through simulated industrial control protocols (VFD speed commands, BMS firing-rate control, pressure relief valves, etc.) or escalates to a human technician when it cannot safely fix the problem itself.

Every action the agent takes is physically reflected on the connected sensor — fixing a problem actually moves the readings back into a safe range, the same way a real SCADA/PLC-integrated system would behave.

---

## 🏭 What It Monitors

| Equipment | Sensors | Example Faults |
|---|---|---|
| 3× Motors | Temperature, Vibration, Current | Overcurrent, Overtemperature, Bearing Fault, Insulation Fault |
| 2× Pumps | Temperature, Pressure, Flow Rate | High Pressure, Low Flow Rate, Cavitation |
| 1× Compressor | Temperature, Pressure, Vibration | High Pressure, Oil Pressure Low |
| 1× Conveyor | Temperature, Speed, Current | Underspeed, Belt Slip |
| 1× Boiler | Temperature, Pressure, Flow Rate | Overpressure, Low Water Flow, Flame Failure |

---

## 🧠 How the Agent Decides

1. **Diagnose** — calls `get_sensor_history` to read live values before acting
2. **Score confidence** (0.0–1.0) in its own diagnosis
3. **Act, only within strict permission boundaries:**
   - ✅ **Electronically fixable + confidence ≥ 80%** → executes a real remediation tool (reduce motor load, open a pressure relief valve, reduce boiler firing rate, etc.)
   - 🔧 **Mechanical fault** (vibration, bearing, belt) → **always** creates a maintenance work order — a bearing cannot be fixed by a network command, regardless of how confident the AI is
   - ⚡ **Electrical fault** (insulation) → **always** escalates to a human, never acts remotely
   - 🚨 **Imminent hazard** (flame failure, severe vibration) → **emergency shutdown**: cuts power to the unit and mandatorily notifies a technician by email/Slack
   - 🚫 **Confidence < 80%** → escalates to a human operator, even if a remediation tool was called — an unverified fix is never marked "resolved"
4. **Generates a full incident report** and logs everything to a permanent audit trail

This permission boundary — what the agent can and cannot do alone — is enforced **in code**, not just in the prompt, so it can't be talked out of it.

---

## 🖥️ Dashboard — 5 Tabs

| Tab | What it shows |
|---|---|
| **🏭 Live Dashboard** | Fire alerts manually or let the agent auto-scan every 25s; live sensor sliders (Manual mode) or simulated live readings (Auto mode); real-time agent reasoning log; full incident detail panel with human approve/reject controls |
| **📡 Sensor Status** | All 8 machines with live readings, color-coded against real thresholds |
| **📋 All Incidents** | Full filterable history + **CSV export** of every incident with root cause, confidence, actions taken, and operator decisions |
| **🔩 Equipment Reference** | Full specs, normal operating ranges, and maintenance schedule for every machine |
| **⚙️ Settings** | Live-editable technician notification emails (no restart needed), Slack webhook, confidence threshold reference |

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env and add your Qwen API key

# 3. Run
python main.py

# 4. Open the dashboard
http://localhost:8000
```

---

## ⚙️ Environment Variables

| Variable | Description | Required? |
|---|---|---|
| `QWEN_API_KEY` | Your Qwen Cloud API key | Yes |
| `QWEN_BASE_URL` | Qwen Cloud endpoint | Yes (default provided) |
| `QWEN_MODEL` | Model to use, e.g. `qwen-flash` or `qwen-plus` | Yes |
| `CONFIDENCE_THRESHOLD` | Auto-remediation cutoff (default `0.80`) | No |
| `OPERATOR_EMAIL` | Comma-separated technician emails for escalations | No (also editable live in Settings tab) |
| `SENDGRID_API_KEY` | Enables real email sending via SendGrid | No — without it, escalations are logged to console instead |
| `SLACK_WEBHOOK_URL` | Enables Slack notifications | No |

---

## 📡 Key API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/alerts/simulate` | Trigger a simulated sensor alert |
| `POST` | `/api/alerts/from-state/{sensor_id}` | Fire an alert based on current live sensor readings |
| `POST` | `/api/scan` | Server-side scan of all sensors for threshold violations |
| `GET` | `/api/incidents` | List all incidents |
| `GET` | `/api/incidents/{alert_id}` | Full incident detail, decisions, actions, report |
| `POST` | `/api/incidents/{alert_id}/operator-response` | Human operator approves/rejects/manually fixes |
| `GET` | `/api/export/csv` | Download full incident history as CSV |
| `GET` / `POST` | `/api/settings` | Read/update notification config live |
| `GET` | `/api/sensor-state` | Live readings, thresholds, and modes for all sensors |
| `WS` | `/ws` | Real-time event stream powering the dashboard |

---

## 🏗 Architecture

```
Sensor Alert (simulated or live state)
        ↓
  FastAPI Ingestion Layer
        ↓
  Qwen Agent Reasoning Loop
  ├─ get_sensor_history()
  ├─ Confidence Assessment (0.0–1.0)
  ├─ Electronically fixable + confidence ≥80% → Remediation tools
  │    (reduce_motor_load, activate_motor_cooling, open_pressure_bypass_valve,
  │     increase_pump_speed, engage_compressor_unloader, adjust_conveyor_tension,
  │     reduce_boiler_firing_rate, open_boiler_feedwater_valve, ...)
  ├─ Mechanical fault → create_maintenance_work_order (mandatory)
  ├─ Electrical fault / low confidence → escalate_to_operator
  └─ Imminent hazard → emergency_shutdown + escalate_to_operator
        ↓
  Sensor state physically updated (server-authoritative, single source of truth)
        ↓
  SQLite Audit Log (incidents, decisions, remediations, reports, operator responses)
        ↓
  WebSocket → Live Dashboard
```

---

## ☁️ Alibaba Cloud Deployment

```bash
# On your Alibaba Cloud ECS instance
git clone https://github.com/TH3-HUNTER/industrialpilot.git
cd industrialpilot
pip install -r requirements.txt
cp .env.example .env
nano .env   # add your real Qwen API key
python main.py
```

The app will be reachable at `http://<your-ECS-public-IP>:8000`.

---

## 📄 License

MIT License — see [LICENSE](LICENSE).
