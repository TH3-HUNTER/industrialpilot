import json, uuid, asyncio, csv, io
from datetime import datetime
from typing import Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from backend.agent.agent import IndustrialPilotAgent
from backend.tools.sensor_simulator import generate_alert, get_sensor_list
from backend.db.database import get_all_incidents, get_incident_detail, get_stats, save_operator_response

router = APIRouter()
agent  = IndustrialPilotAgent()
active_connections: Set[WebSocket] = set()

async def broadcast(data: dict):
    dead = set()
    for ws in active_connections:
        try: await ws.send_text(json.dumps(data, default=str))
        except: dead.add(ws)
    active_connections.difference_update(dead)

@router.websocket("/ws")
async def ws_ep(websocket: WebSocket):
    await websocket.accept()
    active_connections.add(websocket)
    try:
        while True: await websocket.receive_text()
    except WebSocketDisconnect:
        active_connections.discard(websocket)

async def _run(alert):
    from backend.tools.sensor_state import mark_alert_active, clear_active_alert
    mark_alert_active(alert['sensor_id'], alert['alert_id'])
    try:
        await agent.process_alert(alert, websocket_callback=broadcast)
    finally:
        clear_active_alert(alert['sensor_id'])

@router.post("/api/alerts/simulate")
async def simulate(bg: BackgroundTasks, sensor_id: str=None, severity: str=None, alert_type: str=None):
    alert = generate_alert(sensor_id=sensor_id, alert_type_override=alert_type, severity_override=severity)
    bg.add_task(_run, alert)
    return {"message":"Alert fired","alert":alert}

@router.post("/api/alerts/ingest")
async def ingest(payload: dict, bg: BackgroundTasks):
    missing=[f for f in ["sensor_id","alert_type","severity"] if f not in payload]
    if missing: return JSONResponse(status_code=400,content={"error":f"Missing: {missing}"})
    payload.setdefault("alert_id",f"ALT-{uuid.uuid4().hex[:8].upper()}")
    payload.setdefault("timestamp",datetime.utcnow().isoformat())
    bg.add_task(_run, payload)
    return {"message":"Alert accepted","alert_id":payload["alert_id"]}

@router.get("/api/incidents")
async def list_inc(limit: int=100): return {"incidents":get_all_incidents(limit)}

@router.get("/api/incidents/{alert_id}")
async def get_inc(alert_id: str): return get_incident_detail(alert_id)

@router.post("/api/incidents/{alert_id}/operator-response")
async def op_resp(alert_id: str, body: dict):
    save_operator_response(alert_id, body.get("decision","acknowledged"), body.get("notes",""))
    from backend.tools.sensor_state import clear_active_alert, clear_pending_human
    inc = get_incident_detail(alert_id).get("incident", {})
    if inc.get("sensor_id"):
        clear_active_alert(inc["sensor_id"])
        # Operator has now handled this — release the suppression so future genuine
        # faults on this sensor can be detected again.
        clear_pending_human(inc["sensor_id"])
    await broadcast({"type":"operator_response","alert_id":alert_id,"message":f"Operator: {body.get('decision')} on {alert_id}"})
    return {"message":"Recorded"}

@router.get("/api/stats")
async def stats(): return get_stats()

@router.get("/api/export/csv")
async def export_csv():
    """Download every incident as a clean, organized CSV — grouped logically:
    timestamp/identity → what happened → what the agent decided → what it did → outcome.
    Excludes the raw markdown report (too noisy for a spreadsheet); the dashboard already
    shows the full report per-incident for anyone who needs that level of detail."""
    from backend.db.database import get_export_data
    rows = get_export_data()

    buf = io.StringIO()
    # Use \r\n line endings (CSV standard) and quote everything that might contain commas
    writer = csv.writer(buf, quoting=csv.QUOTE_MINIMAL)

    def clean(val) -> str:
        """Flatten any embedded newlines/carriage returns into a single space so every
        incident stays on exactly one physical line — raw text stays readable even
        without a CSV-aware viewer, and the model's multi-line reasoning/report text
        never breaks a row in two."""
        s = "" if val is None else str(val)
        return " ".join(s.split())  # .split() with no args splits on any run of
        # whitespace (including \n, \r, \t, multiple blank lines) and drops empties,
        # so this also collapses accidental double-spacing from blank lines

    # Human-readable headers, grouped in a logical reading order
    writer.writerow([
        "Date", "Time",
        "Alert ID", "Sensor", "Location",
        "Alert Type", "Severity",
        "Reading", "Threshold", "Unit",
        "Root Cause",
        "AI Confidence (%)", "Status",
        "Actions Taken",
        "Operator Decision", "Operator Notes",
    ])

    for r in rows:
        created = (r.get("created_at") or "")
        date_part, time_part = (created.split("T") + [""])[:2] if "T" in created else (created, "")
        time_part = time_part[:8]  # HH:MM:SS, drop microseconds

        writer.writerow([
            clean(date_part),
            clean(time_part),
            clean(r.get("alert_id", "")),
            clean(r.get("sensor_id", "")),
            clean(r.get("location", "")),
            clean(r.get("alert_type", "")),
            clean((r.get("severity") or "").upper()),
            clean(r.get("reading_value", "")),
            clean(r.get("threshold", "")),
            clean(r.get("reading_unit", "")),
            clean(r.get("root_cause", "")),
            clean(r.get("confidence_pct", "")),
            clean((r.get("status") or "").replace("_", " ").upper()),
            clean(r.get("actions_taken", "")),
            clean((r.get("operator_decision") or "").replace("_", " ").upper()),
            clean(r.get("operator_notes", "")),
        ])

    buf.seek(0)
    filename = f"industrialpilot_incidents_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@router.get("/api/sensors")
async def sensors(): return {"sensors":get_sensor_list()}

@router.get("/api/health")
async def health(): return {"status":"ok","agent":"IndustrialPilot v1.0"}

@router.get("/api/deployment-info")
async def deployment_info():
    """
    Proof of Alibaba Cloud deployment (hackathon submission requirement).
    Returns the live Alibaba Cloud ECS instance this backend is running on.
    See backend/deployment/alibaba_cloud_config.py for the source record.
    """
    from backend.deployment.alibaba_cloud_config import get_deployment_info
    return get_deployment_info()

@router.post("/api/clear")
async def clear_data():
    from backend.db.database import clear_all_data
    from backend.tools.sensor_state import LIVE_STATE, ACTIVE_ALERT, PENDING_HUMAN_ACTION
    clear_all_data()
    for sid in ACTIVE_ALERT: ACTIVE_ALERT[sid] = None
    for sid in PENDING_HUMAN_ACTION: PENDING_HUMAN_ACTION[sid] = False
    await broadcast({"type": "data_cleared", "message": "All incident data cleared"})
    return {"message": "All data cleared"}

# ── Sensor state (server is single source of truth) ─────────────────────
from backend.tools.sensor_state import (
    get_all_states, set_metric, get_state, THRESHOLDS, METRIC_CONFIG,
    detect_alert_from_state, set_mode, get_mode, get_all_modes,
    scan_all_sensors, has_active_alert,
)

@router.get("/api/sensor-state")
async def get_sensor_state():
    from backend.tools.sensor_state import get_warn_thresholds
    return {"state": get_all_states(), "thresholds": THRESHOLDS,
            "warn_thresholds": get_warn_thresholds(),
            "config": METRIC_CONFIG, "modes": get_all_modes()}

@router.post("/api/sensor-state/{sensor_id}/{metric}")
async def update_sensor_metric(sensor_id: str, metric: str, body: dict):
    value = float(body.get("value", 0))
    set_metric(sensor_id, metric, value)
    await broadcast({"type": "sensor_update", "sensor_id": sensor_id,
                     "metric": metric, "value": value})
    return {"sensor_id": sensor_id, "metric": metric, "value": value}

@router.post("/api/sensor-mode/{sensor_id}")
async def update_sensor_mode(sensor_id: str, body: dict):
    """Switch a single sensor between auto (server ticks it) and manual (only slider/API changes it)."""
    mode = body.get("mode", "auto")
    set_mode(sensor_id, mode)
    await broadcast({"type": "mode_update", "sensor_id": sensor_id, "mode": mode})
    return {"sensor_id": sensor_id, "mode": mode}

@router.post("/api/alerts/from-state/{sensor_id}")
async def alert_from_state(sensor_id: str, background_tasks: BackgroundTasks):
    if has_active_alert(sensor_id):
        return JSONResponse(status_code=400, content={"error": "Alert already being handled for this sensor."})
    alert = detect_alert_from_state(sensor_id)
    if not alert:
        return JSONResponse(status_code=400, content={
            "error": "All readings within normal range.",
            "current": get_state(sensor_id), "thresholds": THRESHOLDS.get(sensor_id, {})
        })
    background_tasks.add_task(_run, alert)
    return {"message": "Alert fired from live state", "alert": alert}

@router.post("/api/scan")
async def scan_now(background_tasks: BackgroundTasks):
    """Full scan: finds violations NOT already active, fires them. Does NOT tick (ticking is continuous now)."""
    alerts = scan_all_sensors()
    for a in alerts:
        background_tasks.add_task(_run, a)
    return {"alerts_found": len(alerts), "alerts": alerts, "state": get_all_states()}

@router.post("/api/tick")
async def tick_now():
    """Lightweight: advance auto-mode sensor readings by one small step. Called frequently by frontend for smooth motion."""
    from backend.tools.sensor_state import tick_auto_sensors
    tick_auto_sensors()
    return {"state": get_all_states()}

# ── Settings — live-editable notification config + thresholds ───────────
@router.get("/api/settings")
async def get_settings():
    from backend.tools.tools import NOTIFY_CONFIG
    from backend.agent.agent import CONFIDENCE_THRESHOLD
    return {
        "operator_emails": NOTIFY_CONFIG.get("operator_emails", []),
        "slack_webhook_url": NOTIFY_CONFIG.get("slack_webhook_url", ""),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "scan_interval_seconds": 25,  # informational — actual interval lives client-side
    }

@router.post("/api/settings")
async def update_settings(body: dict):
    from backend.tools.tools import NOTIFY_CONFIG
    if "operator_emails" in body:
        raw = body["operator_emails"]
        emails = [e.strip() for e in raw.split(",")] if isinstance(raw, str) else list(raw)
        NOTIFY_CONFIG["operator_emails"] = [e for e in emails if e]
    if "slack_webhook_url" in body:
        NOTIFY_CONFIG["slack_webhook_url"] = body["slack_webhook_url"].strip()
    await broadcast({"type": "settings_updated", "message": "Notification settings updated"})
    return {"message": "Settings updated", "operator_emails": NOTIFY_CONFIG["operator_emails"]}
