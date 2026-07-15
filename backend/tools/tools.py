import os, uuid, httpx
from datetime import datetime
from dotenv import load_dotenv
from backend.db.database import log_remediation
from backend.tools.sensor_state import apply_fix, get_state

load_dotenv()

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
SENDGRID_API_KEY  = os.getenv("SENDGRID_API_KEY", "")
FROM_EMAIL        = os.getenv("FROM_EMAIL", "pilot@factory.com")

# Live, in-memory notification config — editable from the Settings tab without restarting.
# Starts from .env defaults, can be overridden at runtime via /api/settings.
NOTIFY_CONFIG = {
    "operator_emails": [
        e.strip() for e in os.getenv("OPERATOR_EMAIL", "manaihamza2003@gmail.com").split(",") if e.strip()
    ],
    "slack_webhook_url": SLACK_WEBHOOK_URL,
}

TOOL_DEFINITIONS = [
    # ── DIAGNOSTIC ──────────────────────────────────────────────────────
    {"type":"function","function":{
        "name":"get_sensor_history",
        "description":"Read recent sensor history and current live values. Always call this first before any action.",
        "parameters":{"type":"object","properties":{
            "sensor_id":{"type":"string","description":"Sensor ID e.g. MOTOR-A1"},
            "hours":{"type":"integer","description":"Hours of history to retrieve","default":6}
        },"required":["sensor_id"]}
    }},
    # ── MOTOR CONTROLS ───────────────────────────────────────────────────
    {"type":"function","function":{
        "name":"reduce_motor_load",
        "description":"Send speed reduction command to Variable Frequency Drive (VFD). Reduces motor RPM which lowers current draw and heat generation. Safe and reversible. Use for OVERCURRENT or OVERTEMPERATURE on motors.",
        "parameters":{"type":"object","properties":{
            "motor_id":{"type":"string"},
            "reduce_by_percent":{"type":"number","description":"Speed reduction 10-40%. Above 40% risks production stoppage."},
            "alert_id":{"type":"string"}
        },"required":["motor_id","reduce_by_percent","alert_id"]}
    }},
    {"type":"function","function":{
        "name":"activate_motor_cooling",
        "description":"Enable auxiliary forced-air cooling fan on motor housing. Standard response for thermal alerts. Reduces winding temperature by 8-15°C over 10 minutes.",
        "parameters":{"type":"object","properties":{
            "motor_id":{"type":"string"},
            "duration_minutes":{"type":"integer","description":"How long to run cooling fan. Typical: 15-30 min."},
            "alert_id":{"type":"string"}
        },"required":["motor_id","duration_minutes","alert_id"]}
    }},
    {"type":"function","function":{
        "name":"clear_motor_fault_and_restart",
        "description":"Send fault-clear signal then controlled restart command via PLC. Only use AFTER root cause is addressed (e.g. after load reduction). Do NOT use as first action.",
        "parameters":{"type":"object","properties":{
            "motor_id":{"type":"string"},
            "alert_id":{"type":"string"}
        },"required":["motor_id","alert_id"]}
    }},
    # ── PUMP CONTROLS ────────────────────────────────────────────────────
    {"type":"function","function":{
        "name":"open_pressure_bypass_valve",
        "description":"Open bypass valve to relieve excess pressure in pump or hydraulic circuit. Direct pressure reduction. Use for HIGH_PRESSURE or OVERPRESSURE alerts.",
        "parameters":{"type":"object","properties":{
            "unit_id":{"type":"string"},
            "alert_id":{"type":"string"}
        },"required":["unit_id","alert_id"]}
    }},
    {"type":"function","function":{
        "name":"increase_pump_speed",
        "description":"Increase pump VFD frequency to restore flow rate. Use for LOW_FLOW_RATE or CAVITATION alerts where flow is insufficient.",
        "parameters":{"type":"object","properties":{
            "unit_id":{"type":"string"},
            "increase_by_percent":{"type":"number","description":"Speed increase 10-30%"},
            "alert_id":{"type":"string"}
        },"required":["unit_id","increase_by_percent","alert_id"]}
    }},
    # ── COMPRESSOR CONTROLS ─────────────────────────────────────────────
    {"type":"function","function":{
        "name":"engage_compressor_unloader",
        "description":"Activate compressor unloader valve to reduce compression ratio and lower discharge pressure and temperature. Standard response for HIGH_PRESSURE or OVERTEMPERATURE on compressors.",
        "parameters":{"type":"object","properties":{
            "unit_id":{"type":"string"},
            "alert_id":{"type":"string"}
        },"required":["unit_id","alert_id"]}
    }},
    # ── CONVEYOR CONTROLS ───────────────────────────────────────────────
    {"type":"function","function":{
        "name":"adjust_conveyor_tension",
        "description":"Command PLC to adjust belt tension via tensioning drum motor. Resolves BELT_SLIP and UNDERSPEED caused by mechanical slip.",
        "parameters":{"type":"object","properties":{
            "unit_id":{"type":"string"},
            "alert_id":{"type":"string"}
        },"required":["unit_id","alert_id"]}
    }},
    # ── BOILER CONTROLS ─────────────────────────────────────────────────
    {"type":"function","function":{
        "name":"reduce_boiler_firing_rate",
        "description":"Reduce burner firing rate via BMS (Burner Management System) to lower steam pressure and temperature. Primary response for boiler OVERPRESSURE or OVERTEMPERATURE.",
        "parameters":{"type":"object","properties":{
            "unit_id":{"type":"string"},
            "reduce_by_percent":{"type":"number","description":"Firing rate reduction 20-50%"},
            "alert_id":{"type":"string"}
        },"required":["unit_id","reduce_by_percent","alert_id"]}
    }},
    {"type":"function","function":{
        "name":"open_boiler_feedwater_valve",
        "description":"Open feedwater control valve to increase water supply to boiler drum. Use for LOW_WATER_FLOW alerts. Critical safety action — prevents dry firing.",
        "parameters":{"type":"object","properties":{
            "unit_id":{"type":"string"},
            "alert_id":{"type":"string"}
        },"required":["unit_id","alert_id"]}
    }},
    # ── UNIVERSAL ────────────────────────────────────────────────────────
    {"type":"function","function":{
        "name":"create_maintenance_work_order",
        "description":"Generate maintenance work order in CMMS for physical inspection. Use for mechanical faults (bearing, belt, vibration) that require human technician on-site.",
        "parameters":{"type":"object","properties":{
            "unit_id":{"type":"string"},
            "priority":{"type":"string","enum":["low","medium","high","critical"]},
            "fault_description":{"type":"string","description":"Detailed description of fault and symptoms observed"},
            "recommended_action":{"type":"string","description":"What the technician should inspect or replace"},
            "alert_id":{"type":"string"}
        },"required":["unit_id","priority","fault_description","recommended_action","alert_id"]}
    }},
    {"type":"function","function":{
        "name":"emergency_shutdown",
        "description":"Initiate controlled emergency shutdown via safety PLC. ONLY for imminent safety risk (fire, catastrophic pressure, electrical fault). Stops production. Requires operator confirmation to restart.",
        "parameters":{"type":"object","properties":{
            "unit_id":{"type":"string"},
            "safety_reason":{"type":"string","description":"Specific safety hazard requiring shutdown"},
            "alert_id":{"type":"string"}
        },"required":["unit_id","safety_reason","alert_id"]}
    }},
    {"type":"function","function":{
        "name":"escalate_to_operator",
        "description":"Send urgent notification to on-duty operator via Slack and email. Use when confidence is below threshold or when situation requires human judgement.",
        "parameters":{"type":"object","properties":{
            "alert_id":{"type":"string"},
            "sensor_id":{"type":"string","description":"The sensor this escalation is about, e.g. MOTOR-A1"},
            "diagnosis":{"type":"string","description":"Agent's best diagnosis of the problem"},
            "actions_taken":{"type":"array","items":{"type":"string"}},
            "recommended_next_steps":{"type":"array","items":{"type":"string"}},
            "urgency":{"type":"string","enum":["low","medium","high","critical"]}
        },"required":["alert_id","sensor_id","diagnosis","actions_taken","recommended_next_steps","urgency"]}
    }},
]

# ── FIX-TYPE MAPPING: tool name → what sensor_state.apply_fix receives ──
TOOL_FIX_MAP = {
    "reduce_motor_load":             ("throttle_motor",     lambda a: a.get("reduce_by_percent", 25)),
    "activate_motor_cooling":        ("activate_cooling",   lambda a: a.get("duration_minutes", 20)),
    "clear_motor_fault_and_restart": ("restart_motor",       lambda a: None),
    "open_pressure_bypass_valve":    ("reduce_pressure",     lambda a: None),
    "increase_pump_speed":           ("increase_flow",       lambda a: a.get("increase_by_percent", 20)),
    "engage_compressor_unloader":    ("reduce_pressure",     lambda a: None),
    "adjust_conveyor_tension":       ("fix_speed",           lambda a: None),
    "reduce_boiler_firing_rate":     ("reduce_firing_rate",  lambda a: a.get("reduce_by_percent", 30)),
    "open_boiler_feedwater_valve":   ("open_feedwater",      lambda a: None),
}

async def execute_tool(tool_name: str, tool_args: dict) -> dict:
    fn_map = {
        "get_sensor_history":            _get_sensor_history,
        "reduce_motor_load":             _generic_actuator,
        "activate_motor_cooling":        _generic_actuator,
        "clear_motor_fault_and_restart": _generic_actuator,
        "open_pressure_bypass_valve":    _generic_actuator,
        "increase_pump_speed":           _generic_actuator,
        "engage_compressor_unloader":    _generic_actuator,
        "adjust_conveyor_tension":       _generic_actuator,
        "reduce_boiler_firing_rate":     _generic_actuator,
        "open_boiler_feedwater_valve":   _generic_actuator,
        "emergency_shutdown":            _emergency_shutdown,
        "create_maintenance_work_order": _maintenance_order,
        "escalate_to_operator":          _escalate,
    }
    fn = fn_map.get(tool_name)
    if not fn:
        return {"error": f"Unknown tool: {tool_name}"}
    try:
        result = await fn(tool_name, tool_args) if fn in [_generic_actuator] else await fn(**{k:v for k,v in tool_args.items()})
        alert_id = tool_args.get("alert_id", "unknown")
        if alert_id != "unknown":
            log_remediation(alert_id, tool_name, tool_args, result, result.get("success", True))
        return result
    except Exception as e:
        return {"success": False, "error": str(e), "message": f"Tool {tool_name} failed: {e}"}

async def _get_sensor_history(sensor_id: str, hours: int = 6, **_) -> dict:
    import random
    from backend.tools.sensor_state import THRESHOLDS, WARN_THRESHOLDS
    live = get_state(sensor_id)
    if not live:
        return {"success": False, "error": f"Sensor {sensor_id} not found"}
    thresholds = THRESHOLDS.get(sensor_id, {})
    warn_thresholds = WARN_THRESHOLDS.get(sensor_id, {})
    trend_note = []
    for metric, val in live.items():
        t = thresholds.get(metric)
        w = warn_thresholds.get(metric)
        note = f"{metric}={val}"
        if w is not None or t is not None:
            note += f" (warn={w}, critical={t})"
        trend_note.append(note)
    return {
        "success": True, "sensor_id": sensor_id,
        "current_live_values": live,
        "alert_thresholds": thresholds,          # critical/trip levels
        "warn_thresholds": warn_thresholds,       # early-warning levels
        # use these REAL configured levels, not general textbook figures, when
        # reasoning about margin/severity
        "hours_retrieved": hours,
        "trend": "INCREASING" if any(v > 0 for v in live.values()) else "STABLE",
        "summary": f"Current readings for {sensor_id}: {', '.join(trend_note)}. Data from last {hours}h retrieved."
    }

async def _generic_actuator(tool_name: str, tool_args: dict) -> dict:
    import asyncio
    await asyncio.sleep(0.05)
    unit_id = tool_args.get("motor_id") or tool_args.get("unit_id", "UNKNOWN")
    fix_info = TOOL_FIX_MAP.get(tool_name)
    changes = {}
    if fix_info:
        fix_type, amount_fn = fix_info
        amount = amount_fn(tool_args)
        changes = apply_fix(unit_id, fix_type, amount)
    human_labels = {
        "reduce_motor_load":             f"VFD speed reduced by {tool_args.get('reduce_by_percent','?')}%",
        "activate_motor_cooling":        f"Auxiliary cooling fan ON for {tool_args.get('duration_minutes','?')} min",
        "clear_motor_fault_and_restart": "Fault register cleared, motor restarted via PLC",
        "open_pressure_bypass_valve":    "Bypass relief valve OPEN — pressure relieving",
        "increase_pump_speed":           f"Pump VFD increased by {tool_args.get('increase_by_percent','?')}%",
        "engage_compressor_unloader":    "Compressor unloader valve ENGAGED — load reduced",
        "adjust_conveyor_tension":       "Belt tensioning drum adjusted via PLC",
        "reduce_boiler_firing_rate":     f"Burner firing rate reduced {tool_args.get('reduce_by_percent','?')}% via BMS",
        "open_boiler_feedwater_valve":   "Feedwater control valve OPEN — flow restoring",
    }
    return {
        "success": True, "unit_id": unit_id, "tool": tool_name,
        "sensor_changes": changes,
        "command_sent_to": "PLC/VFD/BMS via Modbus TCP",
        "message": human_labels.get(tool_name, tool_name) + (f". Sensor changes: {_fmt(changes)}" if changes else "")
    }

async def _emergency_shutdown(unit_id: str, safety_reason: str, alert_id: str, **_) -> dict:
    from backend.tools.sensor_state import mark_pending_human
    # Full power cutoff — values drop to powered-down baseline (0 current, 0 speed, near-ambient temp)
    changes = apply_fix(unit_id, 'emergency_shutdown')
    mark_pending_human(unit_id)  # de-energized equipment must not re-fire alerts until restarted by a technician

    # Mandatory notification — a powered-down machine is never silent. Technician must be told.
    notif = await _escalate(
        alert_id=alert_id,
        diagnosis=f"EMERGENCY SHUTDOWN executed on {unit_id}. Reason: {safety_reason}. "
                  f"Power has been cut to prevent further damage or safety hazard.",
        actions_taken=[f"emergency_shutdown: power cut to {unit_id}"],
        recommended_next_steps=[
            "Dispatch technician immediately — equipment is offline and unsafe to restart remotely.",
            "Physical inspection required before any restart is authorized.",
            f"Root cause: {safety_reason}",
        ],
        urgency="critical",
    )

    return {
        "success": True, "unit_id": unit_id, "alert_id": alert_id,
        "status": "POWER_CUT — EMERGENCY SHUTDOWN",
        "sensor_changes": changes,
        "production_impact": "LINE_HALTED — power disconnected, requires technician + manual operator restart",
        "notification": notif.get("notifications", {}),
        "message": f"🔴 POWER CUT — {unit_id} shut down and de-energized. Reason: {safety_reason}. "
                  f"Technician notified via {', '.join(k for k,v in notif.get('notifications',{}).items() if 'sent' in str(v) or 'not configured' not in str(v))  or 'Slack/Email'}."
    }

async def _maintenance_order(unit_id: str, priority: str, fault_description: str,
                             recommended_action: str, alert_id: str, **_) -> dict:
    from backend.tools.sensor_state import mark_pending_human
    wo = f"WO-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
    eta = {"low": "48h", "medium": "24h", "high": "4h", "critical": "1h"}.get(priority, "24h")
    mark_pending_human(unit_id)  # suppress re-firing this sensor until the work order is cleared
    return {
        "success": True, "work_order_id": wo, "unit_id": unit_id,
        "priority": priority, "eta": eta,
        "message": f"Work order {wo} created ({priority}). Technician dispatched. ETA: {eta}. Fault: {fault_description}"
    }

async def _escalate(alert_id: str, diagnosis: str, actions_taken: list,
                    recommended_next_steps: list, urgency: str, sensor_id: str = None, **_) -> dict:
    from backend.tools.sensor_state import mark_pending_human
    if sensor_id:
        mark_pending_human(sensor_id)  # suppress re-firing until operator handles it
    sent = {}
    emoji = {"low":"🟡","medium":"🟠","high":"🔴","critical":"🚨"}.get(urgency,"⚠️")

    slack_url = NOTIFY_CONFIG.get("slack_webhook_url", "")
    if slack_url and "your_slack" not in slack_url:
        payload = {"blocks":[
            {"type":"header","text":{"type":"plain_text","text":f"{emoji} IndustrialPilot — Human Required [{urgency.upper()}]"}},
            {"type":"section","text":{"type":"mrkdwn","text":f"*Alert:* `{alert_id}`\n*Diagnosis:* {diagnosis}"}},
            {"type":"section","text":{"type":"mrkdwn","text":
                "*Actions taken:*\n"+"".join(f"• {a}\n" for a in actions_taken)+
                "\n*Recommended next:*\n"+"".join(f"• {s}\n" for s in recommended_next_steps)}},
        ]}
        try:
            async with httpx.AsyncClient() as c:
                r = await c.post(slack_url, json=payload, timeout=5)
                sent["slack"] = "✅ sent" if r.status_code==200 else f"failed {r.status_code}"
        except Exception as e:
            sent["slack"] = f"error: {e}"
    else:
        sent["slack"] = "not configured"

    # ── Email — sends to every address in NOTIFY_CONFIG["operator_emails"] ──
    recipients = NOTIFY_CONFIG.get("operator_emails", [])
    if not recipients:
        sent["email"] = "no recipients configured"
    elif not SENDGRID_API_KEY:
        # No SendGrid key set — log it clearly instead of silently failing
        print(f"[EMAIL — NOT SENT, no SENDGRID_API_KEY] To: {recipients} | [{urgency.upper()}] {alert_id}: {diagnosis}")
        sent["email"] = f"not configured (would send to {', '.join(recipients)})"
    else:
        subject = f"{emoji} IndustrialPilot [{urgency.upper()}] — {alert_id} needs a technician"
        body_text = (
            f"Alert: {alert_id}\n"
            f"Urgency: {urgency.upper()}\n\n"
            f"Diagnosis:\n{diagnosis}\n\n"
            f"Actions already taken:\n" + "".join(f"- {a}\n" for a in actions_taken) +
            f"\nRecommended next steps:\n" + "".join(f"- {s}\n" for s in recommended_next_steps) +
            f"\n— IndustrialPilot autonomous agent"
        )
        email_payload = {
            "personalizations": [{"to": [{"email": e} for e in recipients]}],
            "from": {"email": FROM_EMAIL, "name": "IndustrialPilot"},
            "subject": subject,
            "content": [{"type": "text/plain", "value": body_text}],
        }
        try:
            async with httpx.AsyncClient() as c:
                r = await c.post(
                    "https://api.sendgrid.com/v3/mail/send",
                    headers={"Authorization": f"Bearer {SENDGRID_API_KEY}", "Content-Type": "application/json"},
                    json=email_payload, timeout=8,
                )
                sent["email"] = f"✅ sent to {', '.join(recipients)}" if r.status_code in (200, 202) else f"failed {r.status_code}: {r.text[:150]}"
        except Exception as e:
            sent["email"] = f"error: {e}"

    return {
        "success": True, "alert_id": alert_id,
        "notifications": sent,
        "message": f"Operator notified [{urgency.upper()}]. Diagnosis: {diagnosis}"
    }

def _fmt(changes: dict) -> str:
    if not changes: return ""
    return " | ".join(f"{k}: {v.get('from','?')}→{v.get('to','?')}" if isinstance(v,dict) else f"{k}→{v}" for k,v in changes.items())
