"""
IndustrialPilot — Deterministic Diagnostic & Decision Engine
══════════════════════════════════════════════════════════════════════════
REWORK NOTE: every previous version of this file asked an LLM to read ~600
lines of rules and reliably (a) classify the fault, (b) compute a numeric
confidence, and (c) decide whether to escalate — every single time, with zero
drift. That's why the same pump problem could resolve once and escalate the
next: the *decision* depended on whether one run of free text happened to
match an exact format.

This version moves the decision itself out of the LLM entirely:

  1. SCAN    — read the real live sensor values and real configured
               thresholds (not text the model remembers).
  2. DIAGNOSE — classify the fault with plain Python comparisons against
               those real numbers (mechanical / electrical / operational /
               hard-break / imminent), using the exact signature logic we
               built together.
  3. CONFIDENCE — a real formula based on how much margin the reading has,
               not a lookup table and not something an LLM eyeballs.
  4. DECIDE  — execute the matching action (lower current/load/speed, open
               a valve, adjust firing rate, etc.) or force a shutdown
               (every parameter to a safe/zero baseline) + call maintenance,
               based on the fault category — not a stated confidence number.
  5. VERIFY  — after acting, check the real resulting readings against the
               real thresholds. If it didn't work, escalate — don't just
               trust that it did.
  6. NARRATE — Qwen is called exactly once, only to turn the above into a
               readable paragraph for the incident report. It cannot change
               what already happened to the machine, and if the API call
               fails or times out, the incident still completes normally
               with a plain-text fallback narrative.

This is intentionally NOT "smarter" than a good engineer's checklist — it's
supposed to be exactly as predictable as one, every single time.
"""
import os
import json
import uuid
import asyncio
from datetime import datetime
from openai import AsyncOpenAI
from dotenv import load_dotenv

from backend.tools.tools import execute_tool
from backend.tools.sensor_state import get_state, THRESHOLDS, WARN_THRESHOLDS
from backend.db.database import (
    insert_incident, update_incident_status, log_decision, save_report
)

load_dotenv()

QWEN_API_KEY  = os.getenv("QWEN_API_KEY", "")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL    = os.getenv("QWEN_MODEL", "qwen3.7-plus")
# Kept for the /api/settings endpoint (routes.py imports this for display) — no longer
# used to gate the escalation decision. That decision is now purely fault-type driven.
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.8"))

LOW_ALERT_METRICS = {"flow_m3s", "speed_m_s"}  # for these, LOW is the dangerous direction


# ══════════════════════════════════════════════════════════════════════
# STEP 3 helper — a real confidence number, scaled by real margin
# ══════════════════════════════════════════════════════════════════════
def _margin_ratio(sensor_id: str, metric: str, value: float) -> float:
    """How far the given value sits from its real critical threshold, as a
    0.0-1.0 ratio on the SAFE side (0 = right at the edge, 1 = comfortably
    clear). Negative if it's still on the unsafe side."""
    thresh = THRESHOLDS.get(sensor_id, {}).get(metric)
    if thresh in (None, 0):
        return 0.5
    if metric in LOW_ALERT_METRICS:
        return (value - thresh) / thresh
    return (thresh - value) / thresh


def _scaled(lo: float, hi: float, ratio: float) -> float:
    ratio = max(0.0, min(1.0, ratio))
    return round(lo + (hi - lo) * ratio, 3)


# ══════════════════════════════════════════════════════════════════════
# Diagnosis result — plain, inspectable, no magic
# ══════════════════════════════════════════════════════════════════════
class Diagnosis:
    def __init__(self, category, reason, confidence, plan):
        self.category = category      # 'operational' | 'mechanical' | 'electrical'
                                       # | 'hard_break' | 'imminent' | 'uncertain'
        self.reason = reason          # plain-English explanation of the READINGS
        self.confidence = confidence  # STEP 1 diagnostic confidence (pre-action)
        self.plan = plan              # list of (tool_name, args_dict) to execute in order


def _pct(v):
    return f"{v:.2f}" if isinstance(v, (int, float)) else str(v)


# ══════════════════════════════════════════════════════════════════════
# STEP 2 — DIAGNOSE, per equipment type
# ══════════════════════════════════════════════════════════════════════
def diagnose_motor(sid, live, thresh, warn, alert_type) -> Diagnosis:
    current, temp, vib = live.get("current_a"), live.get("temp_c"), live.get("vib_mm_s")

    # 1. hard break — current near zero while commanded to run
    if current is not None and current < thresh.get("current_a", 58) * 0.1:
        reason = f"Current at {_pct(current)}A while the motor is commanded to run — consistent with a sheared shaft/coupling, not an electrical or thermal fault."
        return Diagnosis("hard_break", reason, 0.97, [
            ("emergency_shutdown", {"safety_reason": f"Near-zero current ({_pct(current)}A) on a running motor — suspected sheared shaft/coupling."}),
            ("create_maintenance_work_order", {"priority": "high",
                "fault_description": f"{sid}: current collapsed to {_pct(current)}A while commanded to run.",
                "recommended_action": "Inspect motor shaft and coupling for shear/disconnection."}),
        ])

    # 2. electrical — insulation fault always escalates, never a remote action
    if alert_type == "INSULATION_FAULT":
        reason = f"Insulation fault reported on {sid} (current {_pct(current)}A) — electrical winding hazard. No remote action can safely address this."
        return Diagnosis("electrical", reason, 0.93, [
            ("escalate_to_operator", {"urgency": "high", "diagnosis": reason,
                "recommended_next_steps": ["Electrician must physically inspect winding insulation before any restart."]}),
        ])

    # 3. electrical — overcurrent with temperature still normal
    if current is not None and thresh.get("current_a") and current > thresh["current_a"] and (temp is None or temp <= warn.get("temp_c", 999)):
        reason = f"Current {_pct(current)}A exceeds the {thresh['current_a']}A trip while temperature stays normal ({_pct(temp)}°C) — this pattern points to an electrical fault (short/ground), not a thermal overload."
        return Diagnosis("electrical", reason, 0.85, [
            ("escalate_to_operator", {"urgency": "high", "diagnosis": reason,
                "recommended_next_steps": ["Electrician to check for short circuit / ground fault before any load is reapplied."]}),
        ])

    # 4. mechanical — vibration in/near critical, or the alert itself says so
    if (vib is not None and thresh.get("vib_mm_s") and vib > warn.get("vib_mm_s", thresh["vib_mm_s"])) or alert_type in ("EXCESSIVE_VIBRATION", "BEARING_FAULT"):
        deep_critical = vib is not None and thresh.get("vib_mm_s") and vib > thresh["vib_mm_s"] * 1.15
        reason = f"Vibration {_pct(vib)} mm/s (warn {warn.get('vib_mm_s')}, critical {thresh.get('vib_mm_s')}) with current {_pct(current)}A and temp {_pct(temp)}°C — bearing wear/misalignment signature, not something a VFD command can fix."
        plan = []
        if deep_critical:
            plan.append(("emergency_shutdown", {"safety_reason": f"Vibration {_pct(vib)} mm/s deep into critical range — imminent bearing/rotor failure risk."}))
        plan.append(("create_maintenance_work_order", {
            "priority": "high" if deep_critical else "medium",
            "fault_description": reason,
            "recommended_action": "Inspect and replace bearings; check shaft alignment."}))
        return Diagnosis("mechanical", reason, 0.93 if deep_critical else 0.86, plan)

    # 5. operational — elevated but under critical, vibration normal
    if (current is not None and thresh.get("current_a") and current > warn.get("current_a", thresh["current_a"])) or \
       (temp is not None and thresh.get("temp_c") and temp > warn.get("temp_c", thresh["temp_c"])):
        reason = f"Current {_pct(current)}A / temp {_pct(temp)}°C elevated above warn levels, vibration normal ({_pct(vib)} mm/s) — operational overload from load, not a mechanical or electrical defect."
        plan = [("reduce_motor_load", {"reduce_by_percent": 20})]
        if temp is not None and thresh.get("temp_c") and temp > warn.get("temp_c", thresh["temp_c"]):
            plan.append(("activate_motor_cooling", {"duration_minutes": 20}))
        return Diagnosis("operational", reason, 0.75, plan)

    reason = f"Alert fired ({alert_type}) but current live readings (current {_pct(current)}A, temp {_pct(temp)}°C, vib {_pct(vib)} mm/s) don't clearly match a known fault signature."
    return Diagnosis("uncertain", reason, 0.45, [
        ("escalate_to_operator", {"urgency": "medium", "diagnosis": reason,
            "recommended_next_steps": ["Manual inspection recommended — readings don't match a known automated procedure."]}),
    ])


def diagnose_pump(sid, live, thresh, warn, alert_type) -> Diagnosis:
    temp, pressure, flow = live.get("temp_c"), live.get("pressure_bar"), live.get("flow_m3s")

    # 1. hard break — flow or pressure near zero while running
    if (flow is not None and thresh.get("flow_m3s") and flow < thresh["flow_m3s"] * 0.15) or \
       (pressure is not None and thresh.get("pressure_bar") and pressure < thresh["pressure_bar"] * 0.15):
        reason = f"Flow {_pct(flow)} m³/s / pressure {_pct(pressure)} bar collapsed near zero while running — consistent with a sheared impeller or pipe rupture."
        return Diagnosis("hard_break", reason, 0.96, [
            ("emergency_shutdown", {"safety_reason": reason}),
            ("create_maintenance_work_order", {"priority": "high",
                "fault_description": reason, "recommended_action": "Inspect impeller and piping for rupture/shear."}),
        ])

    # 2. mechanical — temp elevated while pressure & flow both stay normal (no vibration sensor on pumps)
    if temp is not None and thresh.get("temp_c") and temp > warn.get("temp_c", thresh["temp_c"]) and \
       (pressure is None or not thresh.get("pressure_bar") or pressure <= warn.get("pressure_bar", thresh["pressure_bar"])) and \
       (flow is None or not thresh.get("flow_m3s") or flow >= warn.get("flow_m3s", thresh["flow_m3s"])):
        reason = f"Temperature {_pct(temp)}°C elevated while pressure ({_pct(pressure)} bar) and flow ({_pct(flow)} m³/s) both stay normal — nothing else explains the heat; this is a mechanical seal/bearing friction signature (no vibration sensor on this unit to confirm further)."
        return Diagnosis("mechanical", reason, 0.84, [
            ("create_maintenance_work_order", {"priority": "medium", "fault_description": reason,
                "recommended_action": "Inspect mechanical seal and bearings for friction/wear."}),
        ])

    # 3. operational — deadheading (high pressure + low flow, temp normal)
    if pressure is not None and thresh.get("pressure_bar") and pressure > warn.get("pressure_bar", thresh["pressure_bar"]):
        reason = f"Pressure {_pct(pressure)} bar above warn ({warn.get('pressure_bar')}) with flow {_pct(flow)} m³/s and temp {_pct(temp)}°C normal — deadheading / blocked downstream valve, an operational condition."
        return Diagnosis("operational", reason, 0.80, [("open_pressure_bypass_valve", {})])

    # 4. operational — cavitation / low flow with everything else normal
    if flow is not None and thresh.get("flow_m3s") and flow < thresh["flow_m3s"]:
        reason = f"Flow {_pct(flow)} m³/s below the {thresh['flow_m3s']} m³/s trip with pressure/temp otherwise unremarkable — cavitation / inlet starvation, an operational condition."
        return Diagnosis("operational", reason, 0.75, [("increase_pump_speed", {"increase_by_percent": 15})])

    reason = f"Alert fired ({alert_type}) but readings (temp {_pct(temp)}, pressure {_pct(pressure)}, flow {_pct(flow)}) don't clearly match a known signature."
    return Diagnosis("uncertain", reason, 0.45, [
        ("escalate_to_operator", {"urgency": "medium", "diagnosis": reason,
            "recommended_next_steps": ["Manual inspection recommended."]}),
    ])


def diagnose_compressor(sid, live, thresh, warn, alert_type) -> Diagnosis:
    temp, pressure, vib = live.get("temp_c"), live.get("pressure_psi"), live.get("vib_mm_s")

    if alert_type == "OIL_PRESSURE_LOW":
        reason = f"Oil pressure low on {sid} — bearings can seize within minutes without lubrication. No remote fix; immediate shutdown required."
        return Diagnosis("imminent", reason, 0.95, [
            ("emergency_shutdown", {"safety_reason": reason}),
            ("create_maintenance_work_order", {"priority": "critical", "fault_description": reason,
                "recommended_action": "Check oil level, pump, and thermal valve before restart."}),
        ])

    if pressure is not None and thresh.get("pressure_psi") and pressure < thresh["pressure_psi"] * 0.05:
        reason = f"Pressure near 0 psi while running — consistent with a failed drive coupling."
        return Diagnosis("hard_break", reason, 0.95, [
            ("emergency_shutdown", {"safety_reason": reason}),
            ("create_maintenance_work_order", {"priority": "high", "fault_description": reason,
                "recommended_action": "Inspect drive coupling."}),
        ])

    if (vib is not None and thresh.get("vib_mm_s") and vib > warn.get("vib_mm_s", thresh["vib_mm_s"])) or alert_type == "EXCESSIVE_VIBRATION":
        deep = vib is not None and thresh.get("vib_mm_s") and vib > thresh["vib_mm_s"]
        reason = f"Vibration {_pct(vib)} mm/s with temp {_pct(temp)}°C and pressure {_pct(pressure)} psi — screw/air-end bearing damage signature."
        plan = [("emergency_shutdown", {"safety_reason": reason})] if deep else []
        plan.append(("create_maintenance_work_order", {"priority": "high" if deep else "medium",
            "fault_description": reason, "recommended_action": "Inspect screw rotor and air-end bearings."}))
        return Diagnosis("mechanical", reason, 0.90 if deep else 0.83, plan)

    if temp is not None and thresh.get("temp_c") and temp > warn.get("temp_c", thresh["temp_c"]) and \
       (vib is None or not thresh.get("vib_mm_s") or vib <= warn.get("vib_mm_s", thresh["vib_mm_s"])) and \
       (pressure is None or not thresh.get("pressure_psi") or pressure <= warn.get("pressure_psi", thresh["pressure_psi"])):
        reason = f"Temp {_pct(temp)}°C rising rapidly while vibration ({_pct(vib)} mm/s) and pressure ({_pct(pressure)} psi) both stay normal — oil circuit/lubrication failure; vibration alone won't show this yet."
        deep = thresh.get("temp_c") and temp > thresh["temp_c"] * 0.95
        plan = [("emergency_shutdown", {"safety_reason": reason})] if deep else []
        plan.append(("create_maintenance_work_order", {"priority": "high", "fault_description": reason,
            "recommended_action": "Check oil circuit, cooler, and lubricant level."}))
        return Diagnosis("mechanical", reason, 0.85, plan)

    if pressure is not None and thresh.get("pressure_psi") and pressure > warn.get("pressure_psi", thresh["pressure_psi"]):
        reason = f"Pressure {_pct(pressure)} psi above warn with temp/vibration normal — unloader valve/regulator not modulating correctly, an operational condition."
        return Diagnosis("operational", reason, 0.80, [("engage_compressor_unloader", {})])

    reason = f"Alert fired ({alert_type}) but readings don't clearly match a known signature."
    return Diagnosis("uncertain", reason, 0.45, [
        ("escalate_to_operator", {"urgency": "medium", "diagnosis": reason,
            "recommended_next_steps": ["Manual inspection recommended."]}),
    ])


def diagnose_conveyor(sid, live, thresh, warn, alert_type) -> Diagnosis:
    temp, speed, current = live.get("temp_c"), live.get("speed_m_s"), live.get("current_a")

    if speed is not None and thresh.get("speed_m_s") and speed < thresh["speed_m_s"] * 0.3 and \
       current is not None and current < warn.get("current_a", 999) * 0.6:
        reason = f"Speed collapsed to {_pct(speed)} m/s with current only {_pct(current)}A (not spiking) — consistent with a snapped or derailed belt, not a jam."
        return Diagnosis("hard_break", reason, 0.93, [
            ("create_maintenance_work_order", {"priority": "high", "fault_description": reason,
                "recommended_action": "Inspect belt for breakage/derailment."}),
        ])

    if speed is not None and thresh.get("speed_m_s") and speed < thresh["speed_m_s"] and \
       current is not None and thresh.get("current_a") and current > thresh["current_a"]:
        reason = f"Speed {_pct(speed)} m/s below the jam floor with current spiking to {_pct(current)}A and temp {_pct(temp)}°C — mechanical jam."
        return Diagnosis("mechanical", reason, 0.91, [
            ("emergency_shutdown", {"safety_reason": reason}),
            ("create_maintenance_work_order", {"priority": "high", "fault_description": reason,
                "recommended_action": "Clear jam before restart; inspect for material buildup."}),
        ])

    if speed is not None and thresh.get("speed_m_s") and speed < warn.get("speed_m_s", thresh["speed_m_s"]) and \
       temp is not None and thresh.get("temp_c") and temp > warn.get("temp_c", thresh["temp_c"]):
        reason = f"Speed fluctuating low ({_pct(speed)} m/s) AND temp climbing ({_pct(temp)}°C) — drive-drum friction/polishing, not just slack tension."
        return Diagnosis("mechanical", reason, 0.84, [
            ("adjust_conveyor_tension", {}),
            ("create_maintenance_work_order", {"priority": "medium", "fault_description": reason,
                "recommended_action": "Inspect drive drum surface and belt tensioners."}),
        ])

    if speed is not None and thresh.get("speed_m_s") and speed < warn.get("speed_m_s", thresh["speed_m_s"]):
        reason = f"Speed {_pct(speed)} m/s below warn with temp normal ({_pct(temp)}°C) — simple tension slip, an operational adjustment."
        return Diagnosis("operational", reason, 0.80, [("adjust_conveyor_tension", {})])

    if current is not None and thresh.get("current_a") and current > warn.get("current_a", thresh["current_a"]) and \
       temp is not None and thresh.get("temp_c") and temp <= warn.get("temp_c", thresh["temp_c"]):
        reason = f"Current {_pct(current)}A elevated with speed only slightly reduced ({_pct(speed)} m/s) and temp normal — heavier load on the belt, operational, not mechanical."
        return Diagnosis("operational", reason, 0.72, [("adjust_conveyor_tension", {})])

    reason = f"Alert fired ({alert_type}) but readings don't clearly match a known signature."
    return Diagnosis("uncertain", reason, 0.45, [
        ("escalate_to_operator", {"urgency": "medium", "diagnosis": reason,
            "recommended_next_steps": ["Manual inspection recommended."]}),
    ])


def diagnose_boiler(sid, live, thresh, warn, alert_type) -> Diagnosis:
    temp, pressure, flow = live.get("temp_c"), live.get("pressure_bar"), live.get("flow_m3s")

    if alert_type == "FLAME_FAILURE":
        reason = "Flame failure — burner shut off. Gas accumulation risk; never a remote restart."
        return Diagnosis("imminent", reason, 0.96, [
            ("emergency_shutdown", {"safety_reason": reason}),
            ("escalate_to_operator", {"urgency": "critical", "diagnosis": reason,
                "recommended_next_steps": ["Check fuel supply, ignition, and flame detector before any restart."]}),
        ])

    if pressure is not None and thresh.get("pressure_bar") and pressure < thresh["pressure_bar"] * 0.55 and \
       flow is not None and thresh.get("flow_m3s") and flow < thresh["flow_m3s"] * 0.85:
        reason = f"Pressure ({_pct(pressure)} bar) and flow ({_pct(flow)} m³/s) collapsed together with temp {_pct(temp)}°C — tube rupture / structural leak, dry-fire risk."
        return Diagnosis("imminent", reason, 0.94, [
            ("emergency_shutdown", {"safety_reason": reason}),
            ("create_maintenance_work_order", {"priority": "critical", "fault_description": reason,
                "recommended_action": "Inspect for tube rupture / structural leak before any restart."}),
            ("escalate_to_operator", {"urgency": "critical", "diagnosis": reason,
                "recommended_next_steps": ["Boiler inspector required before restart."]}),
        ])

    if flow is not None and thresh.get("flow_m3s") and flow < thresh["flow_m3s"] and \
       pressure is not None and thresh.get("pressure_bar") and pressure > warn.get("pressure_bar", thresh["pressure_bar"]):
        reason = f"Flow low ({_pct(flow)} m³/s) while pressure stays HIGH ({_pct(pressure)} bar) — internal scale buildup/blockage, not a feedwater supply issue (opening the feedwater valve alone won't fix a blockage)."
        return Diagnosis("mechanical", reason, 0.83, [
            ("create_maintenance_work_order", {"priority": "high", "fault_description": reason,
                "recommended_action": "Schedule descaling / internal inspection."}),
        ])

    if flow is not None and thresh.get("flow_m3s") and flow < thresh["flow_m3s"]:
        reason = f"Flow {_pct(flow)} m³/s below the critical floor with pressure not elevated — genuine feedwater supply shortfall, urgent but operational."
        plan = [("open_boiler_feedwater_valve", {})]
        if flow < thresh["flow_m3s"] * 0.4:
            plan.insert(0, ("emergency_shutdown", {"safety_reason": f"Flow at {_pct(flow)} m³/s, under 40% of the critical floor — dry-fire risk while feedwater is restored."}))
        return Diagnosis("operational", reason, 0.78, plan)

    if pressure is not None and thresh.get("pressure_bar") and pressure > warn.get("pressure_bar", thresh["pressure_bar"]):
        reason = f"Pressure {_pct(pressure)} bar / temp {_pct(temp)}°C above warn with flow normal/low — firing rate exceeds steam demand, operational."
        return Diagnosis("operational", reason, 0.80, [("reduce_boiler_firing_rate", {"reduce_by_percent": 30})])

    reason = f"Alert fired ({alert_type}) but readings don't clearly match a known signature."
    return Diagnosis("uncertain", reason, 0.45, [
        ("escalate_to_operator", {"urgency": "medium", "diagnosis": reason,
            "recommended_next_steps": ["Manual inspection recommended."]}),
    ])


_DIAGNOSERS = {
    "MOTOR": diagnose_motor, "PUMP": diagnose_pump, "COMP": diagnose_compressor,
    "CONV": diagnose_conveyor, "BOIL": diagnose_boiler,
}


def diagnose(sensor_id: str, alert_type: str, live: dict) -> Diagnosis:
    thresh = THRESHOLDS.get(sensor_id, {})
    warn = WARN_THRESHOLDS.get(sensor_id, {})
    prefix = sensor_id.split("-")[0]
    fn = _DIAGNOSERS.get(prefix)
    if not fn:
        reason = f"Unrecognized equipment prefix for {sensor_id}."
        return Diagnosis("uncertain", reason, 0.3, [
            ("escalate_to_operator", {"urgency": "medium", "diagnosis": reason, "recommended_next_steps": ["Manual review required."]}),
        ])
    return fn(sensor_id, live, thresh, warn, alert_type)


# categories that mean "a human needs to make a decision right now" — the equipment
# being safely shut down is NOT the same as needing a live decision: a hard mechanical
# break or a bearing fault already has its outcome decided (ticket + safe state), so
# those are NOT in this set, matching the "auto-resolve + advisory" tier.
ESCALATING_CATEGORIES = {"electrical", "imminent", "uncertain"}
# categories fully handled by ticket/shutdown alone — no live human decision needed,
# just an informational notice so the follow-up repair isn't forgotten
ADVISORY_ONLY_CATEGORIES = {"mechanical", "hard_break"}


class IndustrialPilotAgent:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)

    async def process_alert(self, alert: dict, websocket_callback=None) -> dict:
        alert_id = alert.get("alert_id", f"ALT-{uuid.uuid4().hex[:8].upper()}")
        sensor_id = alert.get("sensor_id", "UNKNOWN")
        alert_type = alert.get("alert_type", "UNKNOWN")
        severity = alert.get("severity", "medium")

        insert_incident(alert_id, sensor_id, alert_type, severity, alert)
        await self._cb(websocket_callback, {
            "type": "agent_start", "alert_id": alert_id,
            "message": f"🔍 Analyzing {alert_id} — {alert_type} on {sensor_id} [{severity.upper()}]"
        })

        tool_results = []

        try:
            # ── STEP 1: SCAN ──────────────────────────────────────────
            hist = await execute_tool("get_sensor_history", {"sensor_id": sensor_id})
            tool_results.append({"tool": "get_sensor_history", "args": {"sensor_id": sensor_id}, "result": hist})
            live_before = get_state(sensor_id)
            await self._cb(websocket_callback, {
                "type": "agent_reasoning", "alert_id": alert_id,
                "message": f"📡 Live readings: {live_before}"
            })

            # ── STEP 2: DIAGNOSE ──────────────────────────────────────
            diag = diagnose(sensor_id, alert_type, live_before)
            await self._cb(websocket_callback, {
                "type": "agent_reasoning", "alert_id": alert_id,
                "message": f"🧠 Diagnosis [{diag.category}]: {diag.reason}"
            })

            # ── STEP 4: DECIDE / execute the plan ─────────────────────
            changed_metrics = set()
            for tool_name, extra_args in diag.plan:
                args = self._build_args(tool_name, sensor_id, alert_id, extra_args)
                await self._cb(websocket_callback, {
                    "type": "tool_call", "alert_id": alert_id, "tool": tool_name,
                    "message": f"⚙️ Calling {tool_name}"
                })
                result = await execute_tool(tool_name, args)
                tool_results.append({"tool": tool_name, "args": args, "result": result})
                for m in (result.get("sensor_changes") or {}):
                    changed_metrics.add(m)
                await self._cb(websocket_callback, {
                    "type": "tool_result", "alert_id": alert_id, "tool": tool_name,
                    "sensor_changes": result.get("sensor_changes", {}),
                    "message": f"✅ {tool_name}: {result.get('message', '')}"
                })

            # ── STEP 5: VERIFY ─────────────────────────────────────────
            live_after = get_state(sensor_id)
            thresh = THRESHOLDS.get(sensor_id, {})
            verified, worst_margin, verify_lines = self._verify(sensor_id, changed_metrics, live_after, thresh)

            escalated = diag.category in ESCALATING_CATEGORIES
            confidence = diag.confidence

            if diag.category == "operational":
                if changed_metrics and verified:
                    confidence = _scaled(0.86, 0.97, worst_margin)
                elif changed_metrics and not verified:
                    # ONE bounded retry with a stronger action before giving up —
                    # not an open-ended loop, just a single second attempt.
                    retry_tool, retry_args = diag.plan[0][0], dict(diag.plan[0][1])
                    if "reduce_by_percent" in retry_args or retry_tool in ("reduce_motor_load", "reduce_boiler_firing_rate"):
                        retry_args["reduce_by_percent"] = min(45, retry_args.get("reduce_by_percent", 20) + 20)
                    elif "increase_by_percent" in retry_args or retry_tool == "increase_pump_speed":
                        retry_args["increase_by_percent"] = min(30, retry_args.get("increase_by_percent", 15) + 15)
                    args = self._build_args(retry_tool, sensor_id, alert_id, retry_args)
                    await self._cb(websocket_callback, {
                        "type": "agent_reasoning", "alert_id": alert_id,
                        "message": f"↻ First action wasn't enough — retrying {retry_tool} with a stronger setting."
                    })
                    result = await execute_tool(retry_tool, args)
                    tool_results.append({"tool": retry_tool, "args": args, "result": result})
                    for m in (result.get("sensor_changes") or {}):
                        changed_metrics.add(m)
                    live_after = get_state(sensor_id)
                    verified, worst_margin, verify_lines = self._verify(sensor_id, changed_metrics, live_after, thresh)
                    if verified:
                        confidence = _scaled(0.80, 0.90, worst_margin)
                    else:
                        confidence = _scaled(0.35, 0.55, worst_margin)
                        escalated = True
                        esc_reason = f"Attempted remediation twice on {sensor_id}; readings still outside safe range: {verify_lines}"
                        result = await execute_tool("escalate_to_operator", self._build_args(
                            "escalate_to_operator", sensor_id, alert_id, {
                                "urgency": "high", "diagnosis": esc_reason,
                                "recommended_next_steps": ["Automated remediation did not resolve this — manual intervention required."]}))
                        tool_results.append({"tool": "escalate_to_operator", "args": {}, "result": result})

            elif diag.category in ADVISORY_ONLY_CATEGORIES:
                # Ticket (and possibly a safety shutdown) already fully resolves this —
                # nothing remote can verify a physical repair, so confidence reflects
                # classification certainty, not a sensor check. Send a low-priority
                # advisory so the follow-up repair isn't forgotten, without blocking
                # on a human decision now.
                advisory = await execute_tool("escalate_to_operator", self._build_args(
                    "escalate_to_operator", sensor_id, alert_id, {
                        "urgency": "info", "diagnosis": diag.reason,
                        "recommended_next_steps": ["Already handled (ticket/shutdown) — this is FYI only, no action needed right now."]}))
                tool_results.append({"tool": "escalate_to_operator", "args": {}, "result": advisory})

            elif diag.category in ESCALATING_CATEGORIES:
                # Guarantee a real notification fires for every escalating category,
                # even if this specific diagnosis didn't already include one in its plan.
                already_notified = any(t["tool"] == "escalate_to_operator" for t in tool_results)
                if not already_notified:
                    urgency = "critical" if diag.category == "imminent" else "high"
                    result = await execute_tool("escalate_to_operator", self._build_args(
                        "escalate_to_operator", sensor_id, alert_id, {
                            "urgency": urgency, "diagnosis": diag.reason,
                            "recommended_next_steps": ["Human review required before this can proceed."]}))
                    tool_results.append({"tool": "escalate_to_operator", "args": {}, "result": result})

            action_tools_called = [t["tool"] for t in tool_results if t["tool"] != "get_sensor_history"]

            # ── STEP 6: NARRATE (Qwen, optional, never changes the decision) ──
            narrative = await self._narrate(sensor_id, alert_type, diag, verify_lines, confidence, escalated)

            log_decision(
                alert_id=alert_id, reasoning=narrative, confidence=confidence,
                action_taken=json.dumps(action_tools_called),
                action_result={"tools": tool_results, "category": diag.category},
                escalated=escalated,
            )
            report = self._build_report(alert_id, alert, tool_results, narrative, diag, confidence, escalated)
            save_report(alert_id, report)
            update_incident_status(alert_id, "escalated" if escalated else "resolved")

            status_msg = (
                f"🚨 ESCALATED [{diag.category}] — confidence {confidence:.0%}"
                if escalated else
                f"✅ AUTO-RESOLVED [{diag.category}] — confidence {confidence:.0%}"
            )
            await self._cb(websocket_callback, {
                "type": "agent_complete", "alert_id": alert_id, "escalated": escalated,
                "confidence": confidence, "message": status_msg
            })
            return {
                "alert_id": alert_id, "status": "escalated" if escalated else "resolved",
                "confidence": confidence, "escalated": escalated,
                "tool_results": tool_results, "category": diag.category,
            }

        except Exception as e:
            error_msg = str(e)
            await self._cb(websocket_callback, {"type": "agent_error", "alert_id": alert_id,
                "message": f"❌ Agent error: {error_msg[:200]}"})
            log_decision(alert_id=alert_id, reasoning=f"Agent error: {error_msg}", confidence=0.0,
                action_taken="none", action_result={"error": error_msg}, escalated=True)
            update_incident_status(alert_id, "error")
            save_report(alert_id, f"# Error Report\nAgent failed: {error_msg}")
            await self._cb(websocket_callback, {"type": "agent_complete", "alert_id": alert_id,
                "escalated": True, "confidence": 0.0, "message": f"⚠️ Incident {alert_id} saved — agent error"})
            return {"alert_id": alert_id, "status": "error", "error": error_msg}

    # ── helpers ────────────────────────────────────────────────────────
    def _build_args(self, tool_name: str, sensor_id: str, alert_id: str, extra: dict) -> dict:
        args = dict(extra)
        args["alert_id"] = alert_id
        if tool_name in ("reduce_motor_load", "activate_motor_cooling", "clear_motor_fault_and_restart"):
            args["motor_id"] = sensor_id
        elif tool_name == "escalate_to_operator":
            args.setdefault("sensor_id", sensor_id)
            args.setdefault("actions_taken", [])
        elif tool_name in ("emergency_shutdown", "create_maintenance_work_order"):
            args["unit_id"] = sensor_id
        else:
            args["unit_id"] = sensor_id
        return args

    def _verify(self, sensor_id, changed_metrics, live_after, thresh):
        if not changed_metrics:
            return False, 0.0, "N/A — no remote reading changed"
        lines, ratios, all_pass = [], [], True
        for m in changed_metrics:
            if m not in thresh or m not in live_after:
                continue
            ratio = _margin_ratio(sensor_id, m, live_after[m])
            ratios.append(ratio)
            passed = ratio >= 0
            all_pass = all_pass and passed
            lines.append(f"{m} {_pct(live_after[m])} vs {thresh[m]} → {'PASS' if passed else 'FAIL'}")
        worst = min(ratios) if ratios else 0.0
        return all_pass, worst, "; ".join(lines)

    async def _narrate(self, sensor_id, alert_type, diag, verify_lines, confidence, escalated) -> str:
        """One optional Qwen call purely to phrase the incident nicely. The decision
        was already made deterministically above — if this call fails, times out, or
        is disabled, the incident still completes with a plain template."""
        fallback = (
            f"{sensor_id} — {alert_type}. {diag.reason} "
            f"Verification: {verify_lines}. Confidence {confidence:.0%}. "
            f"{'Escalated for human review.' if escalated else 'Auto-resolved.'}"
        )
        if not QWEN_API_KEY:
            return fallback
        try:
            resp = await asyncio.wait_for(self.client.chat.completions.create(
                model=QWEN_MODEL,
                messages=[
                    {"role": "system", "content": (
                        "Rewrite the following factual incident summary as 2-3 clear, "
                        "professional sentences for a factory incident report. Do not add "
                        "new facts, numbers, or conclusions beyond what's given — just "
                        "phrase it clearly for a human reader."
                    )},
                    {"role": "user", "content": fallback},
                ],
                temperature=0.3, max_tokens=200,
            ), timeout=8)
            text = resp.choices[0].message.content
            return text.strip() if text else fallback
        except Exception:
            return fallback

    async def _cb(self, fn, data):
        if fn:
            try:
                await fn(data)
            except Exception:
                pass

    def _build_report(self, alert_id, alert, tool_results, narrative, diag, confidence, escalated) -> str:
        seen = {}
        for t in tool_results:
            key = (t["tool"], t["result"].get("message", "done"))
            seen[key] = seen.get(key, 0) + 1
        lines = [f"- {tool}{f' (×{c})' if c > 1 else ''}: {msg}" for (tool, msg), c in seen.items()]
        return f"""# Incident Report — {alert_id}
**Generated:** {datetime.utcnow().isoformat()}

## Alert
- Sensor: {alert.get('sensor_id')}
- Type: {alert.get('alert_type')}
- Severity: {alert.get('severity','').upper()}

## Diagnosis
**Category:** {diag.category}
{diag.reason}

## Confidence
{confidence:.0%}

## Actions Taken
{chr(10).join(lines) or 'None'}

## Outcome
{'ESCALATED_TO_HUMAN' if escalated else 'AUTO_RESOLVED'}

## Narrative
{narrative}
"""
