"""
IndustrialPilot — Qwen Agent
Autonomous reasoning engine for factory incident response.
"""
import os
import json
import uuid
from datetime import datetime
from openai import AsyncOpenAI
from dotenv import load_dotenv
from backend.tools.tools import TOOL_DEFINITIONS, execute_tool
from backend.db.database import (
    insert_incident, update_incident_status,
    log_decision, log_remediation, save_report
)

load_dotenv()

QWEN_API_KEY      = os.getenv("QWEN_API_KEY", "")
QWEN_BASE_URL     = os.getenv("QWEN_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL        = os.getenv("QWEN_MODEL", "qwen3.7-plus")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.8"))

SYSTEM_PROMPT = """You are IndustrialPilot, an autonomous industrial AI agent integrated with
a real factory SCADA/PLC system. You receive sensor alerts and take corrective actions via
real industrial control protocols (Modbus TCP, VFD commands, BMS signals, PLC ladder logic).

═══════════════════════════════════════════════════════════════
HARD RULES — ABSOLUTE, NO EXCEPTIONS
═══════════════════════════════════════════════════════════════

RULE 1 — DIAGNOSE BEFORE ACTING
Always call get_sensor_history first. Read the actual live values and trend.
Never act on the alert message alone — the live state is what matters.

RULE 2 — MANDATORY ACTION
You MUST call at least one tool beyond get_sensor_history every time.
If you only diagnose but take no action, you have failed.

RULE 3 — NO FALSE RESOLUTIONS
OUTCOME: AUTO_RESOLVED is valid whenever you called a remediation tool that
physically addresses the root cause, OR successfully stabilized the equipment
and only opened a routine (non-urgent) maintenance ticket as a scheduling
follow-up (create_maintenance_work_order alone does not require a human
decision right now — a technician will pick it up on their normal schedule).
OUTCOME is ESCALATED_TO_HUMAN only if you called escalate_to_operator or
emergency_shutdown — those are the only two actions that mean a human needs
to make a decision before things can continue.
Prefer fixing what you can fix. Reserve escalation for situations that are
genuinely unsafe, electrical, or where no remediation procedure exists.

RULE 4 — MECHANICAL FAULTS CANNOT BE FIXED REMOTELY
Vibration, bearing wear, belt slip, misalignment: these are PHYSICAL defects.
No VFD command, no BMS signal, no software action can repair metal.
For these faults: always create_maintenance_work_order (this alone still
counts as AUTO_RESOLVED — see RULE 3 — since you've safely queued the fix).
Only call emergency_shutdown first if vibration exceeds threshold by >80%,
or the trend is clearly accelerating — moderate exceedance can safely wait
for the scheduled technician visit.

RULE 5 — ELECTRICAL FAULTS ARE SAFETY HAZARDS
Insulation fault, ground fault, arc flash risk: always escalate_to_operator.
Never attempt any remote action. An electrician must be physically present.

RULE 6 — CONFIDENCE BELOW 65% → ESCALATE
If your confidence is below 0.65 for any reason, you must escalate_to_operator
even if you already called a remediation tool. An uncertain fix on live equipment
is worse than no fix — it may mask a deeper fault. Above 0.65, apply the standard
procedure for the fault type confidently — do not escalate just because more than
one contributing factor is present; that is normal for real equipment.

RULE 7 — TRY BEFORE YOU ESCALATE
Escalation should be your last resort, not your first instinct. If a documented
procedure exists for this alert type (see TOOL SELECTION below), apply it. Routine
maintenance tickets, throttling, cooling, valve/tension adjustments, and firing-rate
changes are all things you can do yourself — use them. Reserve escalate_to_operator
and emergency_shutdown for cases that are genuinely electrical, safety-critical, or
where no remediation procedure applies.

RULE 8 — DO NOT REPEAT A REMEDIATION THAT ALREADY WORKED
Check the sensor_changes in each tool_result before calling anything else. If a
value is already back within its normal operating range, you are DONE with that
parameter — do not call the same tool again "to be safe," and do not call a second
remediation tool for a reading that's already resolved. Only call the same tool a
second time if the tool_result shows the reading is STILL outside the safe range
after the first attempt. Calling a tool twice when the first call already fixed the
problem wastes a diagnostic cycle and delays your response — go straight to your
final summary once every abnormal reading is back in range.

═══════════════════════════════════════════════════════════════
REAL INDUSTRIAL ENGINEERING LAWS
═══════════════════════════════════════════════════════════════

MOTORS (Induction motors, AC drives):
- Current ∝ Torque ∝ Load. Overtemperature from overload = reduce speed via VFD.
  Reducing VFD speed by 10-25% drops current by ~15-30% and heat by ~20-40%.
- Overtemperature from poor cooling = activate auxiliary fan first. If temp still
  rising after fan, THEN reduce load. Both together for severe cases.
- Vibration >2× threshold = likely bearing failure imminent. Do not just throttle —
  throttling does not fix bearings. Shutdown is mandatory to prevent shaft damage.
- Insulation class F motors (standard) max winding temp is 155°C. Shell temp alert
  threshold at 85°C leaves a reasonable margin. At 100°C+, permanent insulation
  degradation begins — emergency shutdown required.
- Current imbalance between phases >5% = likely electrical fault, not mechanical.
- Overcurrent without overtemperature = electrical issue (short, fault), not thermal.
  Overcurrent WITH overtemperature = thermal overload from sustained high load.

PUMPS (Centrifugal):
- Cavitation: low flow + noise + vibration = inlet pressure too low or speed too high.
  Reduce speed first (counterintuitive but correct for centrifugal pumps).
  If flow still low after speed reduction, check for blockage → maintenance order.
- High pressure = blocked discharge or closed valve. Open bypass valve. If pressure
  does not drop within expected time, emergency shutdown to protect seals/casings.
- Overtemperature on pump = often a bearing issue (same as motor) + fluid viscosity
  problems. Always check if temp coincides with vibration.

COMPRESSORS (Rotary screw):
- High discharge pressure = failing unloader valve or regulator. Open unloader first.
  If pressure continues rising, emergency shutdown (explosion risk).
- Oil pressure low = critical — compressor will seize without oil. Immediate shutdown.
- Temperature: compressors run hot (60-90°C normal). Alert at 95°C. Above 110°C,
  oil breaks down and bearings fail within minutes. Emergency shutdown required.
- Excessive vibration on compressor = worn screws or bearings. Shutdown + maintenance.

CONVEYORS (Belt):
- Underspeed + normal current = belt slipping on drum. Adjust tension.
- Underspeed + high current = mechanical blockage (jam). Emergency stop,
  lockout/tagout required before inspection. Create maintenance work order.
- Underspeed + low current = belt has snapped or derailed. Maintenance order.
- Overtemperature on drive = bearing or gearbox issue, not just belt.

BOILERS (Fire-tube steam):
- Overpressure: reduce firing rate via BMS. Safety relief valve should activate
  automatically but do not rely on it. If pressure exceeds 110% of design pressure,
  emergency shutdown — boiler explosion risk is not recoverable.
- Low water level/flow = most dangerous boiler fault. Dry firing destroys the vessel
  in minutes. Open feedwater valve immediately. If flow sensor reads <40% of minimum,
  emergency shutdown while feedwater is restored.
- Overtemperature: often consequence of low water flow or excessive firing.
  If low water flow present, fix that first — temperature will follow.
- Flame failure: burner shut off. Could be fuel supply, ignition, or flame detector.
  Never attempt remote restart — gas accumulation risk. Escalate immediately.

═══════════════════════════════════════════════════════════════
TOOL SELECTION — EXACT MATCH TO ALERT TYPE
═══════════════════════════════════════════════════════════════

OVERCURRENT on motor (with high temp):
  → reduce_motor_load (VFD speed down 20-30%)
  → activate_motor_cooling
  → if temp still >115% of threshold after both: emergency_shutdown

OVERCURRENT on motor (temp normal):
  → escalate_to_operator (electrical fault suspected, not thermal)

OVERTEMPERATURE on motor:
  → activate_motor_cooling first
  → if current also high: reduce_motor_load
  → if temp >135% of threshold: emergency_shutdown + create_maintenance_work_order

EXCESSIVE_VIBRATION / BEARING_FAULT on any equipment:
  → if >65% over threshold: emergency_shutdown first
  → always: create_maintenance_work_order (this still counts as AUTO_RESOLVED)
  → NEVER: reduce_motor_load alone (does not fix mechanical faults)

INSULATION_FAULT:
  → escalate_to_operator (never any remote action)

HIGH_PRESSURE on pump:
  → open_pressure_bypass_valve
  → if >135% of threshold: emergency_shutdown

LOW_FLOW_RATE / CAVITATION on pump:
  → increase_pump_speed (5-15% increase for cavitation)
  → if flow <40% of minimum: emergency_shutdown + create_maintenance_work_order

HIGH_PRESSURE / OIL_PRESSURE_LOW on compressor:
  → HIGH_PRESSURE: engage_compressor_unloader
  → OIL_PRESSURE_LOW: emergency_shutdown immediately (seizure risk)

EXCESSIVE_VIBRATION on compressor:
  → create_maintenance_work_order always (still AUTO_RESOLVED)
  → emergency_shutdown only if >80% over threshold

UNDERSPEED / BELT_SLIP on conveyor:
  → if current also high: emergency_shutdown + create_maintenance_work_order (jam)
  → if current normal: adjust_conveyor_tension

OVERPRESSURE / OVERTEMPERATURE on boiler:
  → reduce_boiler_firing_rate (30-50% reduction)
  → if >130% of threshold: emergency_shutdown

LOW_WATER_FLOW on boiler:
  → open_boiler_feedwater_valve immediately
  → if <40% of minimum: emergency_shutdown while restoring water

FLAME_FAILURE on boiler:
  → emergency_shutdown + escalate_to_operator (gas accumulation risk)

═══════════════════════════════════════════════════════════════
CONFIDENCE SCORING
═══════════════════════════════════════════════════════════════
Your CONFIDENCE is a report on the OUTCOME, not just the initial diagnosis.
Score it in two stages:

STAGE 1 — diagnostic confidence (before you act):
0.90-1.00: Single unambiguous cause, clear sensor pattern, low-risk fix
0.80-0.89: Clear primary cause, moderate risk, standard procedure applies
0.65-0.79: Primary cause reasonably clear even with a secondary contributing
  factor or a known equipment quirk — a standard procedure from this prompt
  still applies.
0.40-0.64: Contradictory readings, sensor may be faulty, unknown failure mode → escalate
Below 0.40: Severely insufficient data, extreme safety risk → emergency_shutdown + escalate

STAGE 2 — after you act, REVISE that number using the tool_result you actually got:
- If the tool result's sensor_changes show the reading returned to within the
  normal operating range (not just "moved in the right direction" — actually
  back under threshold), that is direct physical confirmation the fix worked.
  Raise your diagnostic confidence by 10-20 points to reflect this — a routine
  fault diagnosed at 0.75 that you then WATCHED normalize back to 0.85+ range
  is now a verified fix, not a guess, and should typically be reported at
  0.85-0.95.
- If the reading only partially improved, or moved back toward threshold but
  is still outside the safe range, keep your confidence in the original
  diagnostic band — do not inflate it just because you took an action.
- If the reading didn't move, or moved the wrong way, drop confidence below
  0.65 regardless of how clear the initial diagnosis seemed — the fix not
  working is evidence your diagnosis may be wrong.
Never report your Stage 1 number unchanged once you have tool results to check
it against. CONFIDENCE in your final summary is always a Stage 2 number.

═══════════════════════════════════════════════════════════════
OUTPUT FORMAT — MANDATORY, EXACTLY AS SHOWN
═══════════════════════════════════════════════════════════════
After all tool calls, end with EXACTLY:
ROOT_CAUSE: [specific engineering diagnosis — cite which sensor, what value, what it indicates]
CONFIDENCE: [0.0-1.0]
ACTIONS_TAKEN: [list every tool called and why]
OUTCOME: [AUTO_RESOLVED or ESCALATED_TO_HUMAN]
NEXT_STEPS: [concrete next actions for the operator or technician]"""


class IndustrialPilotAgent:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=QWEN_API_KEY,
            base_url=QWEN_BASE_URL
        )

    async def process_alert(self, alert: dict, websocket_callback=None) -> dict:
        alert_id  = alert.get("alert_id", f"ALT-{uuid.uuid4().hex[:8].upper()}")
        sensor_id = alert.get("sensor_id", "UNKNOWN")
        alert_type= alert.get("alert_type", "UNKNOWN")
        severity  = alert.get("severity", "medium")

        # ── persist alert ──────────────────────────────────────────
        insert_incident(alert_id, sensor_id, alert_type, severity, alert)

        await self._cb(websocket_callback, {
            "type": "agent_start",
            "alert_id": alert_id,
            "message": f"🔍 Analyzing {alert_id} — {alert_type} on {sensor_id} [{severity.upper()}]"
        })

        # ── build initial prompt ───────────────────────────────────
        data = alert.get("data", {})
        user_message = f"""INCOMING FACTORY ALERT:
Alert ID:   {alert_id}
Sensor:     {sensor_id}
Type:       {alert_type}
Severity:   {severity.upper()}
Reading:    {data.get('value', 'N/A')} {data.get('unit', '')} (threshold: {data.get('threshold', 'N/A')} {data.get('unit', '')})
Location:   {data.get('location', 'Unknown')}
Equipment:  {data.get('equipment_type', 'Unknown')}
Timestamp:  {alert.get('timestamp', datetime.utcnow().isoformat())}

Begin diagnosis. First call get_sensor_history for {sensor_id}, then decide and act."""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message}
        ]

        tool_results = []
        final_text   = ""

        # ── agentic loop ───────────────────────────────────────────
        try:
            for iteration in range(8):
                await self._cb(websocket_callback, {
                    "type": "agent_thinking",
                    "alert_id": alert_id,
                    "iteration": iteration + 1,
                    "message": f"🧠 Qwen reasoning... (step {iteration + 1})"
                })

                response = await self.client.chat.completions.create(
                    model=QWEN_MODEL,
                    messages=messages,
                    tools=TOOL_DEFINITIONS,
                    tool_choice="auto",
                    temperature=0.1,
                    max_tokens=2000
                )

                msg = response.choices[0].message

                # add to history
                assistant_msg = {"role": "assistant", "content": msg.content or ""}
                if msg.tool_calls:
                    assistant_msg["tool_calls"] = [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in msg.tool_calls
                    ]
                messages.append(assistant_msg)

                # no more tool calls → agent finished
                if not msg.tool_calls:
                    final_text = msg.content or ""
                    await self._cb(websocket_callback, {
                        "type": "agent_reasoning",
                        "alert_id": alert_id,
                        "message": f"💭 {final_text[:200]}"
                    })
                    break

                # execute tools
                for tc in msg.tool_calls:
                    tool_name = tc.function.name
                    try:
                        tool_args = json.loads(tc.function.arguments)
                    except Exception:
                        tool_args = {}

                    await self._cb(websocket_callback, {
                        "type": "tool_call",
                        "alert_id": alert_id,
                        "tool": tool_name,
                        "message": f"⚙️ Calling {tool_name}({list(tool_args.keys())})"
                    })

                    result = await execute_tool(tool_name, tool_args)
                    tool_results.append({"tool": tool_name, "args": tool_args, "result": result})
                    log_remediation(alert_id, tool_name, tool_args, result, result.get("success", True))

                    sensor_changes = result.get("sensor_changes", {})
                    await self._cb(websocket_callback, {
                        "type": "tool_result",
                        "alert_id": alert_id,
                        "tool": tool_name,
                        "sensor_id": tool_args.get("motor_id") or tool_args.get("unit_id",""),
                        "sensor_changes": sensor_changes,
                        "message": f"✅ {tool_name}: {result.get('message', str(result)[:100])}"
                    })

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result)
                    })

        except Exception as e:
            # ── API error: log it and escalate ────────────────────
            error_msg = str(e)
            await self._cb(websocket_callback, {
                "type": "agent_error",
                "alert_id": alert_id,
                "message": f"❌ Agent error: {error_msg[:200]}"
            })
            # Still save the incident so it shows in dashboard
            log_decision(
                alert_id=alert_id,
                reasoning=f"Agent error: {error_msg}",
                confidence=0.0,
                action_taken="none",
                action_result={"error": error_msg},
                escalated=True
            )
            update_incident_status(alert_id, "error")
            save_report(alert_id, f"# Error Report\nAgent failed: {error_msg}")
            await self._cb(websocket_callback, {
                "type": "agent_complete",
                "alert_id": alert_id,
                "escalated": True,
                "confidence": 0.0,
                "message": f"⚠️ Incident {alert_id} saved — agent encountered API error"
            })
            return {"alert_id": alert_id, "status": "error", "error": error_msg}

        # ── HARD ENFORCEMENT: never allow "resolved" with zero real actions ──
        # Diagnostic-only tools don't count as an action.
        DIAGNOSTIC_ONLY_TOOLS = {"get_sensor_history"}
        # Only these two force a human decision point. create_maintenance_work_order
        # is just scheduling paperwork for a technician's normal rounds — it does NOT
        # mean the situation needs a human right now, so it no longer forces escalation.
        ESCALATION_TOOLS      = {"escalate_to_operator", "emergency_shutdown"}

        action_tools_called = [
            t["tool"] for t in tool_results if t["tool"] not in DIAGNOSTIC_ONLY_TOOLS
        ]

        if not action_tools_called:
            # The model analyzed but took no action — this is never allowed.
            # Force a real decision: escalate, since we have no evidence a fix is safe to apply.
            await self._cb(websocket_callback, {
                "type": "agent_reasoning",
                "alert_id": alert_id,
                "message": "⚠️ No remediation action was taken — forcing escalation to a human operator."
            })
            forced_result = await execute_tool("escalate_to_operator", {
                "alert_id": alert_id,
                "sensor_id": sensor_id,
                "diagnosis": final_text[:300] or "Agent analyzed the alert but did not select a remediation action.",
                "actions_taken": [],
                "recommended_next_steps": ["Manual review required — agent did not act automatically."],
                "urgency": "high",
            })
            tool_results.append({"tool": "escalate_to_operator", "args": {"alert_id": alert_id}, "result": forced_result})
            action_tools_called = ["escalate_to_operator"]

        # ── parse final summary ────────────────────────────────────
        parsed     = self._parse_summary(final_text)
        confidence = parsed.get("confidence", 0.5)

        # Escalated if ANY of these are true:
        #  1. The model itself said ESCALATED_TO_HUMAN
        #  2. An escalation-type tool was called (always requires human follow-through)
        #  3. Confidence is below threshold — even if a remediation tool was called,
        #     low confidence means we cannot trust the fix was correct, so a human
        #     must still review it. This was the bug: a 75% confidence fix was being
        #     marked AUTO-RESOLVED because only tool type was checked, not confidence.
        below_threshold = confidence < CONFIDENCE_THRESHOLD
        escalated = (
            parsed.get("outcome", "").upper() == "ESCALATED_TO_HUMAN"
            or any(t["tool"] in ESCALATION_TOOLS for t in tool_results)
            or below_threshold
        )

        # ── DETERMINISTIC BACKSTOP: don't trust the model's stated confidence alone ──
        # The model is supposed to raise its confidence after seeing a fix verified by
        # sensor_changes, but LLMs are inconsistent about actually doing this. Instead of
        # relying on that, check the real numbers ourselves against the real THRESHOLDS.
        from backend.tools.sensor_state import THRESHOLDS as SENSOR_THRESHOLDS
        LOW_ALERT_METRICS = {"flow_m3s", "speed_m_s"}  # for these, LOW is bad, so "fixed" means value >= threshold

        def _readings_verified_safe() -> bool:
            thresh = SENSOR_THRESHOLDS.get(sensor_id, {})
            saw_any_change = False
            for t in tool_results:
                changes = (t.get("result") or {}).get("sensor_changes") or {}
                for metric, delta in changes.items():
                    to_val = delta.get("to") if isinstance(delta, dict) else None
                    if to_val is None or metric not in thresh:
                        continue
                    saw_any_change = True
                    is_low_alert = metric in LOW_ALERT_METRICS
                    still_bad = (to_val < thresh[metric]) if is_low_alert else (to_val > thresh[metric])
                    if still_bad:
                        return False  # at least one reading never actually came back in range
            return saw_any_change  # True only if we saw real changes AND none were still bad

        # Rule 3/4 exemption: if the ONLY action taken was a maintenance ticket (mechanical
        # fault correctly handled per Rule 4), that is ALREADY a valid AUTO_RESOLVED outcome
        # by design — there is no remote reading to verify, so the confidence gate shouldn't
        # apply to it at all.
        TICKET_ONLY_ACTIONS = {"create_maintenance_work_order"}
        ticket_only_resolution = bool(action_tools_called) and all(
            t in TICKET_ONLY_ACTIONS for t in action_tools_called
        )

        model_chose_escalation = (
            parsed.get("outcome", "").upper() == "ESCALATED_TO_HUMAN"
            or any(t["tool"] in ESCALATION_TOOLS for t in tool_results)
        )
        verified_fix = _readings_verified_safe()
        if (ticket_only_resolution or verified_fix) and not model_chose_escalation:
            below_threshold = False
            escalated = False

        if below_threshold and not any(t["tool"] in ESCALATION_TOOLS for t in tool_results):
            # A remediation tool WAS called, but confidence didn't meet the bar.
            # Notify the operator that an action was taken under uncertainty — they
            # should verify it, not just be told "resolved".
            verify_result = await execute_tool("escalate_to_operator", {
                "alert_id": alert_id,
                "sensor_id": sensor_id,
                "diagnosis": (parsed.get("root_cause") or final_text[:300]),
                "actions_taken": action_tools_called,
                "recommended_next_steps": [
                    f"Confidence was {confidence:.0%}, below the {CONFIDENCE_THRESHOLD:.0%} threshold — verify the remediation actually fixed the issue."
                ],
                "urgency": "medium",
            })
            tool_results.append({"tool": "escalate_to_operator", "args": {"alert_id": alert_id}, "result": verify_result})
            action_tools_called.append("escalate_to_operator")

        log_decision(
            alert_id=alert_id,
            reasoning=final_text,
            confidence=confidence,
            action_taken=json.dumps(action_tools_called),
            action_result={"tools": tool_results, "summary": parsed},
            escalated=escalated
        )

        # generate report
        report = self._build_report(alert_id, alert, tool_results, final_text, parsed)
        save_report(alert_id, report)
        update_incident_status(alert_id, "escalated" if escalated else "resolved")

        action_summary = ", ".join(action_tools_called)
        reason_tag = " (low confidence)" if below_threshold and escalated else ""
        status_msg = (
            f"🚨 ESCALATED{reason_tag} — confidence {confidence:.0%} — action: {action_summary}"
            if escalated else
            f"✅ AUTO-RESOLVED — confidence {confidence:.0%} — action: {action_summary}"
        )
        await self._cb(websocket_callback, {
            "type": "agent_complete",
            "alert_id": alert_id,
            "escalated": escalated,
            "confidence": confidence,
            "message": status_msg
        })

        return {
            "alert_id":     alert_id,
            "status":       "escalated" if escalated else "resolved",
            "confidence":   confidence,
            "escalated":    escalated,
            "tool_results": tool_results,
            "summary":      parsed,
        }

    # ── helpers ────────────────────────────────────────────────────
    async def _cb(self, fn, data):
        if fn:
            try:
                await fn(data)
            except Exception:
                pass

    def _parse_summary(self, text: str) -> dict:
        result = {"root_cause": "Analysis complete", "confidence": 0.75,
                  "actions_taken": [], "outcome": "AUTO_RESOLVED", "next_steps": ""}
        if not text:
            return result
        for line in text.split("\n"):
            if "ROOT_CAUSE:" in line:
                result["root_cause"] = line.split("ROOT_CAUSE:", 1)[-1].strip()
            elif "CONFIDENCE:" in line:
                try:
                    v = line.split("CONFIDENCE:", 1)[-1].strip().replace("%","")
                    val = float(v)
                    result["confidence"] = val / 100 if val > 1 else val
                except Exception:
                    pass
            elif "OUTCOME:" in line:
                result["outcome"] = line.split("OUTCOME:", 1)[-1].strip()
            elif "NEXT_STEPS:" in line:
                result["next_steps"] = line.split("NEXT_STEPS:", 1)[-1].strip()
        return result

    def _build_report(self, alert_id, alert, tool_results, reasoning, summary) -> str:
        # De-duplicate: if the same tool was called multiple times with the same outcome
        # message, show it once with a count, instead of repeating the full line N times.
        seen = {}
        for t in tool_results:
            key = (t['tool'], t['result'].get('message', 'done'))
            seen[key] = seen.get(key, 0) + 1
        lines = []
        for (tool, msg), count in seen.items():
            suffix = f" (×{count})" if count > 1 else ""
            lines.append(f"- {tool}{suffix}: {msg}")
        tools_used = "\n".join(lines)

        return f"""# Incident Report — {alert_id}
**Generated:** {datetime.utcnow().isoformat()}

## Alert
- Sensor: {alert.get('sensor_id')}
- Type: {alert.get('alert_type')}
- Severity: {alert.get('severity','').upper()}

## Root Cause
{summary.get('root_cause','N/A')}

## Agent Confidence
{summary.get('confidence', 0):.0%}

## Actions Taken
{tools_used or 'None'}

## Outcome
{summary.get('outcome','N/A')}

## Next Steps
{summary.get('next_steps','N/A')}

## Agent Reasoning
{reasoning[:1000]}
"""
