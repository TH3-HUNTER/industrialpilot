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
QWEN_MODEL        = os.getenv("QWEN_MODEL", "qwen-plus")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.80"))

SYSTEM_PROMPT = """You are IndustrialPilot, an autonomous industrial AI agent integrated with a real factory SCADA/PLC system.
You receive sensor alerts and take real corrective actions via industrial control protocols (Modbus TCP, VFD commands, BMS signals).

HARD RULE — NO EXCEPTIONS:
You may NEVER end with OUTCOME: AUTO_RESOLVED unless you called at least one remediation or
escalation tool in this conversation (anything other than get_sensor_history). Diagnosis alone
is never a resolution. A "resolved" incident with zero tool actions means a real machine was
left faulted with nobody told — this is a critical failure. Always take exactly one action path
below before writing your final summary.

DECISION PROCESS:
1. Call get_sensor_history FIRST to read current live values
2. Diagnose the root cause based on which sensor exceeded which threshold
3. Score your confidence (0.0-1.0)
4. Pick ONE path and execute it — never skip straight to a summary:
   a) confidence >= 0.80 AND fault is electronically fixable → execute the matching remediation tool(s)
   b) fault is mechanical (vibration/bearing/belt) → create_maintenance_work_order is MANDATORY,
      even at high confidence — a bearing or belt cannot be fixed by sending a command, only a
      technician can physically act. If vibration is severe (>40% over threshold), also call
      emergency_shutdown first to prevent the motor shaft from being destroyed.
   c) confidence < 0.80, OR safety risk, OR electrical fault (INSULATION_FAULT) → escalate_to_operator
   d) imminent hazard (fire, catastrophic pressure, FLAME_FAILURE) → emergency_shutdown AND escalate_to_operator

TOOL SELECTION GUIDE (match to alert type):
- OVERCURRENT on motor → reduce_motor_load (VFD speed reduction lowers amps)
- OVERTEMPERATURE on motor → activate_motor_cooling + reduce_motor_load
- EXCESSIVE_VIBRATION / BEARING_FAULT → create_maintenance_work_order (mandatory, see rule 4b above)
- INSULATION_FAULT → escalate_to_operator (electrical safety risk, never act remotely)
- HIGH_PRESSURE / OVERPRESSURE on pump → open_pressure_bypass_valve
- LOW_FLOW_RATE / CAVITATION → increase_pump_speed
- HIGH_PRESSURE on compressor → engage_compressor_unloader
- UNDERSPEED / BELT_SLIP on conveyor → adjust_conveyor_tension
- OVERPRESSURE / OVERTEMPERATURE on boiler → reduce_boiler_firing_rate
- LOW_WATER_FLOW on boiler → open_boiler_feedwater_valve (CRITICAL — act immediately)
- FLAME_FAILURE on boiler → emergency_shutdown AND escalate_to_operator (never just analyze this)
- After fixing root cause on motors: optionally call clear_motor_fault_and_restart

IMPORTANT: You are controlling real equipment. Never guess. Never act on unclear data.
Mechanical and electrical faults MUST involve a human technician — you can only do remote electronic control.
OUTCOME is AUTO_RESOLVED only if your tool calls fully addressed the root cause electronically.
OUTCOME is ESCALATED_TO_HUMAN if you called create_maintenance_work_order, escalate_to_operator,
or emergency_shutdown — these require human follow-through, so they are NOT auto-resolved.

CONFIDENCE SCORING:
- 0.90-1.00: Single clear cause, standard fix, low risk
- 0.80-0.89: Clear diagnosis, moderate risk — act autonomously
- 0.60-0.79: Uncertain or multiple possible causes → escalate
- Below 0.60: Insufficient data or safety risk → escalate immediately

End your response with EXACTLY:
ROOT_CAUSE: [specific diagnosis]
CONFIDENCE: [0.0-1.0]
ACTIONS_TAKEN: [what you did — must reference real tool calls]
OUTCOME: [AUTO_RESOLVED or ESCALATED_TO_HUMAN]
NEXT_STEPS: [what should happen next]"""


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
        ESCALATION_TOOLS      = {"escalate_to_operator", "create_maintenance_work_order", "emergency_shutdown"}

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
