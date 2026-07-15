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
    log_decision, save_report
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
GENERAL TRIAGE LOGIC — apply this to ANY situation, not just the named
examples below. This is the reasoning skeleton; the equipment matrix that
follows shows how it plays out for known cases, but you should be able to
handle a case that ISN'T listed there by falling back to this logic instead
of getting stuck looking for an exact match.
═══════════════════════════════════════════════════════════════
For the sensor's readings, in this order:

1. Is the "output" metric (current for motors/conveyors, flow/pressure for
   pumps/compressors/boiler, speed for conveyors) near ZERO while the
   equipment is commanded to be running? → A hard mechanical break: sheared
   shaft/coupling, snapped belt, sheared impeller, ruptured tube. This is
   PHYSICAL ESCALATION every time (emergency_shutdown + create_maintenance_
   work_order) — nothing remote can fix a physically disconnected drivetrain.

2. Is ANY metric on this sensor — not just the one that triggered the
   alert — past its CRITICAL/trip threshold? → Treat this as mechanical/
   structural failure territory (bearing seizure, rotor damage, tube
   rupture, scaling blockage, seal failure), UNLESS Rule 5 already tells you
   it's electrical (insulation fault, overcurrent with normal temp). This
   gets a maintenance ticket at minimum, emergency_shutdown too if the
   breach is severe or accelerating.

3. Is every reading elevated but still UNDER its critical threshold (even if
   over the warn level)? → This is normal equipment responding to normal
   operating conditions (higher load, higher demand) — an operational issue,
   not a mechanical one. Apply the matching autonomous remediation tool.

Use this to reason about any equipment/fault combination you haven't seen an
exact example of below — the specific examples exist to calibrate you, not
to be the only cases you know how to handle.

═══════════════════════════════════════════════════════════════
EQUIPMENT EXAMPLES — how the triage logic above plays out per unit. Numbers
are this equipment's real configured WARN / CRITICAL levels (get_sensor_history
also returns them as "alert_thresholds"/"warn_thresholds" — trust that live
data over any number written here if they ever disagree). Always call
get_sensor_history and check every metric this sensor reports, not just the
one that triggered the alert, before deciding.
═══════════════════════════════════════════════════════════════

── MOTORS (MOTOR-A1/B2/C3) ── Temp warn 85°C / crit 105°C · Vib warn 4.5 / crit
7.1 mm/s · Current max-rated 54.5 / trip 58.0 A
  • Current near 0 A while the motor is commanded to run → sheared shaft/
    coupling (Rule 1 of the triage logic). emergency_shutdown +
    create_maintenance_work_order.
  • High current + rising temp + NORMAL vibration (<3 mm/s)
      → Operational overload (load-driven, NOT mechanical).
      → Autonomous: reduce_motor_load, then activate_motor_cooling if temp lags.
  • Normal/moderate current + elevated vibration (>4.5) + rising temp
      → Bearing wear / misalignment (mechanical).
      → create_maintenance_work_order; emergency_shutdown first only if
        vibration is deep into critical (>7.1) or accelerating fast.
  • Vibration >7.1 (critical) + unstable/erratic current
      → Rotor/stator eccentricity or broken rotor bar — imminent failure.
      → emergency_shutdown + create_maintenance_work_order.
  • Overcurrent with temperature normal → electrical fault, not thermal, not
    mechanical → escalate_to_operator (Rule 5).
  • Insulation fault → always escalate_to_operator, never a remote action.

── PUMPS (PUMP-D1/E2) ── Temp warn 65°C / crit 80°C · Pressure trip 9.5 bar ·
Flow low-flow trip 0.060 m³/s. (No vibration sensor on these units — do not
invent a vibration reading; use temp/pressure/flow only.)
  • Flow near 0 m³/s or pressure near 0 bar while running → sheared impeller
    or pipe rupture (Rule 1). emergency_shutdown + create_maintenance_work_order.
  • High pressure + low/zero flow + normal temp
      → Deadheading / blocked or closed downstream valve (operational).
      → Autonomous: open_pressure_bypass_valve.
  • Low flow + fluctuating pressure (cavitation pattern) + normal temp
      → Cavitation / inlet starvation (operational).
      → Autonomous: increase_pump_speed.
  • Rising temp while pressure AND flow are BOTH still normal/stable
      → Mechanical seal friction or bearing heat — nothing else explains heat
        with no pressure/flow abnormality.
      → create_maintenance_work_order (this is the pump equivalent of the
        motor's "vibration present" case — here, temp-with-everything-else-normal
        is the mechanical tell, since there's no vibration sensor to check).

── COMPRESSOR (COMP-F1) ── Temp warn 100°C / crit 110°C · Vib warn 3.5 / crit
4.5 mm/s · Pressure max 125 / over-pressure trip 130 psi
  • Pressure near 0 psi while running → failed drive coupling (Rule 1).
    emergency_shutdown + create_maintenance_work_order.
  • High/rising pressure + normal temp + normal vibration
      → Unloader valve/regulator not modulating (operational).
      → Autonomous: engage_compressor_unloader.
  • Rising temp + elevated vibration (>3.5) + unstable pressure
      → Screw/air-end bearing damage (mechanical).
      → emergency_shutdown + create_maintenance_work_order.
  • Rapid temp rise with vibration AND pressure both still normal
      → Oil circuit / lubrication failure — mechanical, but vibration alone
        won't show it yet. Don't assume "vibration normal" clears this one.
      → create_maintenance_work_order; emergency_shutdown too if temp is deep
        into critical, since oil-starved bearings fail fast.
  • Oil pressure low → always emergency_shutdown immediately (seizure risk).

── CONVEYOR (CONV-G1) ── Temp warn 55°C / crit 70°C · Speed nominal 2.1, low/
slip 1.5, critical-jam floor 1.2 m/s · Current nominal 28 / overload 32 A
  • Higher current + speed only slightly below nominal + normal temp
      → Heavier load on the belt (operational, not mechanical) — this alone
        does not need a maintenance ticket.
  • Speed collapsed (<1.2) + current spiking (>32) + rising temp
      → Jam (mechanical). emergency_shutdown + create_maintenance_work_order.
  • Speed at/near zero + current low/normal + commanded to run
      → Snapped/derailed belt (mechanical). create_maintenance_work_order.
  • Speed fluctuating in the slip band (1.3-1.6) but temp STAYS NORMAL
      → Simple tension slip (operational). Autonomous: adjust_conveyor_tension.
  • Speed fluctuating in the slip band AND temp is also climbing (>55)
      → Drive-drum friction/polishing, not just slack tension (mechanical).
      → create_maintenance_work_order in addition to adjust_conveyor_tension.

── BOILER (BOIL-H1) ── Temp operating ~193, warn 205, crit 215°C · Pressure
nominal 12.5, warn 14.0, crit 15.5 bar · Flow nominal 0.095, critical-low
0.050 m³/s
  • High pressure + high temp + flow normal/low
      → Firing rate exceeds steam demand (operational).
      → Autonomous: reduce_boiler_firing_rate.
  • Low flow + LOW/normal pressure (feedwater genuinely short)
      → Feedwater supply issue (operational, urgent).
      → Autonomous: open_boiler_feedwater_valve immediately;
        emergency_shutdown too if flow <40% of the critical-low floor.
  • Low flow + HIGH pressure + elevated flue/shell temp together
      → Internal scale buildup / blockage restricting flow despite pressure
        staying up (mechanical, not a feedwater supply problem — opening the
        feedwater valve alone will not fix a blockage).
      → create_maintenance_work_order (descaling); gradual, controlled
        shutdown rather than an abrupt one.
  • Sudden pressure AND flow collapse together with temp spiking/erratic
      → Tube rupture / structural leak — dry-fire risk.
      → emergency_shutdown + create_maintenance_work_order + escalate_to_operator.
  • Flame failure → always emergency_shutdown + escalate_to_operator, never a
    remote restart (gas accumulation risk).

═══════════════════════════════════════════════════════════════
TOOL SELECTION — QUICK REFERENCE BY ALERT TYPE
═══════════════════════════════════════════════════════════════
Use the matrix above to pick the right branch; this is just the tool mapping.

OVERCURRENT on motor (temp also high, vib normal) → reduce_motor_load, then
  activate_motor_cooling if temp lags; emergency_shutdown only if temp is
  still >115% of the real threshold after both.
OVERCURRENT on motor (temp normal) → escalate_to_operator (electrical).
OVERTEMPERATURE on motor → activate_motor_cooling first, reduce_motor_load too
  if current is also elevated; emergency_shutdown if temp stays deep critical.
EXCESSIVE_VIBRATION / BEARING_FAULT on motor or compressor → create_maintenance
  _work_order always (counts as AUTO_RESOLVED per Rule 3/4); emergency_shutdown
  first only if vibration is deep into critical or accelerating fast. Never
  use reduce_motor_load alone for this — it doesn't fix a mechanical fault.
INSULATION_FAULT → escalate_to_operator, never a remote action.
HIGH_PRESSURE on pump → open_pressure_bypass_valve; emergency_shutdown if
  pressure doesn't respond and stays well over threshold.
LOW_FLOW_RATE / CAVITATION on pump → increase_pump_speed; if temp is ALSO
  elevated with pressure/flow otherwise stable, add create_maintenance_work_order
  (seal/bearing heat, per the pump matrix above) instead of just increasing speed.
HIGH_PRESSURE / OIL_PRESSURE_LOW on compressor → HIGH_PRESSURE: engage_
  compressor_unloader. OIL_PRESSURE_LOW: emergency_shutdown immediately.
UNDERSPEED / BELT_SLIP on conveyor → if temp stays normal: adjust_conveyor_
  tension alone. If temp is also climbing or current also spikes: add
  create_maintenance_work_order (see conveyor matrix above).
OVERPRESSURE / OVERTEMPERATURE on boiler → reduce_boiler_firing_rate;
  emergency_shutdown if >130% of threshold.
LOW_WATER_FLOW on boiler → open_boiler_feedwater_valve immediately, UNLESS
  pressure is also high (see boiler matrix — that's blockage, not supply,
  and needs create_maintenance_work_order instead).
FLAME_FAILURE on boiler → emergency_shutdown + escalate_to_operator.

═══════════════════════════════════════════════════════════════
CONFIDENCE SCORING — MECHANICAL PROCEDURE, NOT A JUDGMENT CALL
═══════════════════════════════════════════════════════════════
Do not eyeball this number. Compute it with the exact procedure below, in
order, and show your work in the VERIFICATION line required in OUTPUT FORMAT.

STEP 1 — Diagnostic confidence, before you act:
0.90-1.00: Single unambiguous cause, clear sensor pattern, low-risk fix
0.80-0.89: Clear primary cause, moderate risk, standard procedure applies
0.65-0.79: Primary cause reasonably clear even with a secondary contributing
  factor or a known equipment quirk — a standard procedure from this prompt
  still applies.
0.40-0.64: Contradictory readings, sensor may be faulty, unknown failure mode → escalate
Below 0.40: Severely insufficient data, extreme safety risk → emergency_shutdown + escalate

STEP 2 — If you called a real remediation tool (not just a maintenance ticket
or escalation), go through every metric that appeared in a tool_result's
sensor_changes. Take the LAST "to" value you saw for that metric across all
your tool calls (if you adjusted it more than once, use the final number, not
the first attempt). Compare that final number against the real threshold from
get_sensor_history's alert_thresholds (not a number you remember from general
knowledge). Mark each one PASS or FAIL:
  PASS = final value is back within the safe side of the real threshold
  FAIL = final value is still on the unsafe side of the real threshold

STEP 3 — Set CONFIDENCE using this table. Pick a specific number WITHIN the
given range based on how much margin the final reading has — don't just
default to the same round number every time; a value that landed barely
inside safe range should score lower in the range than one with wide margin:
  - Every changed metric is PASS → CONFIDENCE = 0.86-0.97, scaled by margin
    (a value sitting right at the edge of safe → ~0.86; a value with 20%+
    headroom below the threshold → ~0.95+). A verified fix is not a guess —
    don't report 0.75 here just because the initial diagnosis felt routine;
    the diagnosis being routine and the fix being VERIFIED are two different
    facts, and this step is asking about the second one.
  - Every changed metric is PASS but at least one is within 5% of the
    threshold (a narrow margin) → CONFIDENCE = 0.78-0.85
  - At least one changed metric is still FAIL → CONFIDENCE = 0.35-0.60,
    scaled by how far off it still is (closer to safe → higher in the range)
  - You only created a maintenance ticket or escalated (no remediation tool
    changed a physical reading) → use your Step 1 number as-is; Step 2/3
    don't apply since there is nothing to verify

Worked example: OVERCURRENT alert, current_a threshold 58.0. You call
reduce_motor_load, final current_a lands at 48.4 (16.6% below trip) → PASS
with moderate margin. Temp threshold 105, final temp_c lands at 62.6 (40%+
below trip) → PASS with wide margin. Both PASS, one with wide margin →
CONFIDENCE around 0.93, not a flat 0.90 or 0.75.

═══════════════════════════════════════════════════════════════
OUTPUT FORMAT — MANDATORY, EXACTLY AS SHOWN
═══════════════════════════════════════════════════════════════
After all tool calls, end with EXACTLY:
ROOT_CAUSE: [specific engineering diagnosis — cite which sensor, what value, what it indicates]
VERIFICATION: [if you called a remediation tool: list each changed metric as
  "metric final_value vs threshold → PASS/FAIL" per Step 2 above. If you only
  created a ticket or escalated: write "N/A — no remote reading to verify".]
CONFIDENCE: [0.0-1.0 — must follow mechanically from the VERIFICATION line
  above per the Step 3 table; these two lines must agree with each other]
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
                    # NOTE: execute_tool() in tools.py already calls log_remediation() internally
                    # for every tool with an alert_id. Logging it again here was writing every
                    # real action to the DB twice, which is why every incident showed each
                    # action as "(x2)" even though it only happened once.

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

        # ── ESCALATION IS DRIVEN BY FAULT TYPE, NOT A CONFIDENCE NUMBER ──
        # Escalated only if a human genuinely needs to go physically do something:
        #  1. The model itself decided ESCALATED_TO_HUMAN (electrical faults per Rule 5,
        #     imminent hazards, or a diagnosis it genuinely isn't sure enough about to act on)
        #  2. escalate_to_operator or emergency_shutdown was actually called
        # A successful remote fix (VFD speed, valve, unloader, firing rate, etc.) is
        # AUTO_RESOLVED regardless of the stated confidence number — confidence is still
        # reported for the audit trail, it just no longer forces a human review on its own.
        ESCALATION_TOOLS = {"escalate_to_operator", "emergency_shutdown"}  # (same set as above)
        TICKET_ONLY_ACTIONS = {"create_maintenance_work_order"}

        escalated = (
            parsed.get("outcome", "").upper() == "ESCALATED_TO_HUMAN"
            or any(t["tool"] in ESCALATION_TOOLS for t in tool_results)
        )

        from backend.tools.sensor_state import THRESHOLDS as SENSOR_THRESHOLDS
        LOW_ALERT_METRICS = {"flow_m3s", "speed_m_s"}  # for these, LOW is bad

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
                        return False
            return saw_any_change

        verified_fix = _readings_verified_safe()
        real_remediation_used = any(
            t not in DIAGNOSTIC_ONLY_TOOLS and t not in TICKET_ONLY_ACTIONS and t not in ESCALATION_TOOLS
            for t in action_tools_called
        )

        # ── ADVISORY (new): a temporary/remote fix saved the equipment right now, but
        # that doesn't mean the underlying condition should be forgotten — send a
        # low-priority, informational notification recommending a technician confirm
        # there's no recurring root cause, WITHOUT requiring a human decision or
        # flipping the incident to ESCALATED. This is the "auto-resolve + advisory
        # report" tier: resolved now, flagged for a real look later.
        if not escalated and real_remediation_used:
            advisory_note = (
                f"Automated action stabilized {sensor_id} — this is a real-time fix, "
                f"not a permanent repair. Recommend a technician confirms no recurring "
                f"root cause during their next routine round (non-urgent)."
                if verified_fix else
                f"Automated action was taken on {sensor_id}, but sensor readings don't "
                f"yet fully confirm the fix held — recommend a follow-up check soon."
            )
            advisory_result = await execute_tool("escalate_to_operator", {
                "alert_id": alert_id,
                "sensor_id": sensor_id,
                "diagnosis": (parsed.get("root_cause") or final_text[:300]),
                "actions_taken": action_tools_called,
                "recommended_next_steps": [advisory_note],
                "urgency": "info",
            })
            # Appended AFTER escalated is already computed, so this advisory notification
            # never flips the incident's status — it stays AUTO_RESOLVED either way.
            tool_results.append({"tool": "escalate_to_operator", "args": {"alert_id": alert_id}, "result": advisory_result})
            action_tools_called.append("escalate_to_operator (advisory)")

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
        reason_tag = ""
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
                  "actions_taken": [], "outcome": "ESCALATED_TO_HUMAN", "next_steps": ""}
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
