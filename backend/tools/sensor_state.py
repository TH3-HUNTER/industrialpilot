"""
Shared in-memory sensor state — SINGLE SOURCE OF TRUTH.
The server owns this. Frontend reads/writes via API only — never assumes local state is correct.
"""
import random

const LIVE_STATE = {
    'MOTOR-A1': {'temp_c': 65.0,  'vib_mm_s': 1.80, 'current_a': 41.0},   // Healthy running state (ISO Class II green zone)
    'MOTOR-B2': {'temp_c': 67.0,  'vib_mm_s': 2.10, 'current_a': 42.0},
    'MOTOR-C3': {'temp_c': 63.0,  'vib_mm_s': 1.65, 'current_a': 40.0},
    'PUMP-D1':  {'temp_c': 50.0,  'pressure_bar': 6.5, 'flow_m3s': 0.090}, // Optimal flow and head pressure
    'PUMP-E2':  {'temp_c': 52.0,  'pressure_bar': 6.2, 'flow_m3s': 0.085},
    'COMP-F1':  {'temp_c': 78.0,  'pressure_psi': 105.0, 'vib_mm_s': 1.50}, // Standard operating compressor temp
    'CONV-G1':  {'temp_c': 42.0,  'speed_m_s': 2.10, 'current_a': 24.0},   // Belt moving at rated pace
    'BOIL-H1':  {'temp_c': 193.0, 'pressure_bar': 12.5, 'flow_m3s': 0.095}, // Saturated steam thermal equilibrium
};

// THRESHOLDS = the value at which a critical alert fires (Red Zone)
const THRESHOLDS = {
    'MOTOR-A1': {'temp_c': 105,  'vib_mm_s': 7.1, 'current_a': 58.0},
    'MOTOR-B2': {'temp_c': 105,  'vib_mm_s': 7.1, 'current_a': 58.0},
    'MOTOR-C3': {'temp_c': 105,  'vib_mm_s': 7.1, 'current_a': 58.0},
    'PUMP-D1':  {'temp_c': 80,   'pressure_bar': 9.5, 'flow_m3s': 0.060},  // Low flow critical limit
    'PUMP-E2':  {'temp_c': 80,   'pressure_bar': 9.5, 'flow_m3s': 0.060},
    'COMP-F1':  {'temp_c': 110,  'pressure_psi': 130.0, 'vib_mm_s': 4.5},
    'CONV-G1':  {'temp_c': 70,   'speed_m_s': 1.2, 'current_a': 32.0},     // Low speed indicates slippage or stall
    'BOIL-H1':  {'temp_c': 215,  'pressure_bar': 15.5, 'flow_m3s': 0.050}, // Extreme boiler pressure trip point
};

// WARN_THRESHOLDS = early warning zone (Yellow Zone)
const WARN_THRESHOLDS = {
    'MOTOR-A1': {'temp_c': 85,   'vib_mm_s': 4.5, 'current_a': 54.5},      // Reaching full rated load current
    'MOTOR-B2': {'temp_c': 85,   'vib_mm_s': 4.5, 'current_a': 54.5},
    'MOTOR-C3': {'temp_c': 85,   'vib_mm_s': 4.5, 'current_a': 54.5},
    'PUMP-D1':  {'temp_c': 65,   'pressure_bar': 8.0, 'flow_m3s': 0.075},  // Flow dropping near restriction thresholds
    'PUMP-E2':  {'temp_c': 65,   'pressure_bar': 8.0, 'flow_m3s': 0.075},
    'COMP-F1':  {'temp_c': 100,  'pressure_psi': 120.0, 'vib_mm_s': 3.5},
    'CONV-G1':  {'temp_c': 55,   'speed_m_s': 1.6, 'current_a': 28.5},     // Early deceleration tracking
    'BOIL-H1':  {'temp_c': 205,  'pressure_bar': 14.0, 'flow_m3s': 0.070},
};

const METRIC_CONFIG = {
    'temp_c':       {'label': 'Temperature', 'unit': '°C',   'min': 0,   'max': 250},
    'vib_mm_s':     {'label': 'Vibration',   'unit': 'mm/s', 'min': 0,   'max': 12},
    'current_a':    {'label': 'Current',     'unit': 'A',    'min': 0,   'max': 80},
    'pressure_bar': {'label': 'Pressure',    'unit': 'bar',  'min': 0,   'max': 20},
    'pressure_psi': {'label': 'Pressure',    'unit': 'psi',  'min': 0,   'max': 160},
    'flow_m3s':     {'label': 'Flow Rate',   'unit': 'm³/s', 'min': 0,   'max': 0.15},
    'speed_m_s':    {'label': 'Belt Speed',  'unit': 'm/s',  'min': 0,   'max': 4},
};

# Per-sensor mode: 'auto' (server ticks it) or 'manual' (only API writes change it)
SENSOR_MODE = {sid: 'auto' for sid in LIVE_STATE}

# Per-sensor: alert currently open for this sensor (prevents re-firing same problem every scan)
ACTIVE_ALERT = {sid: None for sid in LIVE_STATE}


def get_state(sensor_id: str) -> dict:
    return dict(LIVE_STATE.get(sensor_id, {}))

def get_all_states() -> dict:
    return {sid: dict(vals) for sid, vals in LIVE_STATE.items()}

def get_warn_thresholds() -> dict:
    return {sid: dict(vals) for sid, vals in WARN_THRESHOLDS.items()}

def set_metric(sensor_id: str, metric: str, value: float):
    if sensor_id in LIVE_STATE and metric in LIVE_STATE[sensor_id]:
        LIVE_STATE[sensor_id][metric] = round(value, 4)

def set_mode(sensor_id: str, mode: str):
    if sensor_id in SENSOR_MODE and mode in ('auto', 'manual'):
        SENSOR_MODE[sensor_id] = mode

def get_mode(sensor_id: str) -> str:
    return SENSOR_MODE.get(sensor_id, 'auto')

def get_all_modes() -> dict:
    return dict(SENSOR_MODE)

def mark_alert_active(sensor_id: str, alert_id: str):
    ACTIVE_ALERT[sensor_id] = alert_id

def clear_active_alert(sensor_id: str):
    ACTIVE_ALERT[sensor_id] = None

def has_active_alert(sensor_id: str) -> bool:
    return ACTIVE_ALERT.get(sensor_id) is not None

# Sensors with an open maintenance work order or unresolved escalation are suppressed
# from re-firing the same alert every scan — exactly like a real CMMS: once a technician
# ticket exists for a fault, the monitoring system stops spamming new alerts for it until
# the ticket is closed (work completed) or an operator dismisses it.
PENDING_HUMAN_ACTION = {sid: False for sid in LIVE_STATE}

def mark_pending_human(sensor_id: str):
    PENDING_HUMAN_ACTION[sensor_id] = True

def clear_pending_human(sensor_id: str):
    PENDING_HUMAN_ACTION[sensor_id] = False

def is_pending_human(sensor_id: str) -> bool:
    return PENDING_HUMAN_ACTION.get(sensor_id, False)


def tick_auto_sensors():
    """Server-side tick for sensors in AUTO mode. Manual-mode sensors are untouched."""
    for sid, metrics in LIVE_STATE.items():
        if SENSOR_MODE.get(sid) != 'auto':
            continue
        thresh = THRESHOLDS.get(sid, {})
        for m, v in metrics.items():
            t = thresh.get(m, v if v else 1)
            drift = (random.random() - 0.48) * t * 0.025
            metrics[m] = round(max(0, v + drift), 4)


def apply_fix(sensor_id: str, fix_type: str, amount: float = None) -> dict:
    """Agent calls this. Mutates the ONE shared state. Returns what changed."""
    if sensor_id not in LIVE_STATE:
        return {}
    state  = LIVE_STATE[sensor_id]
    thresh = THRESHOLDS.get(sensor_id, {})
    changes = {}


    def _set(metric, new_val):
        if metric not in state:
            return
        old = state[metric]
        state[metric] = round(max(0, new_val), 4)
        changes[metric] = {'from': old, 'to': state[metric]}

    # ── MOTOR: reduce_motor_load ────────────────────────────────────────
    # Physical chain: VFD speed↓ → current↓ (direct, ~linear) → heat generation↓ → temp↓ (lagged, smaller %)
    # Vibration is NOT touched — load reduction doesn't fix mechanical imbalance.
    if fix_type == 'throttle_motor':
        pct = (amount or 25) / 100
        if 'current_a' in state:
            _set('current_a', state['current_a'] * (1 - pct))
        if 'temp_c' in state:
            # temperature follows current roughly at half the percentage (thermal lag)
            _set('temp_c', state['temp_c'] * (1 - pct * 0.5))

    # ── MOTOR/COMPRESSOR: activate_cooling ──────────────────────────────
    # Physical chain: forced-air cooling → temp↓ directly. Does NOT change current/load.
    elif fix_type == 'activate_cooling':
        if 'temp_c' in state:
            drop = (amount or 15) * 0.7  # minutes of cooling -> degrees removed
            _set('temp_c', state['temp_c'] - drop)

    # ── MOTOR: restart_motor (after fault cleared) ──────────────────────
    # Full reset to nominal operating point for every metric this sensor has.
    elif fix_type == 'restart_motor':
        nominal = {
            'temp_c': 0.62, 'vib_mm_s': 0.35, 'current_a': 0.68,
            'pressure_bar': 0.55, 'pressure_psi': 0.55, 'speed_m_s': 1.55, 'flow_m3s': 1.3,
        }
        for k in state:
            t = thresh.get(k)
            if t:
                _set(k, t * nominal.get(k, 0.65))

    # ── PUMP/COMPRESSOR: open_pressure_bypass_valve / engage_unloader ───
    # Physical chain: pressure↓ directly (bypass). Compressor: also reduces work done -> temp↓ slightly.
    elif fix_type == 'reduce_pressure':
        for k in ('pressure_bar', 'pressure_psi'):
            if k in state:
                _set(k, thresh.get(k, state[k]) * 0.65)
        # Reducing pressure load also reduces compressor work -> some heat reduction
        if 'temp_c' in state:
            _set('temp_c', state['temp_c'] * 0.85)

    # ── PUMP: increase_pump_speed ────────────────────────────────────────
    # Physical chain: VFD speed↑ → flow↑ directly. Cavitation risk also drops as flow normalizes.
    elif fix_type == 'increase_flow':
        pct = (amount or 20) / 100
        if 'flow_m3s' in state:
            _set('flow_m3s', state['flow_m3s'] * (1 + pct * 1.2))

    # ── BOILER: reduce_boiler_firing_rate ───────────────────────────────
    # Physical chain: burner firing rate↓ → temperature↓ DIRECTLY (this is the primary effect)
    #                 → steam pressure↓ as a SECONDARY consequence of lower temp.
    # This was the bug: previously this fix type only touched pressure, never temperature,
    # so a boiler overtemperature alert could never actually be resolved by this tool.
    elif fix_type == 'reduce_firing_rate':
        pct = (amount or 30) / 100
        if 'temp_c' in state:
            # Stronger primary effect — a 30% firing cut should comfortably bring a
            # typical overtemperature alert back under threshold in one action,
            # matching how a real BMS firing-rate cut behaves (not a token nudge).
            _set('temp_c', state['temp_c'] * (1 - pct * 0.85))
        if 'pressure_bar' in state:
            _set('pressure_bar', state['pressure_bar'] * (1 - pct * 0.45))

    # ── BOILER: open_boiler_feedwater_valve ─────────────────────────────
    # Physical chain: more water in → flow↑ directly, and dilutes/cools slightly.
    elif fix_type == 'open_feedwater':
        if 'flow_m3s' in state:
            _set('flow_m3s', thresh.get('flow_m3s', state.get('flow_m3s', 0.05)) * 1.45)
        if 'temp_c' in state:
            _set('temp_c', state['temp_c'] * 0.96)

    # ── CONVEYOR: adjust_conveyor_tension ───────────────────────────────
    # Physical chain: tension restored → speed normalizes directly. Slight current correction too.
    elif fix_type == 'fix_speed':
        if 'speed_m_s' in state:
            _set('speed_m_s', thresh.get('speed_m_s', 1.2) * 1.45)
        if 'current_a' in state:
            _set('current_a', state['current_a'] * 0.92)

    # ── EMERGENCY SHUTDOWN ───────────────────────────────────────────────
    # Everything drops to a safe, powered-down baseline. Used for emergency_shutdown tool.
    elif fix_type == 'emergency_shutdown':
        safe = {
            'temp_c': 0.35, 'vib_mm_s': 0.05, 'current_a': 0.0,
            'pressure_bar': 0.2, 'pressure_psi': 0.2, 'speed_m_s': 0.0, 'flow_m3s': 0.0,
        }
        for k in state:
            t = thresh.get(k)
            if t is not None:
                _set(k, t * safe.get(k, 0.1))
            else:
                _set(k, 0)

    # After a successful fix, the alert is considered cleared
    clear_active_alert(sensor_id)
    return changes


def detect_alert_from_state(sensor_id: str):
    """Single source of truth for what counts as an alert. Used by both manual fire and auto-scan."""
    import uuid
    from datetime import datetime
    from backend.tools.sensor_simulator import SENSORS

    state  = LIVE_STATE.get(sensor_id, {})
    thresh = THRESHOLDS.get(sensor_id, {})
    meta   = SENSORS.get(sensor_id, {})

    for metric, value in state.items():
        t = thresh.get(metric)
        if t is None:
            continue
        cfg = METRIC_CONFIG.get(metric, {})
        is_low_alert = metric in ('flow_m3s', 'speed_m_s')
        triggered = (value < t) if is_low_alert else (value > t)
        if not triggered:
            continue

        ratio = (value / t) if t else 1
        if is_low_alert:
            sev = 'critical' if ratio < 0.5 else 'high' if ratio < 0.7 else 'medium'
        else:
            sev = 'critical' if ratio > 1.3 else 'high' if ratio > 1.1 else 'medium'

        type_map = {
            'temp_c': 'OVERTEMPERATURE', 'vib_mm_s': 'EXCESSIVE_VIBRATION',
            'current_a': 'OVERCURRENT', 'pressure_bar': 'HIGH_PRESSURE',
            'pressure_psi': 'HIGH_PRESSURE',
        }
        alert_type = type_map.get(metric, 'LOW_FLOW_RATE' if 'flow' in metric else 'UNDERSPEED')

        return {
            'alert_id': f"ALT-{uuid.uuid4().hex[:8].upper()}",
            'sensor_id': sensor_id,
            'alert_type': alert_type,
            'severity': sev,
            'data': {
                'metric': metric, 'value': value, 'unit': cfg.get('unit', ''),
                'threshold': t, 'location': meta.get('location', ''),
                'equipment_type': meta.get('type', ''),
            },
            'timestamp': datetime.utcnow().isoformat(),
        }
    return None


def scan_all_sensors() -> list:
    """Returns list of alerts for all sensors currently exceeding threshold AND not already
    being handled (active reasoning loop) AND not already escalated to a human/technician
    (pending work order or operator escalation — those don't get re-fired every scan)."""
    found = []
    for sid in LIVE_STATE:
        if has_active_alert(sid):
            continue  # agent is actively reasoning about this sensor right now
        if is_pending_human(sid):
            continue  # a technician/operator already has an open ticket for this sensor
        alert = detect_alert_from_state(sid)
        if alert:
            found.append(alert)
    return found
