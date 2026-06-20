import random, uuid
from datetime import datetime

SENSORS = {
  'MOTOR-A1':{'type':'motor',      'location':'Production Line 1 — Zone A','thresholds':{'temp_c':85,'vib_mm_s':7.5,'current_a':45}},
  'MOTOR-B2':{'type':'motor',      'location':'Production Line 2 — Zone B','thresholds':{'temp_c':85,'vib_mm_s':7.5,'current_a':45}},
  'MOTOR-C3':{'type':'motor',      'location':'Production Line 3 — Zone C','thresholds':{'temp_c':85,'vib_mm_s':7.5,'current_a':45}},
  'PUMP-D1': {'type':'pump',       'location':'Cooling System — Zone D',   'thresholds':{'temp_c':70,'pressure_bar':12,'flow_m3s':0.05}},
  'PUMP-E2': {'type':'pump',       'location':'Hydraulics — Zone E',       'thresholds':{'temp_c':70,'pressure_bar':12,'flow_m3s':0.05}},
  'COMP-F1': {'type':'compressor', 'location':'Air Supply — Zone F',       'thresholds':{'temp_c':95,'pressure_psi':150,'vib_mm_s':5.0}},
  'CONV-G1': {'type':'conveyor',   'location':'Assembly Line — Zone G',    'thresholds':{'temp_c':60,'speed_m_s':1.2,'current_a':30}},
  'BOIL-H1': {'type':'boiler',     'location':'Steam Plant — Zone H',      'thresholds':{'temp_c':180,'pressure_bar':16,'flow_m3s':0.08}},
}

SCENARIOS = {
  'motor':[
    {'alert_type':'OVERTEMPERATURE',     'metric':'temp_c',    'unit':'°C',   'mul':1.25,'sev':'high'},
    {'alert_type':'EXCESSIVE_VIBRATION', 'metric':'vib_mm_s',  'unit':'mm/s', 'mul':2.8, 'sev':'critical'},
    {'alert_type':'OVERCURRENT',         'metric':'current_a', 'unit':'A',    'mul':1.3, 'sev':'high'},
    {'alert_type':'BEARING_FAULT',       'metric':'vib_mm_s',  'unit':'mm/s', 'mul':3.5, 'sev':'critical'},
    {'alert_type':'INSULATION_FAULT',    'metric':'current_a', 'unit':'A',    'mul':0.3, 'sev':'high'},
  ],
  'pump':[
    {'alert_type':'HIGH_PRESSURE',   'metric':'pressure_bar','unit':'bar',  'mul':1.45,'sev':'critical'},
    {'alert_type':'LOW_FLOW_RATE',   'metric':'flow_m3s',   'unit':'m³/s', 'mul':0.35,'sev':'high'},
    {'alert_type':'OVERTEMPERATURE', 'metric':'temp_c',     'unit':'°C',   'mul':1.3, 'sev':'high'},
    {'alert_type':'CAVITATION',      'metric':'pressure_bar','unit':'bar', 'mul':0.4, 'sev':'high'},
  ],
  'compressor':[
    {'alert_type':'HIGH_PRESSURE',       'metric':'pressure_psi','unit':'psi',  'mul':1.35,'sev':'critical'},
    {'alert_type':'OVERTEMPERATURE',     'metric':'temp_c',      'unit':'°C',   'mul':1.2, 'sev':'high'},
    {'alert_type':'EXCESSIVE_VIBRATION', 'metric':'vib_mm_s',    'unit':'mm/s', 'mul':3.2, 'sev':'critical'},
    {'alert_type':'OIL_PRESSURE_LOW',    'metric':'pressure_psi','unit':'psi',  'mul':0.45,'sev':'high'},
  ],
  'conveyor':[
    {'alert_type':'UNDERSPEED',      'metric':'speed_m_s', 'unit':'m/s','mul':0.45,'sev':'medium'},
    {'alert_type':'OVERTEMPERATURE', 'metric':'temp_c',    'unit':'°C', 'mul':1.4, 'sev':'high'},
    {'alert_type':'OVERCURRENT',     'metric':'current_a', 'unit':'A',  'mul':1.55,'sev':'high'},
    {'alert_type':'BELT_SLIP',       'metric':'speed_m_s', 'unit':'m/s','mul':0.6, 'sev':'medium'},
  ],
  'boiler':[
    {'alert_type':'OVERPRESSURE',    'metric':'pressure_bar','unit':'bar', 'mul':1.3, 'sev':'critical'},
    {'alert_type':'OVERTEMPERATURE', 'metric':'temp_c',     'unit':'°C',  'mul':1.15,'sev':'critical'},
    {'alert_type':'LOW_WATER_FLOW',  'metric':'flow_m3s',   'unit':'m³/s','mul':0.3, 'sev':'high'},
    {'alert_type':'FLAME_FAILURE',   'metric':'temp_c',     'unit':'°C',  'mul':0.4, 'sev':'critical'},
  ],
}

# All scenarios flattened for cross-sensor type matching
ALL_SCENARIOS = [s for scens in SCENARIOS.values() for s in scens]

def generate_alert(sensor_id=None, alert_type_override=None, severity_override=None):
    if not sensor_id or sensor_id not in SENSORS:
        sensor_id = random.choice(list(SENSORS.keys()))
    s     = SENSORS[sensor_id]
    scens = SCENARIOS.get(s['type'], SCENARIOS['motor'])

    if alert_type_override:
        # First try to find it in this sensor's own scenarios
        scen = next((x for x in scens if x['alert_type'] == alert_type_override), None)
        if not scen:
            # Fallback: find it globally and adapt metric/value to this sensor's thresholds
            scen_global = next((x for x in ALL_SCENARIOS if x['alert_type'] == alert_type_override), None)
            if scen_global:
                # Use the sensor's first threshold metric as fallback
                first_metric = list(s['thresholds'].keys())[0]
                scen = {**scen_global, 'metric': first_metric, 'unit': '°C' if 'temp' in first_metric else 'units'}
            else:
                scen = random.choice(scens)
    else:
        scen = random.choice(scens)

    metric = scen['metric']
    thresh = s['thresholds'].get(metric, list(s['thresholds'].values())[0])
    value  = round(thresh * scen['mul'] * random.uniform(0.95, 1.1), 2)
    sev    = severity_override or scen['sev']

    return {
        'alert_id':   f"ALT-{uuid.uuid4().hex[:8].upper()}",
        'sensor_id':  sensor_id,
        'alert_type': scen['alert_type'],
        'severity':   sev,
        'data': {
            'metric':         metric,
            'value':          value,
            'unit':           scen['unit'],
            'threshold':      thresh,
            'location':       s['location'],
            'equipment_type': s['type'],
        },
        'timestamp': datetime.utcnow().isoformat(),
    }

def get_sensor_list():
    return [{'sensor_id': k, **v} for k, v in SENSORS.items()]
