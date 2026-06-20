import sqlite3, json
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent.parent.parent / "data" / "industrialpilot.db"

def get_conn():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_id TEXT UNIQUE NOT NULL,
            sensor_id TEXT NOT NULL,
            alert_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'open',
            raw_data TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_id TEXT NOT NULL,
            reasoning TEXT,
            confidence REAL,
            action_taken TEXT,
            action_result TEXT,
            escalated INTEGER DEFAULT 0,
            timestamp TEXT NOT NULL,
            FOREIGN KEY(alert_id) REFERENCES incidents(alert_id)
        );
        CREATE TABLE IF NOT EXISTS remediations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_id TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            tool_args TEXT,
            tool_result TEXT,
            success INTEGER DEFAULT 1,
            timestamp TEXT NOT NULL,
            FOREIGN KEY(alert_id) REFERENCES incidents(alert_id)
        );
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_id TEXT UNIQUE NOT NULL,
            report_md TEXT,
            generated_at TEXT NOT NULL,
            FOREIGN KEY(alert_id) REFERENCES incidents(alert_id)
        );
        CREATE TABLE IF NOT EXISTS operator_responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_id TEXT NOT NULL,
            decision TEXT,
            notes TEXT,
            responded_at TEXT NOT NULL,
            FOREIGN KEY(alert_id) REFERENCES incidents(alert_id)
        );
    """)
    conn.commit()
    conn.close()

def insert_incident(alert_id, sensor_id, alert_type, severity, raw_data):
    conn = get_conn()
    conn.execute("INSERT OR IGNORE INTO incidents (alert_id,sensor_id,alert_type,severity,status,raw_data,created_at) VALUES (?,?,?,?,'analyzing',?,?)",
        (alert_id,sensor_id,alert_type,severity,json.dumps(raw_data),datetime.utcnow().isoformat()))
    conn.commit(); conn.close()

def update_incident_status(alert_id, status):
    conn = get_conn()
    conn.execute("UPDATE incidents SET status=? WHERE alert_id=?",(status,alert_id))
    conn.commit(); conn.close()

def log_decision(alert_id, reasoning, confidence, action_taken, action_result, escalated):
    conn = get_conn()
    conn.execute("INSERT INTO decisions (alert_id,reasoning,confidence,action_taken,action_result,escalated,timestamp) VALUES (?,?,?,?,?,?,?)",
        (alert_id,reasoning,confidence,action_taken,json.dumps(action_result),1 if escalated else 0,datetime.utcnow().isoformat()))
    conn.commit(); conn.close()

def log_remediation(alert_id, tool_name, tool_args, tool_result, success):
    conn = get_conn()
    conn.execute("INSERT INTO remediations (alert_id,tool_name,tool_args,tool_result,success,timestamp) VALUES (?,?,?,?,?,?)",
        (alert_id,tool_name,json.dumps(tool_args),json.dumps(tool_result),1 if success else 0,datetime.utcnow().isoformat()))
    conn.commit(); conn.close()

def save_report(alert_id, report_md):
    conn = get_conn()
    conn.execute("INSERT OR REPLACE INTO reports (alert_id,report_md,generated_at) VALUES (?,?,?)",
        (alert_id,report_md,datetime.utcnow().isoformat()))
    conn.commit(); conn.close()

def save_operator_response(alert_id, decision, notes):
    conn = get_conn()
    conn.execute("INSERT INTO operator_responses (alert_id,decision,notes,responded_at) VALUES (?,?,?,?)",
        (alert_id,decision,notes,datetime.utcnow().isoformat()))
    # FIX: 'approve' and 'manual_fix' both count as operator_resolved (not just "resolved")
    new_status = "operator_resolved" if decision in ("approve","manual_fix") else "operator_review"
    conn.execute("UPDATE incidents SET status=? WHERE alert_id=?",(new_status,alert_id))
    conn.commit(); conn.close()

def get_all_incidents(limit=100):
    conn = get_conn()
    rows = conn.execute("""
        SELECT i.*, d.confidence, d.escalated, d.action_taken, d.reasoning
        FROM incidents i
        LEFT JOIN decisions d ON i.alert_id = d.alert_id
        ORDER BY i.created_at DESC LIMIT ?
    """,(limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_incident_detail(alert_id):
    conn = get_conn()
    inc = conn.execute("SELECT * FROM incidents WHERE alert_id=?",(alert_id,)).fetchone()
    dec = conn.execute("SELECT * FROM decisions WHERE alert_id=?",(alert_id,)).fetchone()
    rem = conn.execute("SELECT * FROM remediations WHERE alert_id=? ORDER BY timestamp",(alert_id,)).fetchall()
    rep = conn.execute("SELECT * FROM reports WHERE alert_id=?",(alert_id,)).fetchone()
    opr = conn.execute("SELECT * FROM operator_responses WHERE alert_id=? ORDER BY responded_at DESC LIMIT 1",(alert_id,)).fetchone()
    conn.close()
    return {
        "incident": dict(inc) if inc else {},
        "decision": dict(dec) if dec else {},
        "remediations": [dict(r) for r in rem],
        "report": dict(rep) if rep else {},
        "operator_response": dict(opr) if opr else {},
    }

def get_stats():
    conn = get_conn()
    total        = conn.execute("SELECT COUNT(*) as c FROM incidents").fetchone()["c"]
    auto_res     = conn.execute("SELECT COUNT(*) as c FROM incidents WHERE status='resolved'").fetchone()["c"]
    op_res       = conn.execute("SELECT COUNT(*) as c FROM incidents WHERE status='operator_resolved'").fetchone()["c"]
    escalated    = conn.execute("SELECT COUNT(*) as c FROM incidents WHERE status='escalated'").fetchone()["c"]
    analyzing    = conn.execute("SELECT COUNT(*) as c FROM incidents WHERE status='analyzing'").fetchone()["c"]
    errors       = conn.execute("SELECT COUNT(*) as c FROM incidents WHERE status='error'").fetchone()["c"]
    conn.close()
    total_resolved = auto_res + op_res
    return {
        "total": total,
        "auto_resolved": auto_res,
        "operator_resolved": op_res,
        "total_resolved": total_resolved,
        "escalated": escalated,
        "analyzing": analyzing,
        "errors": errors,
        "automation_rate": round(auto_res / total * 100, 1) if total > 0 else 0,
        "resolution_rate": round(total_resolved / total * 100, 1) if total > 0 else 0,
    }


def clear_all_data():
    """Wipe all incident data — called on startup or by user request."""
    conn = get_conn()
    conn.executescript("""
        DELETE FROM incidents;
        DELETE FROM decisions;
        DELETE FROM remediations;
        DELETE FROM reports;
        DELETE FROM operator_responses;
    """)
    conn.commit()
    conn.close()


def get_export_data():
    """Full denormalized export: one row per incident with everything that happened to it,
    including a de-duplicated, human-readable summary of all actions taken."""
    conn = get_conn()
    incidents = conn.execute("""
        SELECT i.*, d.confidence, d.escalated, d.reasoning
        FROM incidents i
        LEFT JOIN decisions d ON i.alert_id = d.alert_id
        ORDER BY i.created_at DESC
    """).fetchall()

    rows = []
    for inc in incidents:
        inc = dict(inc)
        alert_id = inc["alert_id"]

        rem_rows = conn.execute(
            "SELECT tool_name, tool_result FROM remediations WHERE alert_id=? ORDER BY timestamp",
            (alert_id,)
        ).fetchall()
        seen = {}
        for r in rem_rows:
            try:
                msg = json.loads(r["tool_result"] or "{}").get("message", "done")
            except Exception:
                msg = "done"
            key = (r["tool_name"], msg)
            seen[key] = seen.get(key, 0) + 1
        actions = []
        for (tool, msg), count in seen.items():
            suffix = f" (x{count})" if count > 1 else ""
            actions.append(f"{tool}{suffix}: {msg}")
        actions_str = " | ".join(actions) if actions else "None"

        rep = conn.execute("SELECT report_md FROM reports WHERE alert_id=?", (alert_id,)).fetchone()
        opr = conn.execute(
            "SELECT decision, notes FROM operator_responses WHERE alert_id=? ORDER BY responded_at DESC LIMIT 1",
            (alert_id,)
        ).fetchone()

        root_cause = ""
        if inc.get("reasoning"):
            for line in inc["reasoning"].split("\n"):
                if "ROOT_CAUSE:" in line:
                    root_cause = line.split("ROOT_CAUSE:", 1)[-1].strip()
                    break
            if not root_cause:
                root_cause = inc["reasoning"][:200]

        try:
            raw = json.loads(inc.get("raw_data") or "{}")
            data = raw.get("data", {})
        except Exception:
            data = {}

        rows.append({
            "alert_id": alert_id,
            "sensor_id": inc.get("sensor_id", ""),
            "alert_type": inc.get("alert_type", ""),
            "severity": inc.get("severity", ""),
            "status": inc.get("status", ""),
            "confidence_pct": round((inc.get("confidence") or 0) * 100, 1),
            "reading_value": data.get("value", ""),
            "reading_unit": data.get("unit", ""),
            "threshold": data.get("threshold", ""),
            "location": data.get("location", ""),
            "root_cause": root_cause,
            "actions_taken": actions_str,
            "operator_decision": opr["decision"] if opr else "",
            "operator_notes": opr["notes"] if opr else "",
            "created_at": inc.get("created_at", ""),
            "full_report": (rep["report_md"] if rep else "").replace("\n", " | "),
        })
    conn.close()
    return rows
