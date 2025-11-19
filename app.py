# grok_style_predictive_platform.py
import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import random
import plotly.express as px
import os
import uuid

# ---------------------------
# Page config + imports
# ---------------------------
st.set_page_config(
    page_title="Grok-Style Predictive Risk & Maintenance",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------
# Custom CSS (Grok-like dark theme)
# ---------------------------
GROK_CSS = """
/* Google font import */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

:root{
  --bg:#0b0f13;
  --card:#0f1720;
  --muted:#93a0ad;
  --accent:#7c5cff; /* purple neon */
  --accent-2:#00e5a8; /* mint */
  --glass: rgba(255,255,255,0.03);
  --radius:14px;
  --glass-2: rgba(255,255,255,0.02);
}

/* page background */
body, .stApp {
  background: linear-gradient(180deg, #06070a 0%, #0b0f13 100%);
  color: #e6eef6;
  font-family: 'Inter', sans-serif;
}

/* sidebar */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  border-right: 1px solid rgba(255,255,255,0.03);
  padding: 20px 18px;
}

/* header card */
.grok-header {
  background: linear-gradient(180deg, rgba(124,92,255,0.12), rgba(0,0,0,0.0));
  padding: 18px;
  border-radius: 12px;
  border: 1px solid rgba(124,92,255,0.12);
  display:flex;
  align-items:center;
  gap:12px;
  box-shadow: 0 6px 30px rgba(12,14,20,0.6);
}

/* faux logo */
.grok-badge {
  width:56px;
  height:56px;
  border-radius:12px;
  background: linear-gradient(135deg, var(--accent), var(--accent-2));
  display:flex;
  align-items:center;
  justify-content:center;
  font-weight:800;
  color:white;
  font-size:20px;
  box-shadow: 0 6px 24px rgba(0,0,0,0.6);
}

/* headline */
.grok-title {
  font-size:20px;
  font-weight:700;
  letter-spacing: -0.2px;
}

/* subtitle */
.grok-sub {
  color:var(--muted);
  font-size:13px;
  margin-top:2px;
}

/* panels */
.grok-panel {
  background: var(--card);
  border-radius: var(--radius);
  padding: 16px;
  border: 1px solid rgba(255,255,255,0.03);
  box-shadow: 0 6px 18px rgba(0,0,0,0.6);
  animation: fadein .5s ease both;
}

/* metric badges */
.metric-card {
  background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  padding: 12px;
  border-radius: 12px;
  border: 1px solid rgba(255,255,255,0.03);
}

/* buttons */
button.stButton > button {
  background: linear-gradient(90deg, var(--accent), var(--accent-2));
  color: white;
  border: none;
  padding: 8px 14px;
  border-radius: 10px;
  font-weight:600;
  box-shadow: 0 8px 24px rgba(124,92,255,0.12);
}

/* inputs */
div.stTextInput > label, div.stTextArea > label {
  color:var(--muted);
}
input, textarea, select {
  background: rgba(255,255,255,0.02) !important;
  border: 1px solid rgba(255,255,255,0.03) !important;
  color: #e6eef6 !important;
  border-radius: 10px !important;
}

/* table tweaks */
.stDataFrame table {
  background: transparent;
}

/* small helpers */
.kv {
  color: var(--muted);
  font-size:13px;
}

/* chart container */
.chart-wrap {
  padding: 6px;
  border-radius: 10px;
  background: linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.00));
  border: 1px solid rgba(255,255,255,0.03);
}

/* fade animation */
@keyframes fadein {
  from { opacity: 0; transform: translateY(6px);}
  to { opacity: 1; transform: translateY(0);}
}
"""

# Inject CSS
st.markdown(f"<style>{GROK_CSS}</style>", unsafe_allow_html=True)


# ---------------------------
# 🔹 DATABASE INITIALIZATION
# ---------------------------
DB_PATH = "database.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Properties table
    c.execute("""
    CREATE TABLE IF NOT EXISTS properties (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        address TEXT
    )
    """)

    # Sensor readings table
    c.execute("""
    CREATE TABLE IF NOT EXISTS sensor_readings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        property_id INTEGER,
        timestamp TEXT,
        sensor_type TEXT,
        value REAL
    )
    """)

    # Risk scores table
    c.execute("""
    CREATE TABLE IF NOT EXISTS risk_scores (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        property_id INTEGER,
        timestamp TEXT,
        risk_score REAL,
        risk_level TEXT
    )
    """)

    # Maintenance log table
    c.execute("""
    CREATE TABLE IF NOT EXISTS maintenance_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        property_id INTEGER,
        timestamp TEXT,
        action TEXT
    )
    """)

    conn.commit()
    conn.close()


# ---------------------------
# 🔹 PROPERTY FUNCTIONS
# ---------------------------
def add_property(name, address):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO properties (name, address) VALUES (?,?)",
              (name, address))
    conn.commit()
    conn.close()

def get_properties():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM properties")
    rows = c.fetchall()
    conn.close()
    return rows

# ---------------------------
# 🔹 SENSOR FUNCTIONS
# ---------------------------
def add_sensor_readings_from_df(property_id, df):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    for _, row in df.iterrows():
        c.execute("""
            INSERT INTO sensor_readings (property_id, timestamp, sensor_type, value)
            VALUES (?, ?, ?, ?)
        """, (
            property_id,
            row["timestamp"],
            row["sensor_type"],
            float(row["value"])
        ))
    conn.commit()
    conn.close()

def get_sensor_readings(property_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT timestamp, sensor_type, value 
        FROM sensor_readings 
        WHERE property_id=?
        ORDER BY timestamp ASC
    """, (property_id,))
    rows = c.fetchall()
    conn.close()
    return rows

# ---------------------------
# 🔹 SIMULATE SENSOR DATA
# ---------------------------
def simulate_sensor_data():
    start = datetime.now() - timedelta(hours=6)
    rows = []
    sensor_types = ["humidity", "temperature", "water_leak"]
    for i in range(120):
        ts = start + timedelta(minutes=5*i)
        s_type = random.choice(sensor_types)
        if s_type == "humidity":
            value = round(random.uniform(40, 90), 2)
        elif s_type == "temperature":
            value = round(random.uniform(18, 34), 2)
        else:
            value = random.choice([0, 1])
        rows.append([ts.isoformat(), s_type, value])
    df = pd.DataFrame(rows, columns=["timestamp", "sensor_type", "value"])
    return df

# ---------------------------
# 🔹 RISK ANALYSIS
# ---------------------------
def compute_risk_score(df):
    latest = df.tail(10)
    humidity_avg = latest[latest.sensor_type=='humidity'].value.mean() if not latest[latest.sensor_type=='humidity'].empty else 0
    moisture_avg = latest[latest.sensor_type=='water_leak'].value.mean() if not latest[latest.sensor_type=='water_leak'].empty else 0
    temp_latest = latest[latest.sensor_type=='temperature'].value.iloc[-1] if not latest[latest.sensor_type=='temperature'].empty else 25

    score = 0
    if humidity_avg > 70:
        score += 40
    if moisture_avg > 0:
        score += 40
    if temp_latest < 5 or temp_latest > 30:
        score += 20
    score = min(score, 100)

    if score < 40:
        level = 'Low'
    elif score < 70:
        level = 'Medium'
    else:
        level = 'High'

    return score, level

def generate_recommendations(score, level, df):
    recs = []
    humidity_avg = df[df.sensor_type=='humidity'].value.mean() if not df[df.sensor_type=='humidity'].empty else 0
    moisture_avg = df[df.sensor_type=='water_leak'].value.mean() if not df[df.sensor_type=='water_leak'].empty else 0
    temp_latest = df[df.sensor_type=='temperature'].value.iloc[-1] if not df[df.sensor_type=='temperature'].empty else 25

    if humidity_avg > 70:
        recs.append("Humidity high → Check ventilation / dehumidifier.")
    if moisture_avg > 0:
        recs.append("Water leak detected → Inspect immediately!")
    if temp_latest < 5:
        recs.append("Low temperature → Risk of frozen pipes.")
    if temp_latest > 30:
        recs.append("High temperature → Condensation / mold risk.")
    if not recs:
        recs.append("No immediate actions required.")
    return recs

def log_risk(property_id, score, level):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO risk_scores (property_id, timestamp, risk_score, risk_level)
        VALUES (?, ?, ?, ?)
    """, (property_id, datetime.now().isoformat(), score, level))
    conn.commit()
    conn.close()

# ---------------------------
# 🔹 MAINTENANCE LOG
# ---------------------------
def log_maintenance(property_id, action):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO maintenance_log (property_id, timestamp, action)
        VALUES (?, ?, ?)
    """, (property_id, datetime.now().isoformat(), action))
    conn.commit()
    conn.close()

def get_maintenance_log(property_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT timestamp, action FROM maintenance_log
        WHERE property_id=? ORDER BY timestamp DESC
    """, (property_id,))
    rows = c.fetchall()
    conn.close()
    return rows

# ---------------------------
# 🔹 SMALL UTILS
# ---------------------------
def make_header():
    header_html = f"""
    <div class="grok-header">
      <div class="grok-badge">GR</div>
      <div>
        <div class="grok-title">Grok — Predictive Risk & Maintenance (Demo)</div>
        <div class="grok-sub">Lightweight demo · Streamlit · Prototype</div>
      </div>
      <div style="margin-left:auto; text-align:right;">
        <div style="font-size:12px; color:var(--muted)">Workspace</div>
        <div style="font-weight:700;">{os.getenv('USERNAME', os.getenv('USER', 'local'))} · {datetime.now().strftime('%Y-%m-%d')}</div>
      </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

# ---------------------------
# 🔹 STREAMLIT FRONTEND
# ---------------------------
def main():
    init_db()
    make_header()
    st.write("")  # spacer

    # Sidebar
    with st.sidebar:
        st.markdown("## Control Panel")
        st.markdown("---")
        menu = ["Add Property", "Upload/Simulate Sensor Data", "View Sensor Data & Risk"]
        choice = st.radio("", menu, index=2)
        st.markdown("---")
        st.markdown("### Quick Actions")
        if st.button("Generate Demo Property & Data"):
            # create demo property if none exist
            props = get_properties()
            if not props:
                add_property("Demo Building", "12 Grok Lane, Lagos")
            props = get_properties()
            pid = props[0][0]
            df_sim = simulate_sensor_data()
            add_sensor_readings_from_df(pid, df_sim)
            st.success("Demo property & data created!")
            st.experimental_rerun()

        st.markdown("")
        st.markdown("### Theme")
        st.markdown("<div style='font-size:13px; color:var(--muted)'>Grok-style dark theme applied</div>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### About")
        st.markdown("<div style='font-size:13px; color:var(--muted)'>Prototype demo — not production. Use as starting point for your SaaS.</div>", unsafe_allow_html=True)

    # Main content columns
    col1, col2 = st.columns([1.7, 1])

    # LEFT: Pages
    if choice == "Add Property":
        with col1:
            st.markdown('<div class="grok-panel">', unsafe_allow_html=True)
            st.subheader("Add New Property")
            name = st.text_input("Property Name", placeholder="e.g., Midtown Warehouse")
            address = st.text_input("Address", placeholder="Street, City, Country")
            if st.button("Save Property"):
                if name and address:
                    add_property(name, address)
                    st.success("Property added!")
                else:
                    st.error("Please fill all fields!")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="grok-panel">', unsafe_allow_html=True)
            st.markdown("### Properties")
            props = get_properties()
            if props:
                for p in props:
                    st.markdown(f"**{p[1]}**  \n<small class='kv'>{p[2]}</small>", unsafe_allow_html=True)
            else:
                st.info("No properties yet.")
            st.markdown("</div>", unsafe_allow_html=True)

    elif choice == "Upload/Simulate Sensor Data":
        with col1:
            st.markdown('<div class="grok-panel">', unsafe_allow_html=True)
            st.subheader("Upload or Generate Sensor Readings")
            props = get_properties()
            if not props:
                st.warning("No properties found. Add one first.")
            else:
                prop_dict = {f"{p[1]} ({p[2]})": p[0] for p in props}
                selected_prop = st.selectbox("Select Property", list(prop_dict.keys()))
                selected_pid = prop_dict[selected_prop]

                st.write("### Option 1: Upload CSV File")
                uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

                st.write("### Option 2: Generate Sample Data")
                if st.button("Generate Sample"):
                    df_sim = simulate_sensor_data()
                    st.dataframe(df_sim.head(20))
                    add_sensor_readings_from_df(selected_pid, df_sim)
                    st.success("Simulated readings saved!")

                if uploaded_file:
                    try:
                        df = pd.read_csv(uploaded_file)
                        st.dataframe(df.head(20))
                        add_sensor_readings_from_df(selected_pid, df)
                        st.success("Uploaded readings saved!")
                    except Exception as e:
                        st.error(f"Failed to read file: {e}")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="grok-panel">', unsafe_allow_html=True)
            st.markdown("### Tips")
            st.markdown("- CSV must have columns: `timestamp`, `sensor_type`, `value`")
            st.markdown("- sensor_type: humidity|temperature|water_leak")
            st.markdown("- timestamp ISO format recommended (e.g., 2025-11-19T12:00:00)")
            st.markdown("</div>", unsafe_allow_html=True)

    else:  # View Sensor Data & Risk (default)
        with col1:
            st.markdown('<div class="grok-panel">', unsafe_allow_html=True)
            st.subheader("Sensor History & Risk Dashboard")

            props = get_properties()
            if not props:
                st.warning("No properties found. Add one first.")
                st.markdown("</div>", unsafe_allow_html=True)
                return

            prop_dict = {f"{p[1]} ({p[2]})": p[0] for p in props}
            selected_prop = st.selectbox("Select Property", list(prop_dict.keys()))
            selected_pid = prop_dict[selected_prop]

            # Sensor readings
            rows = get_sensor_readings(selected_pid)
            if rows:
                df = pd.DataFrame(rows, columns=["timestamp","sensor_type","value"])
                # parse timestamps for plotting
                try:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                except Exception:
                    pass

                st.markdown("### Sensor Readings")
                st.dataframe(df.tail(50).reset_index(drop=True))

                # Compute risk
                score, level = compute_risk_score(df)
                log_risk(selected_pid, score, level)

                # Metric-style cards
                mcol1, mcol2, mcol3 = st.columns([1,1,1])
                with mcol1:
                    st.markdown(f'<div class="metric-card"><h3 style="margin:0;">{score}</h3><div class="kv">Current Risk Score</div></div>', unsafe_allow_html=True)
                with mcol2:
                    st.markdown(f'<div class="metric-card"><h3 style="margin:0;">{level}</h3><div class="kv">Risk Level</div></div>', unsafe_allow_html=True)
                with mcol3:
                    last_ts = df["timestamp"].max()
                    st.markdown(f'<div class="metric-card"><h3 style="margin:0;">{last_ts}</h3><div class="kv">Last Reading</div></div>', unsafe_allow_html=True)

                # Recommendations
                recs = generate_recommendations(score, level, df)
                st.subheader("Recommendations")
                for r in recs:
                    st.write("• " + r)

                # Maintenance log input
                with st.form("maintenance_form", clear_on_submit=True):
                    action = st.text_input("Log a maintenance action")
                    submitted = st.form_submit_button("Save Action")
                    if submitted:
                        if action:
                            log_maintenance(selected_pid, action)
                            st.success("Maintenance action logged!")
                            st.experimental_rerun()
                        else:
                            st.error("Enter a valid action.")

                st.subheader("Maintenance History")
                log_rows = get_maintenance_log(selected_pid)
                if log_rows:
                    df_log = pd.DataFrame(log_rows, columns=["timestamp","action"])
                    st.dataframe(df_log)
                else:
                    st.info("No maintenance actions recorded yet.")

                # Trend charts
                st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
                fig = px.line(df, x="timestamp", y="value", color="sensor_type", title="Sensor Trends")
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="white"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            else:
                st.info("No sensor readings recorded for this property yet.")
            st.markdown("</div>", unsafe_allow_html=True)

        # RIGHT column: insights + recent risks
        with col2:
            st.markdown('<div class="grok-panel">', unsafe_allow_html=True)
            st.markdown("### Recent Risk Logs")
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("""
                SELECT r.timestamp, p.name, r.risk_score, r.risk_level
                FROM risk_scores r
                LEFT JOIN properties p ON r.property_id = p.id
                ORDER BY r.timestamp DESC LIMIT 8
            """)
            recent = c.fetchall()
            conn.close()
            if recent:
                df_recent = pd.DataFrame(recent, columns=["timestamp","property","score","level"])
                try:
                    df_recent["timestamp"] = pd.to_datetime(df_recent["timestamp"])
                except Exception:
                    pass
                st.dataframe(df_recent)
            else:
                st.info("No risk logs yet.")

            st.markdown("---")
            st.markdown("### Export / Share")
            st.markdown("<div class='kv'>You can export recent logs as CSV for sharing with insurers or teams.</div>", unsafe_allow_html=True)
            if st.button("Export Recent Logs as CSV"):
                conn = sqlite3.connect(DB_PATH)
                df_all = pd.read_sql_query("""
                    SELECT r.timestamp, p.name, r.risk_score, r.risk_level
                    FROM risk_scores r
                    LEFT JOIN properties p ON r.property_id = p.id
                    ORDER BY r.timestamp DESC
                """, conn)
                conn.close()
                csv = df_all.to_csv(index=False).encode('utf-8')
                st.download_button("Download CSV", data=csv, file_name="risk_logs.csv", mime="text/csv")
            st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
