"""
Predictive Risk & Maintenance Platform - MVP (Streamlit)

Features:
- Add properties (manual input)
- Upload sensor CSV or simulate live sensor readings
- Rule-based risk scoring
- Simple ML model (trained on synthetic data) to output risk probability
- Persist data in local SQLite
- Time-series chart of sensor data (Plotly)
- "Request vendor" mock flow to create a job request
- Optional email alert function (configure SMTP settings)

How to run:
1. pip install -r requirements.txt
2. streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import datetime
import io
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pickle
import os
import smtplib
from email.message import EmailMessage

# ----------------------
# Constants & Utilities
# ----------------------
DB_PATH = "prm_platform.db"
MODEL_PATH = "risk_model.pkl"
np.random.seed(42)

# ----------------------
# Database helpers
# ----------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS properties (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        address TEXT,
        building_age INTEGER,
        last_inspection_date TEXT,
        created_at TEXT
    )
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS sensor_readings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        property_id INTEGER,
        timestamp TEXT,
        sensor_type TEXT,
        value REAL,
        FOREIGN KEY(property_id) REFERENCES properties(id)
    )
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS incidents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        property_id INTEGER,
        timestamp TEXT,
        description TEXT,
        severity INTEGER,
        FOREIGN KEY(property_id) REFERENCES properties(id)
    )
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS vendor_requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        property_id INTEGER,
        timestamp TEXT,
        detail TEXT,
        status TEXT,
        FOREIGN KEY(property_id) REFERENCES properties(id)
    )
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        property_id INTEGER,
        timestamp TEXT,
        rule_score INTEGER,
        ml_prob REAL,
        note TEXT,
        FOREIGN KEY(property_id) REFERENCES properties(id)
    )
    """)
    conn.commit()
    conn.close()

def add_property(name, address, building_age, last_inspection):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO properties (name,address,building_age,last_inspection_date,created_at) VALUES (?,?,?,?,?)",
              (name,address,building_age,last_inspection, datetime.datetime.utcnow().isoformat()))
    conn.commit()
    pid = c.lastrowid
    conn.close()
    return pid

def list_properties():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM properties ORDER BY id DESC", conn)
    conn.close()
    return df

def add_sensor_readings_from_df(property_id, df):
    # expect df with 'timestamp','sensor_type','value'
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    for _, row in df.iterrows():
        c.execute("INSERT INTO sensor_readings (property_id,timestamp,sensor_type,value) VALUES (?,?,?)",
                  (property_id, row['timestamp'], row['sensor_type'], float(row['value'])))
    conn.commit()
    conn.close()

def get_sensor_readings(property_id, sensor_type=None, limit_days=30):
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM sensor_readings WHERE property_id = ?"
    params = [property_id]
    if sensor_type:
        query += " AND sensor_type = ?"
        params.append(sensor_type)
    # recent N days
    cutoff = (datetime.datetime.utcnow() - datetime.timedelta(days=limit_days)).isoformat()
    query += " AND timestamp >= ? ORDER BY timestamp ASC"
    params.append(cutoff)
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

def add_vendor_request(property_id, detail):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO vendor_requests (property_id,timestamp,detail,status) VALUES (?,?,?,?)",
              (property_id, datetime.datetime.utcnow().isoformat(), detail, "OPEN"))
    conn.commit()
    rid = c.lastrowid
    conn.close()
    return rid

def add_prediction_record(property_id, rule_score, ml_prob, note=""):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO predictions (property_id,timestamp,rule_score,ml_prob,note) VALUES (?,?,?,?,?)",
              (property_id, datetime.datetime.utcnow().isoformat(), int(rule_score), float(ml_prob), note))
    conn.commit()
    conn.close()

# ----------------------
# Risk Engine - Rule based
# ----------------------
def compute_rule_risk(latest_humidity, leak_events_last_12m, building_age_years, last_inspection_days):
    score = 0
    # humidity strong indicator
    if latest_humidity is None:
        latest_humidity = 50
    if latest_humidity >= 75:
        score += 55
    elif latest_humidity >= 65:
        score += 35
    elif latest_humidity >= 60:
        score += 15

    # leaks
    if leak_events_last_12m >= 3:
        score += 30
    elif leak_events_last_12m == 2:
        score += 20
    elif leak_events_last_12m == 1:
        score += 10

    # building age
    if building_age_years >= 50:
        score += 10
    elif building_age_years >= 30:
        score += 6
    elif building_age_years >= 15:
        score += 3

    # no recent inspection increases score
    if last_inspection_days is None:
        last_inspection_days = 365
    if last_inspection_days >= 365:
        score += 8
    elif last_inspection_days >= 180:
        score += 4

    final = int(min(100, score))
    return final

# ----------------------
# Synthetic data & ML model (works out-of-box)
# ----------------------
def generate_synthetic_dataset(n=2000):
    # features: humidity_mean, humidity_std, leak_count_12m, building_age, last_inspection_days, temp_mean
    humidity_mean = np.random.normal(55, 10, n).clip(20, 100)
    humidity_std = np.random.exponential(3, n).clip(0, 20)
    leak_count = np.random.poisson(0.6, n)
    building_age = np.random.randint(1, 80, n)
    last_inspection_days = np.random.randint(0, 900, n)
    temp_mean = np.random.normal(23, 4, n)

    # risk label: higher prob when humidity_mean high, leak_count high, old building, few inspections
    logits = (
        0.06 * (humidity_mean - 50)
        + 0.35 * leak_count
        + 0.02 * (building_age - 20)
        + 0.01 * (last_inspection_days - 200) / 10
        - 0.02 * (temp_mean - 22)
        + np.random.normal(0, 1, n)
    )
    probs = 1 / (1 + np.exp(-logits))
    labels = (probs > 0.5).astype(int)

    df = pd.DataFrame({
        'humidity_mean': humidity_mean,
        'humidity_std': humidity_std,
        'leak_count': leak_count,
        'building_age': building_age,
        'last_inspection_days': last_inspection_days,
        'temp_mean': temp_mean,
        'label': labels
    })
    return df

def train_or_load_model(force_retrain=False):
    if os.path.exists(MODEL_PATH) and not force_retrain:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        return model

    df = generate_synthetic_dataset(n=2000)
    X = df[['humidity_mean','humidity_std','leak_count','building_age','last_inspection_days','temp_mean']]
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    # evaluate
    y_prob = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, y_prob)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print("Trained model saved. AUC:", auc)
    return model

# ----------------------
# Email alert (optional)
# ----------------------
def send_email_alert(smtp_host, smtp_port, smtp_user, smtp_pass, to_email, subject, body):
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = smtp_user
    msg['To'] = to_email
    msg.set_content(body)
    try:
        server = smtplib.SMTP(smtp_host, smtp_port, timeout=10)
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)
        server.quit()
        return True, "Sent"
    except Exception as e:
        return False, str(e)

# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="Predictive Risk & Maintenance - MVP", layout="wide")
init_db()
model = train_or_load_model()

st.title("Predictive Risk & Maintenance — MVP")
st.markdown("Simple demo of rule-based + ML risk scoring for building damage (water / mold).")

# Sidebar: add property
st.sidebar.header("Add / Select Property")
with st.sidebar.form("add_prop"):
    pname = st.text_input("Property name", value="My Property")
    paddress = st.text_input("Address", value="123 Example St")
    age = st.number_input("Building age (years)", min_value=0, max_value=200, value=25)
    last_inspection = st.date_input("Last inspection date", value=(datetime.date.today() - datetime.timedelta(days=200)))
    submitted = st.form_submit_button("Add property")
    if submitted:
        pid = add_property(pname, paddress, age, last_inspection.isoformat())
        st.sidebar.success(f"Property added (id={pid})")

props = list_properties()
if props.empty:
    st.info("No properties yet. Add one in the sidebar.")
    st.stop()

prop_select = st.sidebar.selectbox("Choose property", options=props['id'].tolist(),
                                   format_func=lambda x: f"{int(x)} - {props[props.id==x].iloc[0].name}")

selected_pid = int(prop_select)
prop_row = props[props.id == selected_pid].iloc[0]

st.sidebar.markdown(f"**Selected:** {prop_row['name']}  \n{prop_row['address']}")
st.sidebar.markdown(f"Building age: {prop_row['building_age']}")

# Main layout
col1, col2 = st.columns([2,1])

with col1:
    st.header("Sensor data & risk")
    st.subheader("Upload sensor CSV or simulate readings")
    st.markdown("CSV format: timestamp(ISO) , sensor_type (humidity/temp/etc) , value")

    uploaded = st.file_uploader("Upload sensor CSV", type=['csv'])
    if uploaded:
        try:
            df_raw = pd.read_csv(uploaded)
            # basic validation
            if not {'timestamp','sensor_type','value'}.issubset(set(df_raw.columns.str.lower())):
                st.warning("CSV must contain columns: timestamp, sensor_type, value (case-insensitive).")
            else:
                # normalize column names
                df_raw.columns = [c.lower() for c in df_raw.columns]
                # convert timestamp to iso if needed
                df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp']).dt.strftime('%Y-%m-%dT%H:%M:%S')
                add_sensor_readings_from_df(selected_pid, df_raw[['timestamp','sensor_type','value']])
                st.success("Sensor readings saved.")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

    st.markdown("---")
    st.write("Or simulate sensor readings for demo:")
    sim_count = st.slider("Number of simulated readings", 5, 200, 30)
    if st.button("Simulate readings (demo)"):
        now = datetime.datetime.utcnow()
        timestamps = [now - datetime.timedelta(minutes=15*i) for i in range(sim_count)][::-1]
        # create humidity readings with occasional spikes
        humid = (np.random.normal(55,6, sim_count) + np.sin(np.linspace(0,6.28, sim_count))*3 + np.random.choice([0,0,8], sim_count, p=[0.8,0.15,0.05])).clip(20,100)
        df_sim = pd.DataFrame({
            'timestamp': [t.isoformat(timespec='seconds') for t in timestamps],
            'sensor_type': ['humidity']*sim_count,
            'value': humid
        })
        add_sensor_readings_from_df(selected_pid, df_sim)
        st.success("Simulated readings inserted.")

    st.markdown("### Recent humidity readings (last 30 days)")
    df_h = get_sensor_readings(selected_pid, sensor_type='humidity', limit_days=30)
    if df_h.empty:
        st.info("No humidity data found for this property (upload CSV or simulate readings).")
    else:
        # show chart
        df_h['timestamp'] = pd.to_datetime(df_h['timestamp'])
        fig = px.line(df_h, x='timestamp', y='value', title='Humidity readings', labels={'value':'Humidity (%)','timestamp':'Time'})
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_h.tail(50))

with col2:
    st.header("Risk scoring")
    # compute some features
    # latest humidity
    df_h_all = get_sensor_readings(selected_pid, sensor_type='humidity', limit_days=3650)
    latest_humidity = None
    humidity_mean = None
    humidity_std = None
    if not df_h_all.empty:
        latest_humidity = float(df_h_all.iloc[-1].value)
        humidity_mean = float(df_h_all['value'].mean())
        humidity_std = float(df_h_all['value'].std())
    # leak events simulate: count incidents in table
    conn = sqlite3.connect(DB_PATH)
    incidents_df = pd.read_sql_query("SELECT * FROM incidents WHERE property_id = ?", conn, params=(selected_pid,))
    conn.close()
    leak_count = len(incidents_df)  # simple proxy
    # building age & last inspection
    building_age = int(prop_row['building_age'])
    last_inspection = prop_row['last_inspection_date']
    try:
        last_inspection_date = pd.to_datetime(last_inspection)
        last_inspection_days = (pd.Timestamp.utcnow() - last_inspection_date).days
    except:
        last_inspection_days = None

    st.write(f"Latest humidity: **{latest_humidity if latest_humidity is not None else 'N/A'}**")
    st.write(f"Avg humidity (all data): **{round(humidity_mean,2) if humidity_mean is not None else 'N/A'}**")
    st.write(f"Leak events recorded (proxy): **{leak_count}**")
    st.write(f"Last inspection (days ago): **{last_inspection_days}**")

    # rule-based
    rule_score = compute_rule_risk(latest_humidity or 50, leak_count, building_age, last_inspection_days)
    st.metric("Rule-based risk score (0-100)", rule_score)

    # ML model prediction - prepare feature vector
    feat = {
        'humidity_mean': humidity_mean if humidity_mean is not None else 50.0,
        'humidity_std': humidity_std if humidity_std is not None else 3.0,
        'leak_count': leak_count,
        'building_age': building_age,
        'last_inspection_days': last_inspection_days if last_inspection_days is not None else 365,
        'temp_mean': 22.0
    }
    X_vec = np.array([[
        feat['humidity_mean'], feat['humidity_std'], feat['leak_count'],
        feat['building_age'], feat['last_inspection_days'], feat['temp_mean']
    ]])
    ml_prob = model.predict_proba(X_vec)[:,1][0]
    st.metric("ML predicted probability of incident (0-1)", f"{ml_prob:.2f}")

    note = st.text_area("Analyst note (optional)", value="Auto-check from dashboard")
    if st.button("Save prediction record"):
        add_prediction_record(selected_pid, rule_score, float(ml_prob), note)
        st.success("Prediction saved.")

    st.markdown("---")
    st.header("Actions & alerts")
    st.write("If risk high, you can request an inspection / vendor dispatch.")
    risk_thresh = st.slider("Vendor dispatch if ML prob >= ", 0.0, 1.0, 0.6, step=0.05)
    if ml_prob >= risk_thresh:
        st.warning("Risk threshold reached — consider requesting vendor.")
    vendor_detail = st.text_area("Vendor request details", value="Please inspect for potential leaks & mold risk.")
    if st.button("Request vendor (create record)"):
        rid = add_vendor_request(selected_pid, vendor_detail)
        st.success(f"Vendor request created (id={rid})")
    st.markdown("Optional: send email alert (configure in code).")

st.markdown("---")
st.header("Admin / Data")
tabs = st.tabs(["Properties","Incidents","Vendor Requests","Predictions","Retrain Model"])

with tabs[0]:
    st.subheader("Properties table")
    st.dataframe(list_properties())

with tabs[1]:
    st.subheader("Incidents (manual add)")
    with st.form("add_incident"):
        desc = st.text_input("Description", "Water leak in kitchen")
        sev = st.slider("Severity (1-5)", 1, 5, 2)
        submitted = st.form_submit_button("Add incident")
        if submitted:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("INSERT INTO incidents (property_id,timestamp,description,severity) VALUES (?,?,?,?)",
                      (selected_pid, datetime.datetime.utcnow().isoformat(), desc, sev))
            conn.commit()
            conn.close()
            st.success("Incident added.")
    conn = sqlite3.connect(DB_PATH)
    df_inc = pd.read_sql_query("SELECT * FROM incidents WHERE property_id = ? ORDER BY timestamp DESC", conn, params=(selected_pid,))
    conn.close()
    st.dataframe(df_inc)

with tabs[2]:
    st.subheader("Vendor requests")
    conn = sqlite3.connect(DB_PATH)
    df_v = pd.read_sql_query("SELECT * FROM vendor_requests WHERE property_id = ? ORDER BY timestamp DESC", conn, params=(selected_pid,))
    conn.close()
    st.dataframe(df_v)

with tabs[3]:
    st.subheader("Predictions")
    conn = sqlite3.connect(DB_PATH)
    df_p = pd.read_sql_query("SELECT * FROM predictions WHERE property_id = ? ORDER BY timestamp DESC", conn, params=(selected_pid,))
    conn.close()
    st.dataframe(df_p)

with tabs[4]:
    st.subheader("Retrain ML model on synthetic data (quick)")
    if st.button("Retrain model"):
        model = train_or_load_model(force_retrain=True)
        st.success("Model retrained and saved to disk.")

st.markdown("### Export data")
if st.button("Export predictions CSV"):
    conn = sqlite3.connect(DB_PATH)
    dfx = pd.read_sql_query("SELECT * FROM predictions", conn)
    conn.close()
    csv = dfx.to_csv(index=False).encode('utf-8')
    st.download_button("Download predictions.csv", data=csv, file_name="predictions.csv", mime='text/csv')

st.markdown("---")
st.info("MVP notes: This demo uses synthetic ML training data so model predictions are illustrative. Replace synthetic training with real labeled incidents from your region for production. Add authentication, API ingestion, insurer & vendor integrations when ready.")
