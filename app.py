import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import random
import plotly.express as px
import os

########################################################
# 🔹 DATABASE INITIALIZATION
########################################################
def init_db():
    conn = sqlite3.connect("database.db")
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

########################################################
# 🔹 PROPERTY FUNCTIONS
########################################################
def add_property(name, address):
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("INSERT INTO properties (name, address) VALUES (?,?)",
              (name, address))
    conn.commit()
    conn.close()

def get_properties():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("SELECT * FROM properties")
    rows = c.fetchall()
    conn.close()
    return rows

########################################################
# 🔹 SENSOR FUNCTIONS
########################################################
def add_sensor_readings_from_df(property_id, df):
    conn = sqlite3.connect("database.db")
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
    conn = sqlite3.connect("database.db")
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

########################################################
# 🔹 SIMULATE SENSOR DATA
########################################################
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

########################################################
# 🔹 RISK ANALYSIS
########################################################
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
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("""
        INSERT INTO risk_scores (property_id, timestamp, risk_score, risk_level)
        VALUES (?, ?, ?, ?)
    """, (property_id, datetime.now().isoformat(), score, level))
    conn.commit()
    conn.close()

########################################################
# 🔹 MAINTENANCE LOG
########################################################
def log_maintenance(property_id, action):
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("""
        INSERT INTO maintenance_log (property_id, timestamp, action)
        VALUES (?, ?, ?)
    """, (property_id, datetime.now().isoformat(), action))
    conn.commit()
    conn.close()

def get_maintenance_log(property_id):
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("""
        SELECT timestamp, action FROM maintenance_log
        WHERE property_id=? ORDER BY timestamp DESC
    """, (property_id,))
    rows = c.fetchall()
    conn.close()
    return rows

########################################################
# 🔹 STREAMLIT FRONTEND
########################################################
def main():
    st.title("🏠 Predictive Risk & Maintenance Platform (Demo)")
    init_db()

    menu = ["Add Property", "Upload/Simulate Sensor Data", "View Sensor Data & Risk"]
    choice = st.sidebar.selectbox("Menu", menu)

    ########################################################
    # PAGE 1: ADD PROPERTY
    ########################################################
    if choice == "Add Property":
        st.subheader("Add New Property")
        name = st.text_input("Property Name")
        address = st.text_input("Address")
        if st.button("Save Property"):
            if name and address:
                add_property(name, address)
                st.success("Property added!")
            else:
                st.error("Please fill all fields!")

    ########################################################
    # PAGE 2: UPLOAD OR SIMULATE SENSOR DATA
    ########################################################
    elif choice == "Upload/Simulate Sensor Data":
        st.subheader("Upload or Generate Sensor Readings")
        props = get_properties()
        if not props:
            st.warning("No properties found. Add one first.")
            return
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
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head(20))
            add_sensor_readings_from_df(selected_pid, df)
            st.success("Uploaded readings saved!")

    ########################################################
    # PAGE 3: VIEW SENSOR DATA & RISK DASHBOARD
    ########################################################
    elif choice == "View Sensor Data & Risk":
        st.subheader("Sensor History & Risk Dashboard")
        props = get_properties()
        if not props:
            st.warning("No properties found. Add one first.")
            return
        prop_dict = {f"{p[1]} ({p[2]})": p[0] for p in props}
        selected_prop = st.selectbox("Select Property", list(prop_dict.keys()))
        selected_pid = prop_dict[selected_prop]

        # Sensor readings
        rows = get_sensor_readings(selected_pid)
        if rows:
            df = pd.DataFrame(rows, columns=["timestamp","sensor_type","value"])
            st.write("### Sensor Readings")
            st.dataframe(df)

            # Compute risk
            score, level = compute_risk_score(df)
            log_risk(selected_pid, score, level)
            st.metric("Current Risk Score", f"{score} ({level})")

            # Recommendations
            recs = generate_recommendations(score, level, df)
            st.subheader("Recommendations")
            for r in recs:
                st.write("- " + r)

            # Maintenance log input
            action = st.text_input("Log a maintenance action")
            if st.button("Save Action"):
                if action:
                    log_maintenance(selected_pid, action)
                    st.success("Maintenance action logged!")
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
            st.subheader("Sensor Trends")
            fig = px.line(df, x="timestamp", y="value", color="sensor_type", title="Sensor Trends")
            st.plotly_chart(fig)

        else:
            st.info("No sensor readings recorded for this property yet.")

if __name__ == "__main__":
    main()
