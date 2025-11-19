import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import random
import os

############################################################
# 🔹 DATABASE INITIALIZATION
############################################################

def init_db():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    # Create property table
    c.execute("""
    CREATE TABLE IF NOT EXISTS properties (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        address TEXT
    )
    """)

    # Create sensor readings table
    c.execute("""
    CREATE TABLE IF NOT EXISTS sensor_readings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        property_id INTEGER,
        timestamp TEXT,
        sensor_type TEXT,
        value REAL
    )
    """)

    conn.commit()
    conn.close()


############################################################
# 🔹 ADD A PROPERTY
############################################################

def add_property(name, address):
    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    c.execute("INSERT INTO properties (name, address) VALUES (?,?)",
              (name, address))

    conn.commit()
    conn.close()


############################################################
# 🔹 GET PROPERTIES
############################################################

def get_properties():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    c.execute("SELECT * FROM properties")
    rows = c.fetchall()

    conn.close()
    return rows


############################################################
# 🔹 INSERT SENSOR READINGS (4 VALUES!)
############################################################

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


############################################################
# 🔹 READ SENSOR READINGS FOR DISPLAY
############################################################

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


############################################################
# 🔹 SIMULATE SENSOR DATA CSV
############################################################

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
            value = random.choice([0, 1])  # leak/no leak

        rows.append([ts.isoformat(), s_type, value])

    df = pd.DataFrame(rows, columns=["timestamp", "sensor_type", "value"])
    return df


############################################################
# 🔹 STREAMLIT FRONTEND
############################################################

def main():
    st.title("🏠 Predictive Risk & Maintenance Platform (Demo)")

    init_db()

    menu = ["Add Property", "Upload / Simulate Sensor Data", "View Sensor Data"]
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
    elif choice == "Upload / Simulate Sensor Data":
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
    # PAGE 3: VIEW SENSOR DATA
    ########################################################
    elif choice == "View Sensor Data":
        st.subheader("Sensor History")

        props = get_properties()
        if not props:
            st.warning("No properties found. Add one first.")
            return

        prop_dict = {f"{p[1]} ({p[2]})": p[0] for p in props}
        selected_prop = st.selectbox("Select Property", list(prop_dict.keys()))
        selected_pid = prop_dict[selected_prop]

        rows = get_sensor_readings(selected_pid)

        if rows:
            df = pd.DataFrame(rows, columns=["timestamp", "sensor_type", "value"])
            st.dataframe(df)
        else:
            st.info("No sensor readings recorded for this property yet.")


if __name__ == "__main__":
    main()
