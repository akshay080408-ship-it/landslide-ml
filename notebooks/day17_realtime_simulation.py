# ============================================
# DAY 17 — Real-time Simulation
# Date: 05 June 2026
# Author: Akshay
# Goal: Simulate live ESP32 streaming data
#       Build real-time prediction pipeline
#       This is the bridge to Day 33 hardware!
# ============================================
# ============================================
# DAY 17 OBSERVATIONS:
# 1. Alerts sent with cooldown: 1
# 2. Early warning caught 3 readings before
#    actual danger (reading 4 vs reading 7)
#    = ~90 seconds advance warning
# 3. SensorBuffer correctly calculates
#    live features without CSV history
# 4. Risk trend (WORSENING/IMPROVING) adds
#    valuable context to predictions
# 5. Anomaly + DANGER both triggered together
#    at reading 7 = high confidence alert
# 6. This exact pipeline structure will run
#    on real ESP32 in Day 33
# ============================================

import pandas as pd
import numpy as np
import joblib
import time
from datetime import datetime
from collections import deque
import warnings
warnings.filterwarnings('ignore')

print("="*55)
print("   DAY 17 — REAL-TIME SIMULATION")
print("="*55)

# ── Load production model package ─────────────
package = joblib.load('models/production_model.pkl')
rf_model = package['rf_model']
iso_forest = package['iso_forest']
scaler = package['scaler']
features = package['features']

print(f"\n✅ Production model loaded")
print(f"   Version: {package['metadata']['version']}")

# Load raw data to simulate streaming
# (in real life this comes from ESP32!)
raw_df = pd.read_csv('data/raw/sensor_data.csv')
print(f"✅ Loaded {len(raw_df)} readings to simulate streaming")

# ════════════════════════════════════════════
# PART 1: SLIDING WINDOW BUFFER CLASS
# ════════════════════════════════════════════
print("\n📌 PART 1: Building Sliding Window Buffer")

class SensorBuffer:
    """
    Maintains rolling history of sensor readings.
    Calculates live features WITHOUT needing
    the full dataset — just like real ESP32!
    """
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.moisture_history = deque(maxlen=window_size)
        self.magnitude_history = deque(maxlen=window_size)
        self.last_moisture = None
        self.last_magnitude = None

    def add_reading(self, accel_x, accel_y, accel_z,
                     moisture, vibration):
        # Calculate magnitude (Day 3 formula)
        magnitude = np.sqrt(
            accel_x**2 + accel_y**2 + accel_z**2
        )

        # Calculate rate of change
        moisture_change = (
            moisture - self.last_moisture
            if self.last_moisture is not None else 0
        )
        magnitude_change = (
            magnitude - self.last_magnitude
            if self.last_magnitude is not None else 0
        )

        # Update history buffers
        self.moisture_history.append(moisture)
        self.magnitude_history.append(magnitude)
        self.last_moisture = moisture
        self.last_magnitude = magnitude

        # Calculate rolling averages
        moisture_avg5 = np.mean(self.moisture_history)

        return {
            'accel_x': accel_x,
            'accel_y': accel_y,
            'moisture': moisture,
            'vibration': vibration,
            'accel_magnitude': magnitude,
            'moisture_avg5': moisture_avg5,
            'moisture_change': moisture_change,
            'magnitude_change': magnitude_change
        }

buffer = SensorBuffer(window_size=5)
print("✅ Buffer created (window_size=5)")
print("   This calculates LIVE features —")
print("   no pre-computed CSV columns needed!")

# ════════════════════════════════════════════
# PART 2: ALERT COOLDOWN SYSTEM
# ════════════════════════════════════════════
print("\n📌 PART 2: Building Alert Cooldown System")

class AlertManager:
    """Prevents alert spam with cooldown timer"""
    def __init__(self, cooldown_readings=10):
        self.cooldown_readings = cooldown_readings
        self.readings_since_alert = cooldown_readings
        self.total_alerts_sent = 0

    def should_alert(self, risk_level):
        if risk_level < 2:  # Not danger
            self.readings_since_alert += 1
            return False

        if self.readings_since_alert >= self.cooldown_readings:
            self.readings_since_alert = 0
            self.total_alerts_sent += 1
            return True
        else:
            self.readings_since_alert += 1
            return False

alert_manager = AlertManager(cooldown_readings=10)
print("✅ Alert manager created")
print("   Cooldown: 10 readings (~5 minutes)")

# ════════════════════════════════════════════
# PART 3: RISK STATE TRACKER
# ════════════════════════════════════════════
print("\n📌 PART 3: Building Risk State Tracker")

class RiskStateTracker:
    """Tracks if risk is improving or worsening"""
    def __init__(self, history_size=5):
        self.risk_history = deque(maxlen=history_size)

    def update(self, risk_level):
        self.risk_history.append(risk_level)

    def get_trend(self):
        if len(self.risk_history) < 3:
            return "GATHERING_DATA"

        recent = list(self.risk_history)
        if recent[-1] > recent[0]:
            return "WORSENING ⚠️"
        elif recent[-1] < recent[0]:
            return "IMPROVING ✅"
        else:
            return "STABLE"

risk_tracker = RiskStateTracker()
print("✅ Risk tracker created")

# ════════════════════════════════════════════
# PART 4: STREAMING SIMULATION
# ════════════════════════════════════════════
print("\n📌 PART 4: Starting Streaming Simulation")
print("(Simulating ESP32 sending data every reading)")
print("\n" + "-"*70)

# Simulate streaming through 100 readings
# (full 2000 would take too long to display)
simulation_log = []
labels_map = {0: '🟢 SAFE', 1: '🟡 WARNING', 2: '🔴 DANGER'}

# Pick readings around a known danger event for demo
demo_start = 0
demo_count = 50

for idx in range(demo_start, demo_start + demo_count):
    row = raw_df.iloc[idx]

    # Step 1: Add to buffer, get live features
    live_features = buffer.add_reading(
        row['accel_x'], row['accel_y'], row['accel_z'],
        row['moisture'], row['vibration']
    )

    # Step 2: Predict using production model
    reading_df = pd.DataFrame([live_features])[features]
    risk_pred = rf_model.predict(reading_df)[0]
    risk_proba = rf_model.predict_proba(reading_df)[0]

    reading_scaled = scaler.transform(reading_df)
    is_anomaly = iso_forest.predict(reading_scaled)[0] == -1

    # Step 3: Update risk tracker
    risk_tracker.update(risk_pred)
    trend = risk_tracker.get_trend()

    # Step 4: Check if alert needed
    alert_triggered = alert_manager.should_alert(risk_pred)

    # Log this "live" reading
    log_entry = {
        'reading_num': idx,
        'risk': labels_map[risk_pred],
        'confidence': round(risk_proba[risk_pred]*100, 1),
        'anomaly': is_anomaly,
        'trend': trend,
        'alert_sent': alert_triggered
    }
    simulation_log.append(log_entry)

    # Print live updates for interesting readings
    if risk_pred >= 1 or is_anomaly or alert_triggered:
        print(f"Reading #{idx:4d} | {labels_map[risk_pred]} "
              f"({risk_proba[risk_pred]*100:.0f}%) | "
              f"Anomaly: {'YES' if is_anomaly else 'no '} | "
              f"Trend: {trend:15s} | "
              f"{'🚨 ALERT SENT!' if alert_triggered else ''}")

print("-"*70)

# ════════════════════════════════════════════
# PART 5: SIMULATION SUMMARY
# ════════════════════════════════════════════
print("\n📌 PART 5: Simulation Summary")

log_df = pd.DataFrame(simulation_log)
risk_counts = log_df['risk'].value_counts()

print(f"\nReadings processed: {len(log_df)}")
print(f"\nRisk distribution during simulation:")
for risk, count in risk_counts.items():
    print(f"  {risk}: {count}")

print(f"\nAnomalies detected: {log_df['anomaly'].sum()}")
print(f"Alerts sent: {alert_manager.total_alerts_sent}")
print(f"(Cooldown prevented spam — "
      f"without it would be "
      f"{(log_df['risk']=='🔴 DANGER').sum()} alerts!)")

# ════════════════════════════════════════════
# PART 6: SIMULATE A DANGER EVENT IN DETAIL
# ════════════════════════════════════════════
print("\n📌 PART 6: Detailed Look at a Danger Event")

# Reset buffer and find a real danger event
buffer2 = SensorBuffer(window_size=5)
tracker2 = RiskStateTracker()

# Find first danger event location
danger_locations = raw_df[
    raw_df['moisture'] > 3500
].index.tolist()

if danger_locations:
    event_idx = danger_locations[0]
    start = max(0, event_idx - 8)
    end = min(len(raw_df), event_idx + 3)

    print(f"\nSimulating readings {start} to {end}")
    print(f"(Danger event happens at reading {event_idx})\n")

    for idx in range(start, end):
        row = raw_df.iloc[idx]
        live_features = buffer2.add_reading(
            row['accel_x'], row['accel_y'], row['accel_z'],
            row['moisture'], row['vibration']
        )
        reading_df = pd.DataFrame([live_features])[features]
        risk_pred = rf_model.predict(reading_df)[0]
        risk_proba = rf_model.predict_proba(reading_df)[0]
        tracker2.update(risk_pred)

        marker = " ⬅️ DANGER EVENT!" if idx == event_idx else ""
        print(f"  Reading {idx}: moisture={row['moisture']:4.0f} "
              f"→ {labels_map[risk_pred]} "
              f"({risk_proba[risk_pred]*100:.0f}%) "
              f"trend={tracker2.get_trend()}{marker}")

# ════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════
print("\n" + "="*55)
print("   DAY 17 COMPLETE — REAL-TIME SIMULATION DONE")
print("="*55)
print(f"\n🔧 Components Built:")
print(f"  ✅ SensorBuffer — live feature calculation")
print(f"  ✅ AlertManager — cooldown spam prevention")
print(f"  ✅ RiskStateTracker — trend detection")
print(f"\n📊 Simulation Results:")
print(f"  Readings processed: {demo_count}")
print(f"  Alerts sent (with cooldown): "
      f"{alert_manager.total_alerts_sent}")
print(f"\n💡 Why This Matters:")
print(f"  This EXACT code structure will run")
print(f"  on Day 33 when real ESP32 connects!")
print(f"  Buffer + Model + Alert = Complete pipeline ✅")
print(f"\n🚀 Next: Day 18 — Flask API (make this a web service!)")
print("="*55)