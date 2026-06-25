# ============================================
# DAY 18 — Flask API Part 1
# Date: 06 June 2026
# Author: Akshay
# Goal: Build web API for landslide prediction
#       ESP32 will call this in Day 33!
# ============================================# ============================================
# DAY 18 OBSERVATIONS:
# 1. All 4 routes working: /, /health, /stats, /predict
# 2. First reading always SAFE (no history yet)
# 3. moisture_change builds up across readings
# 4. Sequence test: SAFE → SAFE(lower conf) → DANGER
# 5. This proves buffer + model work together
#    exactly as designed in Day 17
# 6. API accessible on network IP for future ESP32
# ============================================

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from collections import deque
import sys
import os

# Add parent directory to path so we can
# access models folder from api folder
sys.path.append(
    os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    ))
)

print("="*55)
print("   DAY 18 — LANDSENSE FLASK API")
print("="*55)

# ════════════════════════════════════════════
# STEP 1: CREATE FLASK APP
# ════════════════════════════════════════════
app = Flask(__name__)

# ════════════════════════════════════════════
# STEP 2: LOAD MODEL ONCE AT STARTUP
# (Critical! Not inside route functions!)
# ════════════════════════════════════════════
print("\n📦 Loading production model package...")

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    )),
    'models', 'production_model.pkl'
)
package = joblib.load(MODEL_PATH)

rf_model = package['rf_model']
iso_forest = package['iso_forest']
scaler = package['scaler']
features = package['features']
metadata = package['metadata']

print(f"✅ Model loaded successfully!")
print(f"   Version: {metadata['version']}")
print(f"   Accuracy: "
      f"{metadata['random_forest']['accuracy']*100:.2f}%")
print(f"   Features: {len(features)}")

# ════════════════════════════════════════════
# STEP 3: SENSOR BUFFER (from Day 17)
# Maintains history for live feature calc
# ════════════════════════════════════════════
class SensorBuffer:
    """Same buffer logic from Day 17"""
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.moisture_history = deque(maxlen=window_size)
        self.last_moisture = None
        self.last_magnitude = None

    def process_reading(self, accel_x, accel_y,
                         accel_z, moisture, vibration):
        magnitude = np.sqrt(
            accel_x**2 + accel_y**2 + accel_z**2
        )
        moisture_change = (
            moisture - self.last_moisture
            if self.last_moisture is not None else 0
        )
        magnitude_change = (
            magnitude - self.last_magnitude
            if self.last_magnitude is not None else 0
        )
        self.moisture_history.append(moisture)
        self.last_moisture = moisture
        self.last_magnitude = magnitude
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

# Global buffer (persists across requests)
sensor_buffer = SensorBuffer(window_size=5)

# ════════════════════════════════════════════
# STEP 4: API STATISTICS TRACKER
# ════════════════════════════════════════════
api_stats = {
    'total_predictions': 0,
    'danger_alerts': 0,
    'warning_alerts': 0,
    'safe_readings': 0,
    'anomalies_detected': 0,
    'start_time': datetime.now().isoformat()
}

# ════════════════════════════════════════════
# ROUTE 1: HOME — Welcome message
# ════════════════════════════════════════════
@app.route('/')
def home():
    return jsonify({
        'project': 'LandSense ML API',
        'status': 'running',
        'message': 'Landslide Early Warning System',
        'author': 'Akshay Kumar',
        'endpoints': {
            '/': 'This page',
            '/health': 'Check API health',
            '/predict': 'POST sensor data for prediction',
            '/stats': 'View API usage statistics'
        }
    })

# ════════════════════════════════════════════
# ROUTE 2: HEALTH CHECK
# ════════════════════════════════════════════
@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'model_version': metadata['version'],
        'model_accuracy': metadata['random_forest']['accuracy'],
        'features_expected': features,
        'timestamp': datetime.now().isoformat()
    })

# ════════════════════════════════════════════
# ROUTE 3: PREDICT — Main prediction endpoint
# THIS IS THE MOST IMPORTANT ROUTE!
# ════════════════════════════════════════════
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Step 1: Get JSON data from request
        data = request.get_json()

        if data is None:
            return jsonify({
                'success': False,
                'error': 'No JSON data received'
            }), 400

        # Step 2: Validate required raw fields
        required_raw = ['accel_x', 'accel_y',
                        'accel_z', 'moisture',
                        'vibration']
        for field in required_raw:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing field: {field}'
                }), 400

        # Step 3: Process through buffer
        # (calculates magnitude, rolling avg, etc)
        live_features = sensor_buffer.process_reading(
            data['accel_x'], data['accel_y'],
            data['accel_z'], data['moisture'],
            data['vibration']
        )

        # Step 4: Prepare for model
        reading_df = pd.DataFrame(
            [live_features]
        )[features]

        # Step 5: Random Forest prediction
        risk_pred = rf_model.predict(reading_df)[0]
        risk_proba = rf_model.predict_proba(reading_df)[0]

        labels = {0: 'SAFE', 1: 'WARNING', 2: 'DANGER'}

        # Step 6: Anomaly detection
        reading_scaled = scaler.transform(reading_df)
        anomaly_pred = iso_forest.predict(
            reading_scaled
        )[0]
        is_anomaly = bool(anomaly_pred == -1)

        # Step 7: Update statistics
        api_stats['total_predictions'] += 1
        if risk_pred == 2:
            api_stats['danger_alerts'] += 1
        elif risk_pred == 1:
            api_stats['warning_alerts'] += 1
        else:
            api_stats['safe_readings'] += 1
        if is_anomaly:
            api_stats['anomalies_detected'] += 1

        # Step 8: Build response
        response = {
            'success': True,
            'risk_level': labels[risk_pred],
            'risk_code': int(risk_pred),
            'confidence': round(
                float(risk_proba[risk_pred])*100, 1
            ),
            'probabilities': {
                'safe': round(float(risk_proba[0])*100, 1),
                'warning': round(float(risk_proba[1])*100, 1),
                'danger': round(float(risk_proba[2])*100, 1)
            },
            'is_anomaly': is_anomaly,
            'calculated_features': {
                'accel_magnitude': round(
                    live_features['accel_magnitude'], 1
                ),
                'moisture_avg5': round(
                    live_features['moisture_avg5'], 1
                ),
                'moisture_change': round(
                    live_features['moisture_change'], 1
                )
            },
            'timestamp': datetime.now().isoformat()
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ════════════════════════════════════════════
# ROUTE 4: STATISTICS
# ════════════════════════════════════════════
@app.route('/stats')
def stats():
    return jsonify(api_stats)

# ════════════════════════════════════════════
# RUN THE SERVER
# ════════════════════════════════════════════
if __name__ == '__main__':
    print("\n🚀 Starting Flask server...")
    print("   Local access:   http://127.0.0.1:5000")
    print("   Network access: http://0.0.0.0:5000")
    print("\n   Press CTRL+C to stop the server")
    print("="*55)

    app.run(
        host='0.0.0.0',  # Accessible from network
        port=5000,
        debug=True        # Auto-reload on changes
    )