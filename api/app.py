# ============================================
# DAY 19 — Flask API Hardened (v2)
# Date: 07 June 2026
# Author: Akshay
# Goal: Make API survive bad data,
#       add logging, rate limiting
# ============================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from collections import deque
import logging
import os

# Add parent directory to path so we can
# access models folder from api folder
sys.path.append(
    os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    ))
)

print("="*55)
print("   DAY 19 — HARDENED FLASK API")
print("="*55)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ════════════════════════════════════════════
# STEP 1: CREATE FLASK APP
# ════════════════════════════════════════════
app = Flask(__name__)

# ════════════════════════════════════════════
# STEP 2: LOAD MODEL (same as Day 18)
# ════════════════════════════════════════════
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    )),
    'models', 'production_model.pkl'
)

logger.info("Loading production model...")
package = joblib.load(MODEL_PATH)
rf_model = package['rf_model']
iso_forest = package['iso_forest']
scaler = package['scaler']
features = package['features']
metadata = package['metadata']
logger.info(f"Model loaded - version {metadata['version']}")

# ════════════════════════════════════════════
# STEP 3: SENSOR BUFFER (same as Day 18)
# ════════════════════════════════════════════
class SensorBuffer:
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
            'accel_x': accel_x, 'accel_y': accel_y,
            'moisture': moisture, 'vibration': vibration,
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
    'total_predictions': 0, 'danger_alerts': 0,
    'warning_alerts': 0, 'safe_readings': 0,
    'anomalies_detected': 0, 'errors_caught': 0,
    'start_time': datetime.now().isoformat()
}

# ════════════════════════════════════════════
# ROUTE 1: HOME — Welcome message
# ════════════════════════════════════════════

@app.route('/')
def home():
    return jsonify({
        'project': 'LandSense ML API (Hardened v2)',
        'status': 'running',
        'endpoints': {
            '/': 'This page',
            '/health': 'Check API health',
            '/predict': 'POST sensor data for prediction',
            '/stats': 'View API usage statistics',
            '/api/alerts': 'View recent DANGER alert history'
        }
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'model_version': metadata['version'],
        'model_accuracy': metadata['random_forest']['accuracy'],
        'features_expected': features,
        'errors_caught_so_far': api_stats['errors_caught'],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    # Rate limiting check
    if not rate_limiter.allow_request():
        logger.warning("Rate limit exceeded")
        return jsonify({
            'success': False,
            'error': 'Too many requests. '
                     'Please wait before retrying.'
        }), 429  # 429 = Too Many Requests

    try:
        data = request.get_json(silent=True)

        # Validate thoroughly
        is_valid, error_msg = validate_input(data)
        if not is_valid:
            logger.warning(f"Invalid input: {error_msg}")
            api_stats['errors_caught'] += 1
            return jsonify({
                'success': False,
                'error': error_msg
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

        reading_df = pd.DataFrame(
            [live_features]
        )[features]

        risk_pred = rf_model.predict(reading_df)[0]
        risk_proba = rf_model.predict_proba(reading_df)[0]
        labels = {0: 'SAFE', 1: 'WARNING', 2: 'DANGER'}

        reading_scaled = scaler.transform(reading_df)
        is_anomaly = bool(
            iso_forest.predict(reading_scaled)[0] == -1
        )

        api_stats['total_predictions'] += 1
        email_sent = False
        email_blocked = False
        sms_sent = False
        sms_blocked = False
        if risk_pred == 2:
            api_stats['danger_alerts'] += 1
        elif risk_pred == 1:
            api_stats['warning_alerts'] += 1
        else:
            api_stats['safe_readings'] += 1
        if is_anomaly:
            api_stats['anomalies_detected'] += 1

        response = {
            'success': True,
            'node_id': node_id,
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
            'email_sent': email_sent,
            'email_blocked_by_cooldown': email_blocked,
            'sms_sent': sms_sent,
            'sms_blocked_by_cooldown': sms_blocked,
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
        logger.error(f"Unexpected error: {str(e)}")
        api_stats['errors_caught'] += 1
        return jsonify({
            'success': False,
            'error': f'Internal error: {str(e)}'
        }), 500

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