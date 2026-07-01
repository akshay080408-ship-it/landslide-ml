# ============================================
# DAY 19 — Flask API Hardened (v2)
# Date: 07 June 2026
# Author: Akshay
# Goal: Make API survive bad data,
#       add logging, rate limiting
# ============================================

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from collections import deque
import logging
import os
import time

# ════════════════════════════════════════════
# STEP 1: SETUP LOGGING
# WHY: Track every request for debugging
# ════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../api/api_log.txt'),
        logging.StreamHandler()  # Also print to terminal
    ]
)
logger = logging.getLogger('LandSenseAPI')

print("="*55)
print("   DAY 19 — HARDENED FLASK API")
print("="*55)

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
    """
    Maintains a sliding window of recent sensor readings.
    Calculates live features (magnitude, rolling avg,
    rate of change) without needing full CSV history.
    Used for real-time ESP32 data processing.
    """
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

sensor_buffer = SensorBuffer(window_size=5)

api_stats = {
    'total_predictions': 0, 'danger_alerts': 0,
    'warning_alerts': 0, 'safe_readings': 0,
    'anomalies_detected': 0, 'errors_caught': 0,
    'start_time': datetime.now().isoformat()
}

# ════════════════════════════════════════════
# STEP 4: SIMPLE RATE LIMITER
# WHY: Prevent overload from buggy/malicious
#      rapid-fire requests
# ════════════════════════════════════════════
class SimpleRateLimiter:
    def __init__(self, min_interval_seconds=0.5):
        self.min_interval = min_interval_seconds
        self.last_request_time = 0

    def allow_request(self):
        now = time.time()
        if now - self.last_request_time < self.min_interval:
            return False
        self.last_request_time = now
        return True

rate_limiter = SimpleRateLimiter(min_interval_seconds=0.3)

# ════════════════════════════════════════════
# STEP 5: INPUT VALIDATION (hardened)
# WHY: Catch EVERY type of bad data
# ════════════════════════════════════════════
def validate_input(data):
    """
    Thoroughly validates sensor input.
    Returns (is_valid, error_message)
    """
    if data is None:
        return False, "No JSON data received"

    if not isinstance(data, dict):
        return False, "Data must be a JSON object"

    required = ['accel_x', 'accel_y', 'accel_z',
                'moisture', 'vibration']

    for field in required:
        if field not in data:
            return False, f"Missing required field: {field}"
        if data[field] is None:
            return False, f"Field '{field}' cannot be null"

    # Check types - must be numbers
    numeric_fields = ['accel_x', 'accel_y', 'accel_z',
                       'moisture']
    for field in numeric_fields:
        try:
            float(data[field])
        except (ValueError, TypeError):
            return False, (
                f"Field '{field}' must be a number, "
                f"got: {type(data[field]).__name__}"
            )

    # Check vibration is 0 or 1
    if data['vibration'] not in [0, 1, 0.0, 1.0]:
        return False, "vibration must be 0 or 1"

    # Check ranges (ESP32 12-bit ADC: 0-4095)
    for field in ['accel_x', 'accel_y', 'accel_z']:
        val = float(data[field])
        if not (0 <= val <= 4095):
            return False, (
                f"{field} out of range (0-4095): {val}"
            )

    moisture_val = float(data['moisture'])
    if not (0 <= moisture_val <= 4095):
        return False, (
            f"moisture out of range (0-4095): {moisture_val}"
        )

    return True, "Valid"

# ════════════════════════════════════════════
# ROUTES
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
            '/stats': 'View API usage statistics'
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

        # Process through buffer
        live_features = sensor_buffer.process_reading(
            float(data['accel_x']), float(data['accel_y']),
            float(data['accel_z']), float(data['moisture']),
            int(data['vibration'])
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
        if risk_pred == 2:
            api_stats['danger_alerts'] += 1
            logger.info(
                f"DANGER detected - "
                f"confidence: {risk_proba[2]*100:.1f}%"
            )
        elif risk_pred == 1:
            api_stats['warning_alerts'] += 1
        else:
            api_stats['safe_readings'] += 1
        if is_anomaly:
            api_stats['anomalies_detected'] += 1

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
        logger.info(
            f"Prediction: {labels[risk_pred]} "
            f"({response['confidence']}%)"
        )
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

if __name__ == '__main__':
    logger.info("Starting Flask server...")
    print("\n🚀 Starting Hardened Flask server...")
    print("   Local access:   http://127.0.0.1:5000")
    print("   Press CTRL+C to stop")
    print("="*55)
    app.run(host='0.0.0.0', port=5000, debug=True)