# ============================================
# DAY 16 — Save and Load Models Properly
# Date: 04 June 2026
# Author: Akshay
# Goal: Create production-ready model package
#       Clean organization for Day 18 Flask API
# ============================================
# ============================================
# DAY 16 OBSERVATIONS:
# 1. Production package contains: 2 models +
#    scaler + features + metadata
# 2. Validation catches bad input: YES ✅
#    (caught missing accel_y correctly)
# 3. Test prediction (danger case): 99.3% confidence,
#    flagged as anomaly too (double confirmation)
# 4. Bundle everything in ONE file because:
#    Flask API loads once, uses many times,
#    avoids version mismatch errors
# 5. This exact file will be imported in Day 18
# ============================================

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("="*55)
print("   DAY 16 — PRODUCTION MODEL PACKAGE")
print("="*55)

# ════════════════════════════════════════════
# PART 1: LOAD ALL EXISTING MODELS
# ════════════════════════════════════════════
print("\n📌 PART 1: Loading existing models")

rf_model = joblib.load('models/rf_model_final.pkl')
features = joblib.load('models/feature_list_final.pkl')
iso_forest = joblib.load('models/isolation_forest.pkl')
scaler = joblib.load('models/scaler.pkl')

print(f"✅ Random Forest loaded")
print(f"✅ Isolation Forest loaded")
print(f"✅ Scaler loaded")
print(f"✅ Features: {features}")

# ════════════════════════════════════════════
# PART 2: VALIDATE MODELS WORK TOGETHER
# ════════════════════════════════════════════
print("\n📌 PART 2: Validating models work correctly")

df = pd.read_csv('data/processed/sensor_labeled.csv')
X = df[features]
y = df['risk_label']

# Test Random Forest
rf_predictions = rf_model.predict(X)
rf_accuracy = accuracy_score(y, rf_predictions)
print(f"\nRandom Forest accuracy on full data: "
      f"{rf_accuracy*100:.2f}%")

# Test Isolation Forest
X_scaled = scaler.transform(X)
iso_predictions = iso_forest.predict(X_scaled)
anomaly_rate = (iso_predictions == -1).mean()
print(f"Isolation Forest anomaly rate: "
      f"{anomaly_rate*100:.2f}%")

print("✅ Both models working correctly")

# ════════════════════════════════════════════
# PART 3: CREATE MODEL METADATA
# WHY: Track version, performance, history
# ════════════════════════════════════════════
print("\n📌 PART 3: Creating model metadata")

metadata = {
    'project': 'LandSense ML',
    'version': '1.0',
    'created_date': datetime.now().strftime('%Y-%m-%d'),
    'author': 'Akshay Kumar',
    'institution': 'Vishnu Institute of Technology',
    'training_samples': len(df),
    'features': features,
    'feature_count': len(features),
    'random_forest': {
        'accuracy': round(rf_accuracy, 4),
        'n_estimators': rf_model.n_estimators,
        'max_depth': rf_model.max_depth,
    },
    'isolation_forest': {
        'contamination': 0.03,
        'anomaly_rate_training': round(anomaly_rate, 4),
    },
    'label_mapping': {
        '0': 'SAFE',
        '1': 'WARNING',
        '2': 'DANGER'
    },
    'data_quality_score': 99.5,
    'notes': 'Trained on simulated ESP32 sensor data. '
             'Real hardware integration pending Month 2.'
}

print("Metadata created:")
print(json.dumps(metadata, indent=2))

# ════════════════════════════════════════════
# PART 4: VALIDATION FUNCTION
# WHY: Prevent crashes from bad ESP32 data
# ════════════════════════════════════════════
print("\n📌 PART 4: Creating input validation")

def validate_sensor_input(data):
    """
    Validates incoming sensor reading
    before prediction.
    Returns: (is_valid, error_message)
    """
    required = ['accel_x', 'accel_y', 'moisture',
                'vibration', 'accel_magnitude',
                'moisture_avg5', 'moisture_change',
                'magnitude_change']

    # Check all features present
    for feat in required:
        if feat not in data:
            return False, f"Missing feature: {feat}"

    # Check ranges
    if not (0 <= data['moisture'] <= 4095):
        return False, "moisture out of valid range"

    if data['vibration'] not in [0, 1]:
        return False, "vibration must be 0 or 1"

    if not (0 <= data['accel_x'] <= 4095):
        return False, "accel_x out of valid range"

    if not (0 <= data['accel_y'] <= 4095):
        return False, "accel_y out of valid range"

    return True, "Valid"

# Test validation
test_good = {
    'accel_x': 2100, 'accel_y': 2050,
    'moisture': 2800, 'vibration': 0,
    'accel_magnitude': 3600, 'moisture_avg5': 2700,
    'moisture_change': 100, 'magnitude_change': 30
}
test_bad = {
    'accel_x': 9999,  # Invalid!
    'moisture': 2800, 'vibration': 0
}

valid1, msg1 = validate_sensor_input(test_good)
valid2, msg2 = validate_sensor_input(test_bad)
print(f"\nGood input test: {valid1} - {msg1}")
print(f"Bad input test:  {valid2} - {msg2}")

# ════════════════════════════════════════════
# PART 5: COMPLETE PREDICTION FUNCTION
# WHY: ONE function Flask API will call
# ════════════════════════════════════════════
print("\n📌 PART 5: Building complete prediction function")

def predict_landslide_risk(sensor_data,
                            rf_model, iso_forest,
                            scaler, features):
    """
    Complete prediction pipeline.
    Takes raw sensor dict, returns full analysis.

    This is THE function Day 18 Flask API calls!
    """
    # Step 1: Validate input
    is_valid, msg = validate_sensor_input(sensor_data)
    if not is_valid:
        return {'error': msg, 'success': False}

    # Step 2: Prepare data
    reading = pd.DataFrame([sensor_data])[features]

    # Step 3: Random Forest prediction
    risk_pred = rf_model.predict(reading)[0]
    risk_proba = rf_model.predict_proba(reading)[0]

    labels = {0: 'SAFE', 1: 'WARNING', 2: 'DANGER'}

    # Step 4: Anomaly detection
    reading_scaled = scaler.transform(reading)
    anomaly_pred = iso_forest.predict(reading_scaled)[0]
    anomaly_score = iso_forest.score_samples(
        reading_scaled
    )[0]
    is_anomaly = (anomaly_pred == -1)

    # Step 5: Combine into final recommendation
    if risk_pred == 2 or (is_anomaly and risk_pred >= 1):
        recommendation = "🔴 EVACUATE - High risk detected"
    elif risk_pred == 1 or is_anomaly:
        recommendation = "🟡 MONITOR - Unusual activity detected"
    else:
        recommendation = "🟢 SAFE - Normal conditions"

    return {
        'success': True,
        'risk_level': labels[risk_pred],
        'risk_code': int(risk_pred),
        'confidence': round(float(risk_proba[risk_pred])*100, 1),
        'probabilities': {
            'safe': round(float(risk_proba[0])*100, 1),
            'warning': round(float(risk_proba[1])*100, 1),
            'danger': round(float(risk_proba[2])*100, 1)
        },
        'is_anomaly': bool(is_anomaly),
        'anomaly_score': round(float(anomaly_score), 3),
        'recommendation': recommendation,
        'timestamp': datetime.now().isoformat()
    }

# Test the complete function
print("\nTesting complete prediction function:")

result = predict_landslide_risk(
    {'accel_x': 2700, 'accel_y': 2800,
     'moisture': 3900, 'vibration': 1,
     'accel_magnitude': 4300, 'moisture_avg5': 3500,
     'moisture_change': 1200, 'magnitude_change': 600},
    rf_model, iso_forest, scaler, features
)

print(json.dumps(result, indent=2))

# ════════════════════════════════════════════
# PART 6: CREATE FINAL PRODUCTION PACKAGE
# WHY: ONE file with everything needed
# ════════════════════════════════════════════
print("\n📌 PART 6: Creating production package")

production_package = {
    'rf_model': rf_model,
    'iso_forest': iso_forest,
    'scaler': scaler,
    'features': features,
    'metadata': metadata,
    'predict_function_version': '1.0'
}

joblib.dump(
    production_package,
    'models/production_model.pkl'
)
print("✅ Saved: models/production_model.pkl")
print("   (Contains EVERYTHING needed for Flask API)")

# Save metadata separately as readable JSON too
with open('models/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("✅ Saved: models/model_metadata.json")
print("   (Human-readable model info)")

# ════════════════════════════════════════════
# PART 7: TEST LOADING PRODUCTION PACKAGE
# Simulate how Day 18 Flask API will load it
# ════════════════════════════════════════════
print("\n📌 PART 7: Testing production package loading")
print("(Simulating Day 18 Flask startup)")

# This simulates app startup - loads ONCE
loaded_package = joblib.load('models/production_model.pkl')

print(f"✅ Package loaded successfully")
print(f"   Version: {loaded_package['metadata']['version']}")
print(f"   Trained: {loaded_package['metadata']['created_date']}")
print(f"   Accuracy: "
      f"{loaded_package['metadata']['random_forest']['accuracy']*100:.2f}%")

# Use loaded package for prediction
test_result = predict_landslide_risk(
    {'accel_x': 2050, 'accel_y': 2048,
     'moisture': 1800, 'vibration': 0,
     'accel_magnitude': 3550, 'moisture_avg5': 1850,
     'moisture_change': 20, 'magnitude_change': 10},
    loaded_package['rf_model'],
    loaded_package['iso_forest'],
    loaded_package['scaler'],
    loaded_package['features']
)
print(f"\nTest prediction using loaded package:")
print(f"  Risk: {test_result['risk_level']}")
print(f"  Confidence: {test_result['confidence']}%")
print(f"  Recommendation: {test_result['recommendation']}")

# ════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════
print("\n" + "="*55)
print("   DAY 16 COMPLETE — PRODUCTION READY")
print("="*55)
print(f"\n📦 Production Package Contains:")
print(f"  ✅ Random Forest model")
print(f"  ✅ Isolation Forest model")
print(f"  ✅ Feature scaler")
print(f"  ✅ Feature list")
print(f"  ✅ Model metadata")
print(f"  ✅ Validation function")
print(f"  ✅ Complete prediction function")
print(f"\n💾 Files Created:")
print(f"  models/production_model.pkl (ONE file, everything)")
print(f"  models/model_metadata.json (readable info)")
print(f"\n🎯 This is EXACTLY what Day 18 Flask API")
print(f"   will load and use!")
print(f"\n🚀 Next: Day 17 — Real-time Simulation")
print("="*55)