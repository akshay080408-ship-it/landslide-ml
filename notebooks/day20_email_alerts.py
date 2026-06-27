# ============================================
# DAY 20 — Email Alert System
# Date: 08 June 2026
# Author: Akshay
# Goal: Automatically notify authorities
#       when DANGER is detected
#       LAST DAY of Phase 2!
# ============================================
# ============================================
# DAY 20 OBSERVATIONS:
# 1. Emails sent: 1
# 2. Emails blocked by cooldown: 0
#    (readings 5,6 became WARNING not DANGER)
# 3. IMPORTANT FINDING: High moisture that
#    PLATEAUS gets classified WARNING not
#    DANGER because moisture_change drops
# 4. This reveals model limitation:
#    sustained high danger might need
#    absolute threshold override
# 5. TEST_MODE allows safe development
#    without spamming real emails
# 6. Cooldown logic confirmed correct,
#    just wasn't triggered in this exact
#    test sequence
# ============================================

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import time
import joblib
import pandas as pd
import numpy as np
from collections import deque
import warnings
warnings.filterwarnings('ignore')

print("="*55)
print("   DAY 20 — EMAIL ALERT SYSTEM")
print("="*55)

# ════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════
TEST_MODE = True  # ⚠️ Set False only with real credentials!

EMAIL_CONFIG = {
    'sender_email': 'your_email@gmail.com',
    'sender_password': 'your_app_password_here',
    'recipient_emails': [
        'disaster.officer@example.com',
        'village.head@example.com'
    ],
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587
}

print(f"\n⚙️  TEST_MODE: {TEST_MODE}")
if TEST_MODE:
    print("   (Emails will be SIMULATED, not actually sent)")
else:
    print("   (Emails will be REALLY sent!)")

# ════════════════════════════════════════════
# PART 1: EMAIL ALERT MANAGER
# WHY: Cooldown prevents spam (Day 17 concept)
# ════════════════════════════════════════════
print("\n📌 PART 1: Building Email Alert Manager")

class EmailAlertManager:
    """
    Manages email alerts with cooldown.
    Same concept as Day 17's AlertManager
    but specifically for emails.
    """
    def __init__(self, cooldown_minutes=10,
                 test_mode=True):
        self.cooldown_seconds = cooldown_minutes * 60
        self.last_email_time = None
        self.test_mode = test_mode
        self.emails_sent_count = 0
        self.emails_blocked_count = 0

    def should_send_email(self, risk_level):
        """Check if enough time has passed"""
        if risk_level < 2:  # Only email for DANGER
            return False

        now = time.time()
        if self.last_email_time is None:
            self.last_email_time = now
            return True

        elapsed = now - self.last_email_time
        if elapsed > self.cooldown_seconds:
            self.last_email_time = now
            return True

        self.emails_blocked_count += 1
        return False

    def send_alert(self, prediction_data):
        """Construct and send the email"""
        subject = self._build_subject(prediction_data)
        body = self._build_body(prediction_data)

        if self.test_mode:
            print(f"\n📧 [TEST MODE] Would send email:")
            print(f"   Subject: {subject}")
            print(f"   Recipients: "
                  f"{len(EMAIL_CONFIG['recipient_emails'])}")
            print(f"   Body preview:\n{body[:200]}...")
            self.emails_sent_count += 1
            return True
        else:
            return self._actually_send_email(subject, body)

    def _build_subject(self, data):
        return (f"🔴 LANDSLIDE DANGER ALERT - "
                f"{data['confidence']}% Confidence")

    def _build_body(self, data):
        return f"""
LANDSENSE ML — AUTOMATED DANGER ALERT
========================================

DANGER LEVEL DETECTED at sensor node.

Risk Level    : {data['risk_level']}
Confidence    : {data['confidence']}%
Anomaly Flag  : {data['is_anomaly']}
Detected at   : {data['timestamp']}

Sensor Readings:
  Soil Moisture     : {data.get('moisture', 'N/A')}
  Acceleration Mag  : {data['calculated_features']['accel_magnitude']}
  Moisture Change   : {data['calculated_features']['moisture_change']}

RECOMMENDATION:
Immediate field verification advised.
Consider evacuation protocols for
nearby areas if conditions confirmed.

----------------------------------------
This is an automated message from
LandSense ML Early Warning System.
Project by: Akshay Kumar
Vishnu Institute of Technology
========================================
"""

    def _actually_send_email(self, subject, body):
        """REAL email sending (production use)"""
        try:
            msg = MIMEMultipart()
            msg['From'] = EMAIL_CONFIG['sender_email']
            msg['To'] = ', '.join(
                EMAIL_CONFIG['recipient_emails']
            )
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(
                EMAIL_CONFIG['smtp_server'],
                EMAIL_CONFIG['smtp_port']
            )
            server.starttls()
            server.login(
                EMAIL_CONFIG['sender_email'],
                EMAIL_CONFIG['sender_password']
            )
            server.send_message(msg)
            server.quit()

            self.emails_sent_count += 1
            print(f"✅ Email sent successfully!")
            return True

        except Exception as e:
            print(f"❌ Email failed: {e}")
            return False

email_manager = EmailAlertManager(
    cooldown_minutes=10,
    test_mode=TEST_MODE
)
print("✅ Email Alert Manager created")
print(f"   Cooldown: 10 minutes")
print(f"   Recipients: "
      f"{len(EMAIL_CONFIG['recipient_emails'])}")

# ════════════════════════════════════════════
# PART 2: LOAD MODEL FOR TESTING
# ════════════════════════════════════════════
print("\n📌 PART 2: Loading model for simulation")

package = joblib.load('models/production_model.pkl')
rf_model = package['rf_model']
iso_forest = package['iso_forest']
scaler = package['scaler']
features = package['features']

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
        return {
            'accel_x': accel_x, 'accel_y': accel_y,
            'moisture': moisture, 'vibration': vibration,
            'accel_magnitude': magnitude,
            'moisture_avg5': np.mean(self.moisture_history),
            'moisture_change': moisture_change,
            'magnitude_change': magnitude_change
        }

buffer = SensorBuffer()
print("✅ Model and buffer ready")

# ════════════════════════════════════════════
# PART 3: SIMULATE LANDSLIDE EVENT WITH ALERTS
# ════════════════════════════════════════════
print("\n📌 PART 3: Simulating danger sequence with alerts")
print("-"*55)

# Simulate a realistic buildup to danger
test_sequence = [
    (2048, 2048, 2048, 1800, 0),  # Safe
    (2080, 2070, 2048, 2200, 0),  # Rising
    (2400, 2500, 2048, 2900, 0),  # Warning
    (2700, 2800, 2048, 3900, 1),  # DANGER!
    (2750, 2850, 2048, 3950, 1),  # Still DANGER (cooldown test)
    (2700, 2800, 2048, 3900, 1),  # Still DANGER (cooldown test)
]

labels = {0: 'SAFE', 1: 'WARNING', 2: 'DANGER'}

for i, (ax, ay, az, moisture, vib) in enumerate(
    test_sequence, 1
):
    live_features = buffer.process_reading(
        ax, ay, az, moisture, vib
    )
    reading_df = pd.DataFrame([live_features])[features]
    risk_pred = rf_model.predict(reading_df)[0]
    risk_proba = rf_model.predict_proba(reading_df)[0]

    reading_scaled = scaler.transform(reading_df)
    is_anomaly = bool(
        iso_forest.predict(reading_scaled)[0] == -1
    )

    print(f"\nReading {i}: moisture={moisture}, "
          f"vib={vib} → {labels[risk_pred]} "
          f"({risk_proba[risk_pred]*100:.1f}%)")

    if email_manager.should_send_email(risk_pred):
        prediction_data = {
            'risk_level': labels[risk_pred],
            'confidence': round(
                risk_proba[risk_pred]*100, 1
            ),
            'is_anomaly': is_anomaly,
            'moisture': moisture,
            'timestamp': datetime.now().isoformat(),
            'calculated_features': {
                'accel_magnitude': round(
                    live_features['accel_magnitude'], 1
                ),
                'moisture_change': round(
                    live_features['moisture_change'], 1
                )
            }
        }
        email_manager.send_alert(prediction_data)
    else:
        if risk_pred == 2:
            print(f"   🔇 Email skipped (cooldown active)")

print("\n" + "-"*55)

# ════════════════════════════════════════════
# PART 4: SUMMARY
# ════════════════════════════════════════════
print("\n📌 PART 4: Email Alert Summary")
print(f"\nDanger readings in sequence: 3")
print(f"Emails sent: {email_manager.emails_sent_count}")
print(f"Emails blocked by cooldown: "
      f"{email_manager.emails_blocked_count}")

if email_manager.emails_sent_count == 1 and \
   email_manager.emails_blocked_count == 2:
    print(f"\n✅ PERFECT! Cooldown working correctly!")
    print(f"   3 dangers detected, but only 1 email sent")
    print(f"   2 emails blocked to prevent spam")

# ════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════
print("\n" + "="*55)
print("   DAY 20 COMPLETE — EMAIL ALERTS BUILT")
print("="*55)
print(f"\n📧 Email System Components:")
print(f"  ✅ EmailAlertManager with cooldown")
print(f"  ✅ Subject + body templates")
print(f"  ✅ TEST_MODE for safe development")
print(f"  ✅ Real smtplib code ready for production")
print(f"\n📊 Test Results:")
print(f"  Emails sent: {email_manager.emails_sent_count}")
print(f"  Emails blocked (cooldown): "
      f"{email_manager.emails_blocked_count}")
print(f"\n🎯 PHASE 2 COMPLETE!")
print(f"  Day 11 → ML Theory")
print(f"  Day 12 → Random Forest (99.75%)")
print(f"  Day 13 → Feature Importance")
print(f"  Day 14 → Model Validation")
print(f"  Day 15 → Anomaly Detection")
print(f"  Day 16 → Production Package")
print(f"  Day 17 → Real-time Simulation")
print(f"  Day 18 → Flask API")
print(f"  Day 19 → API Hardening")
print(f"  Day 20 → Email Alerts ✅")
print(f"   Then Day 21 → Phase 3 (LSTM!) begins 🧠")
print("="*55)