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
import sqlite3
import json
import time
import smtplib
import requests as http_requests  # named to avoid clashing with Flask's `request`
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ════════════════════════════════════════════
# STEP 1: SETUP LOGGING
# WHY: Track every request for debugging
# ════════════════════════════════════════════
LOG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'api_log.txt'
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('LandSenseAPI')

print("="*55)
print("   DAY 19 — HARDENED FLASK API")
print("="*55)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ════════════════════════════════════════════
# PERSISTENCE (SQLite)
# WHY: alert_history and field_reports previously
# lived only in memory — a server restart during
# testing (or, worse, mid-demo) would silently wipe
# every alert and every citizen report. SQLite is
# stdlib (no new dependency), file-based (no setup),
# and enough for a Sept 7 demo — a real production
# deployment would want Postgres/etc, but this closes
# the actual risk (data loss) with minimal complexity.
#
# Pattern: in-memory lists (alert_history,
# field_reports) stay exactly as the rest of the code
# already uses them — every .append() is now mirrored
# to SQLite right after, and on startup the lists are
# REPOPULATED from SQLite so a restart doesn't lose
# anything. Nothing else in the codebase needs to
# change how it reads alert_history/field_reports.
# ════════════════════════════════════════════
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'landsense.db')

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    logger.info(f"SQLite ready at {DB_PATH}")

def db_save(table, entry):
    """Insert one JSON-serializable dict into `table` ('alerts' or 'reports')."""
    conn = get_db()
    conn.execute(
        f"INSERT INTO {table} (data, created_at) VALUES (?, ?)",
        (json.dumps(entry), datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

def db_load_all(table):
    """Load every row from `table`, oldest first — same order the
    in-memory lists expect, since they're appended to chronologically."""
    conn = get_db()
    rows = conn.execute(f"SELECT data FROM {table} ORDER BY id ASC").fetchall()
    conn.close()
    return [json.loads(row['data']) for row in rows]

init_db()

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

# ── Per-node buffers ──────────────────────────
# WHY: the dashboard now shows multiple map nodes.
# One shared SensorBuffer would mix node A's
# moisture history into node B's rolling average.
# Each node_id gets its own buffer instead; a
# request with no node_id falls back to "default"
# so the original single-node behaviour is unchanged.
node_buffers = {}

def get_buffer(node_id):
    if node_id not in node_buffers:
        node_buffers[node_id] = SensorBuffer(window_size=5)
    return node_buffers[node_id]

# ── Node registry + live state ────────────────
# WHY: GET /api/nodes (used by the map dashboard)
# needs somewhere to read CURRENT state from. This
# is the backend becoming the single source of
# truth: any /predict call updates node_states for
# that node_id; /api/nodes just reads it back out.
#
# NODE_REGISTRY holds static metadata (name, map
# coordinates, whether it's meant to be real
# hardware) for known deployment sites. A /predict
# call with an unknown node_id still works — it
# just won't have a name/lat/lng until you either
# add it here or send them in the request payload
# (see /predict below).
NODE_REGISTRY = {
    'shillong_01':  {'name': 'Shillong, Meghalaya',            'lat': 25.5788, 'lng': 91.8933, 'hardware': True},
    'itanagar_01':  {'name': 'Itanagar, Arunachal Pradesh',    'lat': 27.0844, 'lng': 93.6053, 'hardware': False},
    'kohima_01':    {'name': 'Kohima, Nagaland',               'lat': 25.6751, 'lng': 94.1086, 'hardware': False},
    'aizawl_01':    {'name': 'Aizawl, Mizoram',                'lat': 23.7271, 'lng': 92.7176, 'hardware': False},
    'gangtok_01':   {'name': 'Gangtok, Sikkim',                'lat': 27.3389, 'lng': 88.6065, 'hardware': False},
}

# node_id -> {risk_level, confidence, last_updated, ...}
# Populated live by /predict, read by GET /api/nodes.
node_states = {}

# ── Weather / rainfall layer ──────────────────
# WHY: PS 26001 asks for rainfall data. Real IMD
# API access wasn't available in time (that process
# needs an approved registration), so this uses
# Open-Meteo's free precipitation API instead — real
# meteorological data (blended from national weather
# services), not fabricated, just not IMD-branded
# specifically. Every response says which one it
# actually got: "OPEN_METEO" (real) or "DEMO_DATA"
# (only as a fallback if the API call fails).
#
# TO SWITCH TO REAL IMD DATA LATER: once IMD API
# access is approved, replace the requests.get() call
# below with IMD's endpoint — the rest of the app
# (this endpoint's shape, the dashboard reading it)
# doesn't need to change.
#
# Cached for 30 minutes per node (not indefinitely,
# unlike terrain — rainfall actually changes).
weather_cache = {}  # node_id -> (fetched_at_unix, result)
WEATHER_CACHE_SECONDS = 1800

import random  # only used for the DEMO_DATA fallback path

def get_weather_for_node(node_id, lat, lng, current_risk=None):
    cached = weather_cache.get(node_id)
    if cached and (time.time() - cached[0]) < WEATHER_CACHE_SECONDS:
        result = dict(cached[1])
        result['trend'], result['trend_reason'] = compute_risk_trend(
            result.get('forecast_3d_mm'), current_risk
        )
        return result

    try:
        resp = http_requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lng,
                "daily": "precipitation_sum",
                "past_days": 1, "forecast_days": 3,
                "timezone": "auto"
            },
            timeout=5
        )
        resp.raise_for_status()
        daily = resp.json()["daily"]
        # past_days=1 + forecast_days=3 gives 4 entries:
        # [yesterday, today, tomorrow, day-after-tomorrow]
        rainfall_mm = round(daily["precipitation_sum"][0], 1)          # last 24h, actual
        forecast_3d_mm = round(sum(daily["precipitation_sum"][1:4]), 1)  # next 3 days, forecast

        if rainfall_mm < 15:
            level = 'LOW'
        elif rainfall_mm < 50:
            level = 'MODERATE'
        else:
            level = 'HEAVY'

        result = {
            'node_id': node_id,
            'rainfall_mm_24h': rainfall_mm,
            'forecast_3d_mm': forecast_3d_mm,
            'level': level,
            'source': 'OPEN_METEO',  # real data — see comment block above
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Weather fetch failed for {node_id}, falling back to demo data: {e}")
        # Fallback ONLY on failure — clearly labeled, same as before.
        rng = random.Random(f"{node_id}-{int(time.time() // 600)}")
        rainfall_mm = round(rng.uniform(0, 85), 1)
        forecast_3d_mm = round(rng.uniform(0, 150), 1)
        level = 'LOW' if rainfall_mm < 15 else ('MODERATE' if rainfall_mm < 50 else 'HEAVY')
        result = {
            'node_id': node_id,
            'rainfall_mm_24h': rainfall_mm,
            'forecast_3d_mm': forecast_3d_mm,
            'level': level,
            'source': 'DEMO_DATA',  # fallback — Open-Meteo call failed
            'timestamp': datetime.now().isoformat()
        }

    weather_cache[node_id] = (time.time(), result)
    result = dict(result)
    result['trend'], result['trend_reason'] = compute_risk_trend(forecast_3d_mm, current_risk)
    return result

# ── Weather-linked risk trend ─────────────────
# WHY: PS 26001 asks for "weather-linked risk
# forecasts" on the dashboard. This is a simple,
# TRANSPARENT rule combining real 3-day rainfall
# forecast with the model's current risk classification
# — NOT a trained predictive model (that would need
# real historical landslide-triggering-rainfall data
# to validate, which we don't have). Explainable
# heuristic > black-box guess for a safety tool.
def compute_risk_trend(forecast_3d_mm, current_risk):
    if forecast_3d_mm is None:
        return 'UNKNOWN', 'No forecast data available'

    heavy_forecast = forecast_3d_mm > 100
    moderate_forecast = forecast_3d_mm > 40

    if heavy_forecast and current_risk in ('WARNING', 'DANGER'):
        return 'ESCALATING', f'Heavy rain forecast ({forecast_3d_mm}mm/3d) + current {current_risk} risk'
    if moderate_forecast and current_risk == 'WARNING':
        return 'WATCH', f'Moderate rain forecast ({forecast_3d_mm}mm/3d) while already at WARNING'
    if heavy_forecast:
        return 'WATCH', f'Heavy rain forecast ({forecast_3d_mm}mm/3d) — monitor closely'
    return 'STABLE', f'Forecast rainfall ({forecast_3d_mm}mm/3d) not expected to raise risk'

# ── Historical landslide records ──────────────
# WHY: PS 26001 explicitly asks for historical
# landslide records as a data input. This is a
# SAMPLE of REAL, PUBLISHED, CITED events — not
# fabricated — from a peer-reviewed inventory of
# the Aizawl (Mizoram) region:
#
#   Source: "Frictional timescales and the impact
#   of climate change-driven extreme weather on
#   rainfall-triggered landslides in Mizoram, NE
#   India" (arXiv:2606.23281), which itself draws on
#   GSI's Bhukosh/Bhusanket landslide database
#   (https://bhukosh.gsi.gov.in), published
#   literature, and verified government situation
#   reports.
#
# For the FULL national inventory (GSI has
# documented 91,000+ historical landslides, ~34,000
# field-verified), the real integration point is
# GSI's Bhukosh portal / National Geoscience Data
# Repository — that's a data-sharing request, not
# something to fake. This sample exists to prove
# the pipeline (storage -> API -> map layer) works
# end-to-end, ready to swap in the full feed.
HISTORICAL_LANDSLIDES = [
    # ── Mizoram (Aizawl region) ──
    # Source: Frictional timescales... rainfall-triggered
    # landslides in Mizoram, NE India (arXiv:2606.23281),
    # compiled from GSI Bhukosh/Bhusanket + verified reports.
    {'id': 'LS1', 'date': '2016-09-17', 'location': 'Tlangval, Aizawl',        'lat': 23.73, 'lng': 92.72, 'fatalities': 5,  'type': 'Debris slide',    'trigger': 'Monsoon rainfall', 'state': 'Mizoram', 'source': 'arXiv:2606.23281 (Mizoram landslide inventory, citing GSI Bhukosh)', 'reference_url': 'https://arxiv.org/pdf/2606.23281'},
    {'id': 'LS2', 'date': '2017-09-17', 'location': 'College Veng, Aizawl',    'lat': 23.72, 'lng': 92.72, 'fatalities': 0,  'type': 'Shale failure',   'trigger': 'Monsoon rainfall', 'state': 'Mizoram', 'source': 'arXiv:2606.23281 (Mizoram landslide inventory, citing GSI Bhukosh)', 'reference_url': 'https://arxiv.org/pdf/2606.23281'},
    {'id': 'LS3', 'date': '2019-07-02', 'location': 'Durtlang Leitan, Aizawl', 'lat': 23.77, 'lng': 92.74, 'fatalities': 3,  'type': 'Translational slide', 'trigger': 'Monsoon rainfall', 'state': 'Mizoram', 'source': 'arXiv:2606.23281 (Mizoram landslide inventory, citing GSI Bhukosh)', 'reference_url': 'https://arxiv.org/pdf/2606.23281'},
    {'id': 'LS5', 'date': '2021-06-13', 'location': 'Ngaizel, Kulikawn, Aizawl', 'lat': 23.73, 'lng': 92.73, 'fatalities': 0, 'type': 'Rockslide',      'trigger': 'Monsoon rainfall', 'state': 'Mizoram', 'source': 'arXiv:2606.23281 (Mizoram landslide inventory, citing GSI Bhukosh)', 'reference_url': 'https://arxiv.org/pdf/2606.23281'},
    {'id': 'LS6', 'date': '2024-05-28', 'location': 'Melthum quarry, Aizawl',  'lat': 23.70, 'lng': 92.72, 'fatalities': 15, 'type': 'Quarry collapse', 'trigger': 'Cyclone Remal', 'state': 'Mizoram', 'source': 'arXiv:2606.23281 (Mizoram landslide inventory, citing GSI Bhukosh)', 'reference_url': 'https://arxiv.org/pdf/2606.23281'},

    # ── Meghalaya (East Khasi Hills) ──
    {
        'id': 'MG1', 'date': '2024-06 (exact day within 25 May – 20 Jun 2024 study window)',
        'location': 'Mawlai bypass, Mawkynroh, Shillong', 'lat': 25.6314, 'lng': 91.8840,
        'fatalities': 0, 'type': 'Slope failure (4 injured)', 'trigger': 'Monsoon rainfall',
        'state': 'Meghalaya',
        'source': 'Badavath, Kumar & Sahoo (2025), Landslides 22, 605–614, "Recent rainfall-induced landslides in East Khasi Hills, Meghalaya" (Springer, DOI: 10.1007/s10346-024-02406-6)',
        'reference_url': 'https://link.springer.com/article/10.1007/s10346-024-02406-6'
    },
    {
        'id': 'MG2', 'date': '2022-06-17',
        'location': 'Kenmynsaw / Dangar / Boro Ryngku / Betgora villages, near Mawsynram, East Khasi Hills',
        'lat': 25.29889, 'lng': 91.58139,  # anchored at Mawsynram — exact village coordinates not published in source
        'fatalities': 9, 'type': 'Multiple debris slides / house collapse', 'trigger': 'Record monsoon rainfall',
        'state': 'Meghalaya',
        'source': 'Scroll.in (2022), "Meghalaya\'s record rainfall triggers devastating landslides and flash floods", citing East Khasi Hills Deputy Commissioner Isawanda Laloo. Contains real photos (credited to Conrad Sangma / Twitter).',
        'reference_url': 'https://scroll.in/article/1026977/meghalayas-record-rainfall-triggers-devastating-landslides-and-flash-floods-leaving-34-dead'
    },

    # ── Assam (Dima Hasao) ──
    {
        'id': 'AS1', 'date': '2022-05-16',
        'location': 'New Haflong railway station, Dima Hasao', 'lat': 25.1481, 'lng': 93.0319,
        'fatalities': 3, 'type': 'Channelized debris flow / mudflow (buried a passenger train)',
        'trigger': 'Extreme monsoon rainfall (~540mm district-wide, May 2022)',
        'state': 'Assam',
        'source': 'AGU Landslide Blog (Dave Petley, 2022), citing Roy et al. (2023), Landslides 20, 97–109. Contains real photos of the buried train (credited to Anup Biswas / Nezine). Fatality count is district-wide for the 11–18 May 2022 event, not this site alone.',
        'reference_url': 'https://blogs.agu.org/landslideblog/2022/11/07/dima-hasao-1/'
    },
]
HISTORICAL_SOURCE_CITATION = (
    "Sample of REAL, individually-cited historical landslide events across 3 NER "
    "states (Mizoram, Meghalaya, Assam), drawn from peer-reviewed literature and "
    "verified government reports — see each event's own 'source' field. This is a "
    "small proof-of-pipeline sample, not the full inventory. Full national dataset "
    "(91,000+ records, ~34,000 field-verified) available via GSI's Bhukosh portal: "
    "https://bhukosh.gsi.gov.in"
)

# ── Terrain / slope layer ─────────────────────
# WHY: PS 26001 asks for terrain/slope data. Unlike
# rainfall, this is REAL data — Open-Meteo's free
# elevation API (api.open-meteo.com, no key needed,
# built on Copernicus DEM). We query the node's
# point plus two nearby points (~110m east and
# north) and compute an approximate slope from the
# elevation differences. This is a genuine (if
# coarse) slope estimate, not fabricated — accuracy
# depends on the DEM's resolution, which is a fair
# caveat to state in the pitch.
#
# Results are cached per node_id since terrain is
# static — no need to re-call the API every poll.
terrain_cache = {}

def get_terrain_for_node(node_id, lat, lng):
    if node_id in terrain_cache:
        return terrain_cache[node_id]

    # ~0.001 degrees is roughly 111m at these latitudes —
    # close enough for a coarse local slope estimate.
    offset = 0.001
    lats = f"{lat},{lat + offset},{lat}"
    lngs = f"{lng},{lng},{lng + offset}"

    try:
        resp = http_requests.get(
            "https://api.open-meteo.com/v1/elevation",
            params={"latitude": lats, "longitude": lngs},
            timeout=5
        )
        resp.raise_for_status()
        elevations = resp.json().get("elevation", [])
        if len(elevations) != 3:
            raise ValueError(f"expected 3 elevation points, got {len(elevations)}")

        center_elev, north_elev, east_elev = elevations
        distance_m = 111.0  # approx meters per 0.001 degree at this latitude

        import math
        rise_north = north_elev - center_elev
        rise_east = east_elev - center_elev
        slope_deg = math.degrees(math.atan(
            math.sqrt(rise_north**2 + rise_east**2) / distance_m
        ))

        result = {
            'node_id': node_id,
            'elevation_m': round(center_elev, 1),
            'slope_deg': round(slope_deg, 1),
            'source': 'Open-Meteo Elevation API (Copernicus DEM GLO-90, ~90m resolution)',
            'note': 'Coarse local slope estimate from a 3-point sample, not a full slope map',
            'status': 'ok'
        }
    except Exception as e:
        logger.error(f"Terrain fetch failed for {node_id}: {e}")
        result = {
            'node_id': node_id,
            'elevation_m': None,
            'slope_deg': None,
            'source': 'Open-Meteo Elevation API',
            'note': 'Fetch failed — check network access to api.open-meteo.com',
            'status': 'error'
        }

    terrain_cache[node_id] = result
    return result

# ── Field / citizen reports ───────────────────
# WHY: PS 26001 asks for citizens/field officials
# to be able to report road status and upload
# geo-tagged photos of cracks/slope movement. This
# is a lightweight in-memory version, now backed by
# SQLite (see PERSISTENCE section above) so a server
# restart doesn't lose submitted reports.
field_reports = db_load_all('reports')
MAX_FIELD_REPORTS = 200
VALID_ISSUE_TYPES = {'landslide', 'blocked_road', 'crack_slope_movement', 'flooding', 'other'}
VALID_ROAD_STATUS = {'OPEN', 'CAUTION', 'BLOCKED'}

api_stats = {
    'total_predictions': 0, 'danger_alerts': 0,
    'warning_alerts': 0, 'safe_readings': 0,
    'anomalies_detected': 0, 'errors_caught': 0,
    'start_time': datetime.now().isoformat()
}

# ════════════════════════════════════════════
# EMAIL ALERT SYSTEM
# WHY: adapted from notebooks/day20_email_alerts.py,
# which ran standalone (its own model + test loop,
# not connected to this live /predict flow). This
# wires it into the real backend so a DANGER
# prediction from ANY node actually triggers it.
#
# TEST_MODE=True (default) simulates sending and
# prints to console/log instead of using real
# credentials — safe for demo day. Split into two
# separate flags (not one shared TEST_MODE) so you
# can go live with email while SMS is still pending
# Twilio setup, or vice versa — flip either
# independently once ITS credentials are ready.
# ════════════════════════════════════════════
EMAIL_TEST_MODE = os.environ.get('EMAIL_TEST_MODE', 'true').lower() != 'false'
SMS_TEST_MODE = os.environ.get('SMS_TEST_MODE', 'true').lower() != 'false'

# WHY environment variables here: this file is about to go
# on GitHub for Render deployment. Hardcoded real credentials
# in a repo (even a "private" hackathon one) is a bad habit —
# os.environ.get() reads real values from Render's dashboard
# instead, and falls back to harmless placeholders locally.
EMAIL_CONFIG = {
    'sender_email': os.environ.get('EMAIL_SENDER', 'your_email@gmail.com'),
    'sender_password': os.environ.get('EMAIL_APP_PASSWORD', 'your_app_password_here'),  # Gmail App Password, NOT your real password
    'recipient_emails': [
        e.strip() for e in os.environ.get(
            'EMAIL_RECIPIENTS', 'disaster.officer@example.com,village.head@example.com'
        ).split(',')
    ],
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587
}

class EmailAlertManager:
    """
    Same logic as day20_email_alerts.py's manager, but
    cooldown is now tracked PER NODE (a dict keyed by
    node_id) instead of one global timer — otherwise a
    DANGER alert from Shillong would block Itanagar's
    DANGER alert from firing during the same cooldown
    window.
    """
    def __init__(self, cooldown_minutes=10, test_mode=True):
        self.cooldown_seconds = cooldown_minutes * 60
        self.last_email_time = {}  # node_id -> timestamp
        self.test_mode = test_mode
        self.emails_sent_count = 0
        self.emails_blocked_count = 0

    def should_send_email(self, node_id, risk_code):
        if risk_code < 2:  # Only DANGER triggers email
            return False

        now = time.time()
        last = self.last_email_time.get(node_id)
        if last is None or (now - last) > self.cooldown_seconds:
            self.last_email_time[node_id] = now
            return True

        self.emails_blocked_count += 1
        return False

    def send_alert(self, prediction_data):
        subject = self._build_subject(prediction_data)
        body = self._build_body(prediction_data)

        if self.test_mode:
            logger.info(
                f"[TEST MODE] Would email: {subject}"
            )
            self.emails_sent_count += 1
            return True
        return self._actually_send_email(subject, body)

    def _build_subject(self, data):
        return (f"LANDSLIDE DANGER ALERT - {data['node_id']} - "
                f"{data['confidence']}% Confidence")

    def _build_body(self, data):
        # WHY TRILINGUAL: PS 26001 asks for multilingual
        # notifications. English + Hindi + Assamese now.
        # HONEST CONFIDENCE NOTE: Hindi is a language I'm
        # highly confident in. Assamese is added in good
        # faith with real effort, but with LOWER confidence
        # than Hindi — recommend a native Assamese speaker
        # review these exact lines before relying on them in
        # a real deployment. Khasi/Mizo/others still excluded
        # entirely rather than guessed at — the risk of a
        # wrong safety-critical translation outweighs the
        # value of a token gesture in a language I can't
        # verify at all.
        return f"""
LANDSENSE ML — AUTOMATED DANGER ALERT
लैंडसेंस एमएल — स्वचालित खतरे की चेतावनी
লেণ্ডছেন্স এমএল — স্বয়ংক্ৰিয় বিপদ সতৰ্কবাণী
========================================
Node / स्थान / স্থান         : {data['node_id']}
Risk Level / जोखिम स्तर / বিপদৰ স্তৰ : {data['risk_level']}
Confidence / विश्वास / বিশ্বাস      : {data['confidence']}%
Anomaly Flag          : {data['is_anomaly']}
Detected at / समय / সময়       : {data['timestamp']}

Sensor Readings:
  Soil Moisture     : {data.get('moisture', 'N/A')}
  Acceleration Mag  : {data['calculated_features']['accel_magnitude']}
  Moisture Change   : {data['calculated_features']['moisture_change']}

RECOMMENDATION / सिफारिश / পৰামৰ্শ:
Immediate field verification advised.
कृपया तुरंत स्थल का निरीक्षण करें।
অনুগ্ৰহ কৰি তৎক্ষণাৎ স্থান পৰিদৰ্শন কৰক।
----------------------------------------
This is an automated message from
LandSense ML Early Warning System.
========================================
"""

    def _actually_send_email(self, subject, body):
        try:
            msg = MIMEMultipart()
            msg['From'] = EMAIL_CONFIG['sender_email']
            msg['To'] = ', '.join(EMAIL_CONFIG['recipient_emails'])
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(
                EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port']
            )
            server.starttls()
            server.login(
                EMAIL_CONFIG['sender_email'],
                EMAIL_CONFIG['sender_password']
            )
            server.send_message(msg)
            server.quit()
            self.emails_sent_count += 1
            logger.info("Email sent successfully")
            return True
        except Exception as e:
            logger.error(f"Email failed: {e}")
            return False

email_manager = EmailAlertManager(cooldown_minutes=10, test_mode=EMAIL_TEST_MODE)

# ════════════════════════════════════════════
# SMS ALERT SYSTEM
# WHY: PS 26001 explicitly asks for SMS/app-based
# alerts, not just email — email requires internet
# and an inbox check; SMS reaches a basic phone
# with no data connection, which matters for the
# remote NER villages this is actually for.
#
# Same TEST_MODE-first pattern as email: logs what
# would be sent, no Twilio account needed for the
# demo. Flip TEST_MODE=False and fill in
# SMS_CONFIG once you have a Twilio trial number
# (free tier: twilio.com/try-twilio, takes ~5 min
# to get a number + auth token).
# ════════════════════════════════════════════
SMS_CONFIG = {
    'account_sid': os.environ.get('TWILIO_ACCOUNT_SID', 'your_twilio_account_sid_here'),
    'auth_token': os.environ.get('TWILIO_AUTH_TOKEN', 'your_twilio_auth_token_here'),
    'from_number': os.environ.get('TWILIO_FROM_NUMBER', '+1XXXXXXXXXX'),
    'recipient_numbers': [
        n.strip() for n in os.environ.get(
            'SMS_RECIPIENTS', '+91XXXXXXXXXX,+91XXXXXXXXXX'
        ).split(',')
    ]
}

class SmsAlertManager:
    """
    Mirrors EmailAlertManager's per-node cooldown so a
    DANGER alert from one node doesn't block SMS to
    another. Cooldown is tracked SEPARATELY from email's
    (its own dict) — the two channels don't share state,
    so if you ever silence one channel it doesn't
    accidentally silence the other.
    """
    def __init__(self, cooldown_minutes=10, test_mode=True):
        self.cooldown_seconds = cooldown_minutes * 60
        self.last_sms_time = {}  # node_id -> timestamp
        self.test_mode = test_mode
        self.sms_sent_count = 0
        self.sms_blocked_count = 0
        self._client = None

    def should_send_sms(self, node_id, risk_code):
        if risk_code < 2:  # Only DANGER triggers SMS
            return False

        now = time.time()
        last = self.last_sms_time.get(node_id)
        if last is None or (now - last) > self.cooldown_seconds:
            self.last_sms_time[node_id] = now
            return True

        self.sms_blocked_count += 1
        return False

    def send_alert(self, prediction_data):
        message = self._build_message(prediction_data)

        if self.test_mode:
            logger.info(f"[TEST MODE] Would SMS: {message}")
            self.sms_sent_count += 1
            return True
        return self._actually_send_sms(message)

    def _build_message(self, data):
        # Real SMS is billed/limited per ~160 chars — adding
        # Hindi + Assamese means this spans more segments,
        # an acceptable tradeoff for reaching more people.
        # Same confidence caveat as the email version: Hindi
        # is high-confidence, Assamese is good-faith but
        # recommend native-speaker review before real use.
        return (
            f"LandSense ALERT: DANGER at {data['node_id']} "
            f"({data['confidence']}% confidence). "
            f"Immediate field verification advised. "
            f"Time: {data['timestamp']}\n"
            f"चेतावनी: {data['node_id']} में खतरा ({data['confidence']}% विश्वास)। "
            f"कृपया तुरंत जांच करें।\n"
            f"সতৰ্কবাণী: {data['node_id']}ত বিপদ ({data['confidence']}% বিশ্বাস)। "
            f"অনুগ্ৰহ কৰি তৎক্ষণাৎ পৰীক্ষা কৰক।"
        )

    def _actually_send_sms(self, message):
        try:
            # Imported here, not at the top of the file, so the
            # whole API doesn't fail to start if the `twilio`
            # package isn't installed and you're still in TEST_MODE.
            from twilio.rest import Client
            if self._client is None:
                self._client = Client(SMS_CONFIG['account_sid'], SMS_CONFIG['auth_token'])

            for number in SMS_CONFIG['recipient_numbers']:
                self._client.messages.create(
                    body=message, from_=SMS_CONFIG['from_number'], to=number
                )
            self.sms_sent_count += 1
            logger.info("SMS sent successfully")
            return True
        except Exception as e:
            logger.error(f"SMS failed: {e}")
            return False

sms_manager = SmsAlertManager(cooldown_minutes=10, test_mode=SMS_TEST_MODE)

# Alert history — GET /api/alerts reads this. Now
# backed by SQLite (see PERSISTENCE section above) so
# a server restart doesn't lose the alert log.
alert_history = db_load_all('alerts')
MAX_ALERT_HISTORY = 100

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

        # Process through this node's own buffer
        node_id = str(data.get('node_id', 'default'))
        live_features = get_buffer(node_id).process_reading(
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
        email_sent = False
        email_blocked = False
        sms_sent = False
        sms_blocked = False
        if risk_pred == 2:
            api_stats['danger_alerts'] += 1
            logger.info(
                f"DANGER detected on {node_id} - "
                f"confidence: {risk_proba[2]*100:.1f}%"
            )
            timestamp = datetime.now().isoformat()
            confidence = round(float(risk_proba[2])*100, 1)
            alert_payload = {
                'node_id': node_id,
                'risk_level': 'DANGER',
                'confidence': confidence,
                'is_anomaly': is_anomaly,
                'moisture': live_features['moisture'],
                'timestamp': timestamp,
                'calculated_features': {
                    'accel_magnitude': round(live_features['accel_magnitude'], 1),
                    'moisture_change': round(live_features['moisture_change'], 1)
                }
            }

            if email_manager.should_send_email(node_id, int(risk_pred)):
                email_sent = email_manager.send_alert(alert_payload)
            else:
                email_blocked = True
                logger.info(f"Email skipped for {node_id} (cooldown active)")

            # SMS cooldown is tracked independently of email's —
            # see SmsAlertManager's docstring for why.
            if sms_manager.should_send_sms(node_id, int(risk_pred)):
                sms_sent = sms_manager.send_alert(alert_payload)
            else:
                sms_blocked = True
                logger.info(f"SMS skipped for {node_id} (cooldown active)")

            alert_entry = {
                'node_id': node_id,
                'risk_level': 'DANGER',
                'confidence': confidence,
                'timestamp': timestamp,
                'email_sent': email_sent,
                'email_blocked_by_cooldown': email_blocked,
                'sms_sent': sms_sent,
                'sms_blocked_by_cooldown': sms_blocked
            }
            alert_history.append(alert_entry)
            db_save('alerts', alert_entry)
            del alert_history[:-MAX_ALERT_HISTORY]  # keep only the most recent
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

        # Update this node's live state so GET /api/nodes
        # (used by the map dashboard) can read it back out.
        meta = NODE_REGISTRY.get(node_id, {})
        node_states[node_id] = {
            'node_id': node_id,
            'name': meta.get('name', node_id),
            'lat': meta.get('lat'),
            'lng': meta.get('lng'),
            'risk_level': labels[risk_pred],
            'confidence': response['confidence'],
            'source': 'HARDWARE' if meta.get('hardware') else 'SIMULATED',
            'last_updated': response['timestamp']
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

@app.route('/api/alerts')
def get_alerts():
    """Most recent DANGER alerts, newest first."""
    return jsonify({
        'count': len(alert_history),
        'alerts': list(reversed(alert_history))
    })

@app.route('/api/nodes')
def get_nodes():
    """
    Current state of every known node, for the map
    dashboard. Nodes with no lat/lng (e.g. the
    'default' fallback node from requests sent
    without a node_id) are skipped — the dashboard
    has nowhere to place them on the map.
    """
    nodes = [
        state for state in node_states.values()
        if state.get('lat') is not None and state.get('lng') is not None
    ]
    return jsonify(nodes)

@app.route('/api/reports', methods=['POST'])
def submit_report():
    """
    A citizen/field officer submits a road-status or
    hazard report. Expects JSON:
      {
        "location_name": "NH-40 near Shillong",  (required)
        "lat": 25.57, "lng": 91.89,               (optional)
        "issue_type": "blocked_road",             (required, see VALID_ISSUE_TYPES)
        "road_status": "BLOCKED",                 (optional, see VALID_ROAD_STATUS)
        "description": "Landslide debris blocking both lanes",
        "photo_base64": "data:image/jpeg;base64,...",  (optional)
        "reporter_name": "Field Officer Rai"      (optional)
      }
    """
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({'success': False, 'error': 'No JSON body provided'}), 400

        location_name = str(data.get('location_name', '')).strip()
        issue_type = str(data.get('issue_type', '')).strip().lower()

        if not location_name:
            return jsonify({'success': False, 'error': 'location_name is required'}), 400
        if issue_type not in VALID_ISSUE_TYPES:
            return jsonify({
                'success': False,
                'error': f'issue_type must be one of {sorted(VALID_ISSUE_TYPES)}'
            }), 400

        road_status = str(data.get('road_status', '')).strip().upper()
        if road_status and road_status not in VALID_ROAD_STATUS:
            return jsonify({
                'success': False,
                'error': f'road_status must be one of {sorted(VALID_ROAD_STATUS)}'
            }), 400

        report = {
            'id': len(field_reports) + 1,
            'location_name': location_name,
            'lat': data.get('lat'),
            'lng': data.get('lng'),
            'issue_type': issue_type,
            'road_status': road_status or None,
            'description': str(data.get('description', ''))[:1000],
            'has_photo': bool(data.get('photo_base64')),
            'photo_base64': data.get('photo_base64'),  # now persisted to SQLite too — see PERSISTENCE section
            'reporter_name': str(data.get('reporter_name', 'Anonymous'))[:100],
            'timestamp': datetime.now().isoformat()
        }
        field_reports.append(report)
        db_save('reports', report)
        del field_reports[:-MAX_FIELD_REPORTS]

        logger.info(f"Field report received: {issue_type} at {location_name}")

        response_report = {k: v for k, v in report.items() if k != 'photo_base64'}
        return jsonify({'success': True, 'report': response_report}), 201

    except Exception as e:
        logger.error(f"Report submission error: {str(e)}")
        return jsonify({'success': False, 'error': f'Internal error: {str(e)}'}), 500

@app.route('/api/reports', methods=['GET'])
def get_reports():
    """Most recent field reports, newest first. Excludes
    the (potentially large) base64 photo data by default —
    pass ?include_photos=1 to include it."""
    include_photos = request.args.get('include_photos') == '1'
    reports = list(reversed(field_reports))
    if not include_photos:
        reports = [{k: v for k, v in r.items() if k != 'photo_base64'} for r in reports]
    return jsonify({'count': len(reports), 'reports': reports})

@app.route('/api/weather')
def get_weather():
    """
    Rainfall + 3-day forecast + weather-linked risk trend
    for every known node. Rainfall/forecast are real
    (Open-Meteo) with DEMO_DATA as a labeled fallback only
    if that call fails. Trend is a transparent rule
    combining the forecast with the model's current risk —
    see compute_risk_trend()'s docstring for why it's a
    rule, not a trained predictive model.
    """
    weather = [
        get_weather_for_node(
            node_id, meta['lat'], meta['lng'],
            current_risk=node_states.get(node_id, {}).get('risk_level')
        )
        for node_id, meta in NODE_REGISTRY.items()
    ]
    return jsonify({'nodes': weather})  # each node's own 'source' field says OPEN_METEO (real) or DEMO_DATA (fallback)

@app.route('/api/historical-landslides')
def get_historical_landslides():
    """
    Real, cited sample of historical landslide records
    (see HISTORICAL_SOURCE_CITATION above for the source).
    NOT the full GSI inventory — a documented sample proving
    the data pipeline, ready to be swapped for the real feed.
    """
    return jsonify({
        'count': len(HISTORICAL_LANDSLIDES),
        'events': HISTORICAL_LANDSLIDES,
        'citation': HISTORICAL_SOURCE_CITATION
    })

@app.route('/api/terrain')
def get_terrain():
    """
    Real elevation + coarse slope estimate per node, via
    Open-Meteo's free elevation API. Cached after first
    fetch since terrain doesn't change between polls.
    """
    terrain = [
        get_terrain_for_node(node_id, meta['lat'], meta['lng'])
        for node_id, meta in NODE_REGISTRY.items()
    ]
    return jsonify({'nodes': terrain})

# ── Emergency response prioritization ─────────
# WHY: PS 26001 explicitly asks for "emergency
# response prioritisation" on the dashboard — nothing
# ranks which node or report needs attention first.
# This is a simple, TRANSPARENT weighted score (not
# ML) — deliberately explainable: every item in the
# queue shows exactly why it's ranked where it is,
# which matters more for a disaster-response tool
# than a black-box score would.
RISK_SCORE = {'DANGER': 100, 'WARNING': 40, 'SAFE': 0}
ROAD_SCORE = {'BLOCKED': 90, 'CAUTION': 45, 'OPEN': 0}
ISSUE_SCORE = {'landslide': 50, 'crack_slope_movement': 35, 'blocked_road': 20, 'flooding': 20, 'other': 0}

def compute_priority_queue():
    items = []

    for node_id, state in node_states.items():
        if state.get('lat') is None:
            continue  # can't prioritize something with no location
        risk = state.get('risk_level', 'SAFE')
        confidence = state.get('confidence') or 0
        score = RISK_SCORE.get(risk, 0) + (confidence * 0.2)
        if score <= 0:
            continue  # SAFE nodes aren't an emergency-response item
        items.append({
            'type': 'sensor_node',
            'id': node_id,
            'location': state.get('name', node_id),
            'lat': state.get('lat'), 'lng': state.get('lng'),
            'priority_score': round(score, 1),
            'reasons': [f"{risk} risk ({confidence}% model confidence)"]
        })

    # Only recent reports (last 50) so an old resolved report from
    # days ago doesn't permanently clog the top of the queue.
    for report in field_reports[-50:]:
        score = 0
        reasons = []
        road_status = report.get('road_status')
        if road_status:
            road_pts = ROAD_SCORE.get(road_status, 0)
            if road_pts > 0:
                score += road_pts
                reasons.append(f"Road status: {road_status}")
        issue_pts = ISSUE_SCORE.get(report.get('issue_type'), 0)
        if issue_pts > 0:
            score += issue_pts
            reasons.append(f"Citizen report: {report.get('issue_type', '').replace('_', ' ')}")
        if score <= 0:
            continue  # e.g. an 'other' report with OPEN road status isn't urgent
        items.append({
            'type': 'field_report',
            'id': report.get('id'),
            'location': report.get('location_name'),
            'lat': report.get('lat'), 'lng': report.get('lng'),
            'priority_score': round(score, 1),
            'reasons': reasons
        })

    items.sort(key=lambda x: x['priority_score'], reverse=True)
    return items

@app.route('/api/priority')
def get_priority():
    """Ranked emergency-response queue — sensor nodes and
    field reports combined, highest priority first, each
    with a plain-language reason for its ranking."""
    return jsonify({'queue': compute_priority_queue()})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'true').lower() == 'true'
    logger.info("Starting Flask server...")
    print("\n🚀 Starting Hardened Flask server...")
    print(f"   Local access:   http://127.0.0.1:{port}")
    print("   Press CTRL+C to stop")
    print("="*55)
    app.run(host='0.0.0.0', port=port, debug=debug_mode)