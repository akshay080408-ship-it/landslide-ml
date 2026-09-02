"""
LandSense ML — Node Simulator
==============================
WHY THIS EXISTS: the new map dashboard (the one that
polls GET /api/nodes) only DISPLAYS state — it doesn't
generate any data itself. Something has to POST
readings to /predict for each node, the way a real
ESP32 would. Until hardware's back, this script plays
that role for all 5 demo nodes.

Run this in a THIRD terminal window, alongside:
  Terminal 1: python api/app.py
  Terminal 2: (open dashboard/map.html in a browser)
  Terminal 3: python api/simulate_nodes.py   <-- this file

Once your real ESP32 is back and running the firmware,
you can stop simulating that one node here (just remove
it from NODES below) — the firmware will feed it directly
and this script keeps simulating the rest.
"""

import time
import requests

API_URL = "https://YOUR-RENDER-URL.onrender.com/predict"  # replace with your real Render URL

# Must match NODE_REGISTRY in api/app.py
NODES = ["shillong_01", "itanagar_01", "kohima_01", "aizawl_01", "gangtok_01"]

# Cycles through Safe -> Warning -> Danger -> back down,
# same shape the ESP32 firmware sends.
DEMO_READINGS = [
    {"accel_x": 2048, "accel_y": 2050, "accel_z": 2048, "moisture": 1800, "vibration": 0},
    {"accel_x": 2060, "accel_y": 2055, "accel_z": 2048, "moisture": 1950, "vibration": 0},
    {"accel_x": 2200, "accel_y": 2250, "accel_z": 2048, "moisture": 2900, "vibration": 0},
    {"accel_x": 2300, "accel_y": 2400, "accel_z": 2048, "moisture": 3100, "vibration": 0},
    {"accel_x": 2700, "accel_y": 2800, "accel_z": 2048, "moisture": 3900, "vibration": 1},
]

POLL_INTERVAL_SECONDS = 5

def main():
    # Offset each node's starting index so they don't all
    # show the same risk level in lockstep.
    indices = {node_id: i * 2 for i, node_id in enumerate(NODES)}

    print(f"Simulating {len(NODES)} nodes -> {API_URL}")
    print("Press Ctrl+C to stop.\n")

    while True:
        for node_id in NODES:
            reading = dict(DEMO_READINGS[indices[node_id] % len(DEMO_READINGS)])
            reading["node_id"] = node_id
            indices[node_id] += 1

            try:
                resp = requests.post(API_URL, json=reading, timeout=5)
                data = resp.json()
                if data.get("success"):
                    print(f"  {node_id:15s} -> {data['risk_level']:8s} ({data['confidence']}%)")
                else:
                    print(f"  {node_id:15s} -> error: {data.get('error')}")
            except requests.exceptions.RequestException as e:
                print(f"  {node_id:15s} -> could not reach API: {e}")

            # Small stagger between nodes so the shared
            # rate limiter (1 req / 0.3s) doesn't reject any.
            time.sleep(0.4)

        time.sleep(POLL_INTERVAL_SECONDS)

if __name__ == "__main__":
    main()