import requests
import time

readings = [
    {'accel_x': 2050, 'accel_y': 2048, 'accel_z': 2048,
     'moisture': 1800, 'vibration': 0},
    {'accel_x': 2100, 'accel_y': 2080, 'accel_z': 2048,
     'moisture': 2200, 'vibration': 0},
    {'accel_x': 2700, 'accel_y': 2800, 'accel_z': 2048,
     'moisture': 3900, 'vibration': 1},
]

for i, r in enumerate(readings, 1):
    resp = requests.post(
        'http://127.0.0.1:5000/predict', json=r
    )
    result = resp.json()
    print(f"Reading {i}: {result['risk_level']} "
          f"({result['confidence']}%) - "
          f"moisture_change: "
          f"{result['calculated_features']['moisture_change']}")
    time.sleep(0.5)