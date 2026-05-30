# ============================================
# DAY 5 — Visualization Part 2
# Date: 29 May 2026
# Author: Akshay
# Goal: Deep dive analysis of sensor patterns
#       Focus on danger events and timing
# ============================================
# DAY 5 OBSERVATIONS:
# 1. Peak danger hour: 16:00 (4 PM)
# 2. Danger is sudden spike in simulated data
#    Real data will show gradual buildup
# 3. Rolling average effectively removes
#    noise while keeping danger spikes ✅
# 4. Magnitude vs Moisture best feature pair
#    for separating risk levels
# 5. Smoothed vs Smoothed loses separation
#    → Keep raw features alongside smoothed
# 6. Both sensors spike together at danger
#    → Confirms 0.89 correlation from Day 4
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── Load data ─────────────────────────────────
df = pd.read_csv('data/processed/sensor_features.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Create risk labels
def get_risk(row):
    if row['moisture'] > 3500:
        return 2
    elif row['moisture'] > 2800:
        return 1
    else:
        return 0

df['risk'] = df.apply(get_risk, axis=1)

# Get danger and warning indices
danger_idx = df[df['risk'] == 2].index.tolist()
warning_idx = df[df['risk'] == 1].index.tolist()
safe_idx = df[df['risk'] == 0].index.tolist()

print("="*50)
print("   DAY 5 — VISUALIZATION PART 2")
print("="*50)
print(f"Danger events  : {len(danger_idx)}")
print(f"Warning events : {len(warning_idx)}")
print(f"Safe readings  : {len(safe_idx)}")

# ════════════════════════════════════════════
# GRAPH 1 — Danger Event Deep Dive
# Zoom into first 3 danger events
# Show 10 readings before and after
# ════════════════════════════════════════════
print("\nCreating Graph 1: Danger Event Deep Dive...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    'Danger Event Deep Dive\n'
    '10 readings before → danger → 10 after',
    fontsize=13, fontweight='bold'
)

# Show first 3 danger events
for idx, (ax, d_idx) in enumerate(
    zip(axes, danger_idx[:3])
):
    # Get window: 10 before to 10 after
    start = max(0, d_idx - 10)
    end = min(len(df), d_idx + 10)
    window = df.iloc[start:end].copy()
    window = window.reset_index(drop=True)

    # Find danger position in window
    danger_pos = d_idx - start

    # Plot moisture
    ax.plot(
        window['moisture'],
        color='green',
        linewidth=2,
        label='Moisture',
        marker='o',
        markersize=4
    )

    # Plot accel_magnitude on twin axis
    ax2 = ax.twinx()
    ax2.plot(
        window['accel_magnitude'],
        color='steelblue',
        linewidth=2,
        label='Magnitude',
        linestyle='--',
        marker='s',
        markersize=4
    )

    # Mark danger point
    ax.axvline(
        x=danger_pos,
        color='red',
        linewidth=2.5,
        linestyle='-',
        label='Danger!'
    )

    # Shade danger zone
    ax.axvspan(
        danger_pos - 1,
        danger_pos + 1,
        alpha=0.2,
        color='red'
    )

    ax.set_title(f'Danger Event {idx+1}\n'
                 f'(Row {d_idx})')
    ax.set_xlabel('Reading (relative)')
    ax.set_ylabel('Moisture', color='green')
    ax2.set_ylabel('Magnitude', color='steelblue')
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)
    ax.axhline(
        y=3500, color='red',
        linestyle=':', alpha=0.5
    )

plt.tight_layout()
plt.savefig('plots/day05_danger_deepdive.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("✅ Graph 1 saved")

# ════════════════════════════════════════════
# GRAPH 2 — Rolling Average Effectiveness
# Raw vs Smoothed comparison
# ════════════════════════════════════════════
print("Creating Graph 2: Rolling Average...")

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
fig.suptitle(
    'Rolling Average Effectiveness\n'
    'Raw sensor data vs Smoothed average',
    fontsize=13, fontweight='bold'
)

# Show first 200 readings for clarity
sample = df.iloc[:200]

# Raw moisture
axes[0].plot(
    sample['moisture'],
    color='lightgreen',
    alpha=0.8,
    linewidth=0.8,
    label='Raw moisture (noisy)'
)
axes[0].plot(
    sample['moisture_avg5'],
    color='darkgreen',
    linewidth=2.5,
    label='5-reading average (smooth)'
)
axes[0].axhline(
    y=3500, color='red',
    linestyle='--', label='Danger threshold'
)
axes[0].axhline(
    y=2800, color='orange',
    linestyle='--', label='Warning threshold'
)
axes[0].set_title('Moisture: Raw vs Smoothed')
axes[0].set_ylabel('Moisture Value')
axes[0].legend()
axes[0].set_facecolor('#f8f8f8')

# Raw magnitude vs smoothed
axes[1].plot(
    sample['accel_magnitude'],
    color='lightblue',
    alpha=0.8,
    linewidth=0.8,
    label='Raw magnitude (noisy)'
)
axes[1].plot(
    sample['magnitude_avg5'],
    color='steelblue',
    linewidth=2.5,
    label='5-reading average (smooth)'
)
axes[1].set_title(
    'Acceleration Magnitude: Raw vs Smoothed'
)
axes[1].set_ylabel('Magnitude Value')
axes[1].set_xlabel('Reading Number (first 200)')
axes[1].legend()
axes[1].set_facecolor('#f8f8f8')

plt.tight_layout()
plt.savefig('plots/day05_rolling_average.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("✅ Graph 2 saved")

# ════════════════════════════════════════════
# GRAPH 3 — Hour of Day Analysis
# When does danger happen most?
# ════════════════════════════════════════════
print("Creating Graph 3: Hour Analysis...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    'Time-based Analysis\n'
    'When do danger events occur?',
    fontsize=13, fontweight='bold'
)

# Average risk by hour
hourly_risk = df.groupby('hour')['risk'].mean()
colors_hour = [
    'red' if r > 0.1
    else 'orange' if r > 0.05
    else 'green'
    for r in hourly_risk.values
]

axes[0].bar(
    hourly_risk.index,
    hourly_risk.values,
    color=colors_hour,
    edgecolor='black',
    alpha=0.8
)
axes[0].set_title('Average Risk Level by Hour')
axes[0].set_xlabel('Hour of Day')
axes[0].set_ylabel('Average Risk (0=safe 2=danger)')
axes[0].set_xticks(range(0, 24, 2))
axes[0].axhline(
    y=0.05, color='orange',
    linestyle='--', label='Warning level'
)
axes[0].legend()

# Count of danger events by hour
danger_by_hour = df[df['risk'] == 2].groupby(
    'hour'
).size()
all_hours = pd.Series(
    0, index=range(24)
)
all_hours.update(danger_by_hour)

axes[1].bar(
    all_hours.index,
    all_hours.values,
    color='red',
    edgecolor='black',
    alpha=0.7
)
axes[1].set_title('Danger Event Count by Hour')
axes[1].set_xlabel('Hour of Day')
axes[1].set_ylabel('Number of Danger Events')
axes[1].set_xticks(range(0, 24, 2))

plt.tight_layout()
plt.savefig('plots/day05_hour_analysis.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("✅ Graph 3 saved")

# ════════════════════════════════════════════
# GRAPH 4 — Feature Pair Relationships
# Which feature pairs best separate risks?
# ════════════════════════════════════════════
print("Creating Graph 4: Feature Pairs...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    'Feature Pair Relationships\n'
    'Colored by Risk Level',
    fontsize=13, fontweight='bold'
)

color_map = {0: 'green', 1: 'orange', 2: 'red'}
label_map = {0: 'Safe', 1: 'Warning', 2: 'Danger'}

pairs = [
    ('accel_magnitude', 'moisture',
     axes[0,0], 'Magnitude vs Moisture'),
    ('moisture_change', 'moisture',
     axes[0,1], 'Moisture Change vs Moisture'),
    ('accel_magnitude', 'moisture_change',
     axes[1,0], 'Magnitude vs Moisture Change'),
    ('moisture_avg5', 'magnitude_avg5',
     axes[1,1], 'Smoothed Moisture vs Magnitude'),
]

for feat1, feat2, ax, title in pairs:
    for risk_level in [0, 1, 2]:
        mask = df['risk'] == risk_level
        ax.scatter(
            df[mask][feat1],
            df[mask][feat2],
            c=color_map[risk_level],
            label=label_map[risk_level],
            alpha=0.5,
            s=10
        )
    ax.set_title(title)
    ax.set_xlabel(feat1)
    ax.set_ylabel(feat2)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/day05_feature_pairs.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("✅ Graph 4 saved")

# ════════════════════════════════════════════
# GRAPH 5 — Danger Buildup Pattern
# Average sensor values leading to danger
# ════════════════════════════════════════════
print("Creating Graph 5: Danger Buildup...")

# Collect 10 readings before each danger event
window_size = 10
moisture_windows = []
magnitude_windows = []

for d_idx in danger_idx:
    if d_idx >= window_size:
        m_window = df.iloc[
            d_idx-window_size:d_idx
        ]['moisture'].values
        mag_window = df.iloc[
            d_idx-window_size:d_idx
        ]['accel_magnitude'].values

        moisture_windows.append(m_window)
        magnitude_windows.append(mag_window)

if moisture_windows:
    avg_moisture = np.mean(moisture_windows, axis=0)
    avg_magnitude = np.mean(
        magnitude_windows, axis=0
    )

    fig, axes = plt.subplots(
        2, 1, figsize=(12, 8)
    )
    fig.suptitle(
        'Average Sensor Pattern Leading to Danger\n'
        '(Average of all danger event buildups)',
        fontsize=13, fontweight='bold'
    )

    x = range(1, window_size + 1)

    axes[0].plot(
        x, avg_moisture,
        color='green',
        linewidth=2.5,
        marker='o',
        markersize=6,
        label='Avg moisture before danger'
    )
    axes[0].fill_between(
        x, avg_moisture,
        alpha=0.2, color='green'
    )
    axes[0].axhline(
        y=3500, color='red',
        linestyle='--', label='Danger threshold'
    )
    axes[0].axhline(
        y=2800, color='orange',
        linestyle='--', label='Warning threshold'
    )
    axes[0].set_title(
        'Average Moisture — 10 Readings Before Danger'
    )
    axes[0].set_xlabel(
        'Reading (1=10 before, 10=just before danger)'
    )
    axes[0].set_ylabel('Moisture Value')
    axes[0].legend()
    axes[0].set_facecolor('#f8f8f8')

    axes[1].plot(
        x, avg_magnitude,
        color='steelblue',
        linewidth=2.5,
        marker='s',
        markersize=6,
        label='Avg magnitude before danger'
    )
    axes[1].fill_between(
        x, avg_magnitude,
        alpha=0.2, color='steelblue'
    )
    axes[1].axhline(
        y=4000, color='red',
        linestyle='--', label='Danger threshold'
    )
    axes[1].set_title(
        'Average Magnitude — 10 Readings Before Danger'
    )
    axes[1].set_xlabel(
        'Reading (1=10 before, 10=just before danger)'
    )
    axes[1].set_ylabel('Magnitude Value')
    axes[1].legend()
    axes[1].set_facecolor('#f8f8f8')

    plt.tight_layout()
    plt.savefig('plots/day05_danger_buildup.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Graph 5 saved")

# ── Final Summary ─────────────────────────────
print("\n" + "="*50)
print("   DAY 5 COMPLETE — VISUALIZATION 2")
print("="*50)
print("\n5 graphs created:")
print("  1. plots/day05_danger_deepdive.png")
print("  2. plots/day05_rolling_average.png")
print("  3. plots/day05_hour_analysis.png")
print("  4. plots/day05_feature_pairs.png")
print("  5. plots/day05_danger_buildup.png")
print("\n🔍 KEY OBSERVATIONS TO NOTE:")
peak_hour = hourly_risk.idxmax()
print(f"  Peak danger hour: {peak_hour}:00")
print(f"  Rolling average smooths noise ✅")
print(f"  Danger buildup visible in graph 5 ✅")
print(f"  Feature pairs clearly separated ✅")
print("\n✅ Ready for Day 6!")
print("="*50)