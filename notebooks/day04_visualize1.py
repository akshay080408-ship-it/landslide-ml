# ============================================
# DAY 4 — Visualization Part 1
# Date: 28 May 2026
# Author: Akshay
# Goal: Convert sensor numbers into
#       meaningful visual graphs
# ============================================

# DAY 4 OBSERVATIONS:
# 1. Most correlated with risk: accel_magnitude (0.89)
# 2. Safe and danger PERFECTLY separated ✅
# 3. accel_z has near zero correlation (-0.03)
#    → Can remove from ML model
# 4. 3 clear clusters in scatter plot
#    → ML accuracy expected: 95%+
# 5. moisture_change correlation: 0.41
#    → Rate of change is valuable feature
# ============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load feature engineered data ─────────────
df = pd.read_csv('data/processed/sensor_features.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

print("="*50)
print("   DAY 4 — VISUALIZATION PART 1")
print("="*50)
print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

# ── Create risk labels for coloring ──────────
# 0=safe 1=warning 2=danger
def get_risk(row):
    if row['moisture'] > 3500:
        return 2
    elif row['moisture'] > 2800:
        return 1
    else:
        return 0

df['risk'] = df.apply(get_risk, axis=1)

# Color for each risk level
color_map = {0: 'green', 1: 'orange', 2: 'red'}
colors = df['risk'].map(color_map)

print("✅ Risk labels created for coloring")

# ════════════════════════════════════════════
# GRAPH 1 — Sensor Overview (4 subplots)
# ════════════════════════════════════════════
print("\nCreating Graph 1: Sensor Overview...")

fig, axes = plt.subplots(4, 1, figsize=(14, 10))
fig.suptitle(
    'LandSense — Sensor Overview\n'
    'All sensors over 2000 readings',
    fontsize=14, fontweight='bold'
)

# ── Subplot 1: Acceleration Magnitude ────────
axes[0].plot(
    df['accel_magnitude'],
    color='steelblue',
    alpha=0.7,
    linewidth=0.8,
    label='Accel Magnitude'
)
axes[0].axhline(
    y=4000,
    color='red',
    linestyle='--',
    linewidth=1.5,
    label='Danger threshold'
)
axes[0].set_title('Acceleration Magnitude')
axes[0].set_ylabel('Magnitude Value')
axes[0].legend(loc='upper right')
axes[0].set_facecolor('#f8f8f8')

# ── Subplot 2: Soil Moisture ──────────────────
axes[1].plot(
    df['moisture'],
    color='green',
    alpha=0.5,
    linewidth=0.8,
    label='Raw moisture'
)
axes[1].plot(
    df['moisture_avg5'],
    color='darkgreen',
    linewidth=1.5,
    label='5-reading average'
)
axes[1].axhline(
    y=3500,
    color='red',
    linestyle='--',
    linewidth=1.5,
    label='Danger (3500)'
)
axes[1].axhline(
    y=2800,
    color='orange',
    linestyle='--',
    linewidth=1.0,
    label='Warning (2800)'
)
axes[1].set_title('Soil Moisture')
axes[1].set_ylabel('Moisture Value')
axes[1].legend(loc='upper right')
axes[1].set_facecolor('#f8f8f8')

# ── Subplot 3: Vibration Events ───────────────
axes[2].fill_between(
    range(len(df)),
    df['vibration'],
    color='red',
    alpha=0.6,
    label='Vibration detected'
)
axes[2].set_title('Vibration Events')
axes[2].set_ylabel('0=No  1=Yes')
axes[2].set_ylim(-0.1, 1.5)
axes[2].legend(loc='upper right')
axes[2].set_facecolor('#f8f8f8')

# ── Subplot 4: Moisture Rate of Change ────────
axes[3].plot(
    df['moisture_change'],
    color='orange',
    alpha=0.7,
    linewidth=0.8,
    label='Moisture change'
)
axes[3].axhline(
    y=0,
    color='black',
    linewidth=0.5
)
axes[3].axhline(
    y=500,
    color='red',
    linestyle='--',
    linewidth=1.0,
    label='Rapid rise threshold'
)
axes[3].set_title('Moisture Rate of Change')
axes[3].set_ylabel('Change per reading')
axes[3].set_xlabel('Reading Number')
axes[3].legend(loc='upper right')
axes[3].set_facecolor('#f8f8f8')

plt.tight_layout()
plt.savefig('plots/day04_sensor_overview.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("✅ Graph 1 saved: plots/day04_sensor_overview.png")

# ════════════════════════════════════════════
# GRAPH 2 — Distribution Histograms
# ════════════════════════════════════════════
print("\nCreating Graph 2: Distributions...")

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle(
    'Sensor Value Distributions',
    fontsize=14, fontweight='bold'
)

# Moisture distribution
axes[0,0].hist(
    df['moisture'],
    bins=50,
    color='green',
    alpha=0.7,
    edgecolor='black'
)
axes[0,0].axvline(
    x=3500, color='red',
    linestyle='--', label='Danger'
)
axes[0,0].axvline(
    x=2800, color='orange',
    linestyle='--', label='Warning'
)
axes[0,0].set_title('Soil Moisture Distribution')
axes[0,0].set_xlabel('Moisture Value')
axes[0,0].set_ylabel('Count')
axes[0,0].legend()

# Acceleration magnitude distribution
axes[0,1].hist(
    df['accel_magnitude'],
    bins=50,
    color='steelblue',
    alpha=0.7,
    edgecolor='black'
)
axes[0,1].axvline(
    x=4000, color='red',
    linestyle='--', label='Danger'
)
axes[0,1].set_title('Acceleration Magnitude Distribution')
axes[0,1].set_xlabel('Magnitude Value')
axes[0,1].set_ylabel('Count')
axes[0,1].legend()

# Moisture change distribution
axes[1,0].hist(
    df['moisture_change'],
    bins=50,
    color='orange',
    alpha=0.7,
    edgecolor='black'
)
axes[1,0].axvline(
    x=0, color='black',
    linewidth=1.0
)
axes[1,0].set_title('Moisture Rate of Change')
axes[1,0].set_xlabel('Change Value')
axes[1,0].set_ylabel('Count')

# Risk level distribution
risk_counts = df['risk'].value_counts().sort_index()
bars = axes[1,1].bar(
    ['🟢 Safe', '🟡 Warning', '🔴 Danger'],
    [risk_counts.get(0,0),
     risk_counts.get(1,0),
     risk_counts.get(2,0)],
    color=['green', 'orange', 'red'],
    edgecolor='black',
    alpha=0.8
)
axes[1,1].set_title('Risk Level Distribution')
axes[1,1].set_ylabel('Count')
for bar, count in zip(
    bars,
    [risk_counts.get(0,0),
     risk_counts.get(1,0),
     risk_counts.get(2,0)]
):
    axes[1,1].text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 10,
        str(count),
        ha='center', fontweight='bold'
    )

plt.tight_layout()
plt.savefig('plots/day04_distributions.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("✅ Graph 2 saved: plots/day04_distributions.png")

# ════════════════════════════════════════════
# GRAPH 3 — Scatter Plot
# ════════════════════════════════════════════
print("\nCreating Graph 3: Scatter Plot...")

plt.figure(figsize=(10, 6))
scatter_colors = {
    0: 'green',
    1: 'orange',
    2: 'red'
}
scatter_labels = {
    0: 'Safe',
    1: 'Warning',
    2: 'Danger'
}

for risk_level in [0, 1, 2]:
    mask = df['risk'] == risk_level
    plt.scatter(
        df[mask]['accel_magnitude'],
        df[mask]['moisture'],
        c=scatter_colors[risk_level],
        label=scatter_labels[risk_level],
        alpha=0.6,
        s=15
    )

plt.axhline(
    y=3500, color='red',
    linestyle='--', linewidth=1.5,
    label='Danger moisture threshold'
)
plt.axhline(
    y=2800, color='orange',
    linestyle='--', linewidth=1.0,
    label='Warning moisture threshold'
)
plt.title(
    'Acceleration vs Moisture\n'
    'Colored by Risk Level',
    fontsize=13, fontweight='bold'
)
plt.xlabel('Acceleration Magnitude')
plt.ylabel('Soil Moisture')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/day04_scatter.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("✅ Graph 3 saved: plots/day04_scatter.png")

# ════════════════════════════════════════════
# GRAPH 4 — Correlation Heatmap
# ════════════════════════════════════════════
print("\nCreating Graph 4: Correlation Heatmap...")

cols = [
    'accel_x', 'accel_y', 'accel_z',
    'moisture', 'vibration',
    'accel_magnitude', 'moisture_change',
    'risk'
]
corr = df[cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    linewidths=0.5,
    vmin=-1,
    vmax=1,
    center=0
)
plt.title(
    'Sensor Correlation Matrix\n'
    'How strongly each sensor'
    ' relates to others',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig('plots/day04_correlation.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("✅ Graph 4 saved: plots/day04_correlation.png")

# ── Final Summary ─────────────────────────────
print("\n" + "="*50)
print("   DAY 4 COMPLETE — VISUALIZATION")
print("="*50)
print("\n4 graphs created:")
print("  1. plots/day04_sensor_overview.png")
print("  2. plots/day04_distributions.png")
print("  3. plots/day04_scatter.png")
print("  4. plots/day04_correlation.png")

print("\n🔍 KEY OBSERVATIONS TO NOTE:")
print(f"  Most common risk: SAFE "
      f"({risk_counts.get(0,0)} readings)")
print(f"  Danger readings: "
      f"{risk_counts.get(2,0)} (rare!)")

# Highest correlation with risk
risk_corr = corr['risk'].drop('risk')
top_feature = risk_corr.abs().idxmax()
print(f"  Most correlated with risk: {top_feature}")

print("\n✅ Ready for Day 5!")
print("="*50)