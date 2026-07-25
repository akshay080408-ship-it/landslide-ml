# ============================================
# DAY 23 — Sequence Data Preparation for LSTM
# Date: 21 July 2026
# Author: Akshay
# Goal: Convert 2000 rows into sequences
#       that LSTM can train on
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*55)
print("   DAY 23 — LSTM SEQUENCE PREPARATION")
print("="*55)

# ════════════════════════════════════════════
# STEP 1: LOAD DATA
# ════════════════════════════════════════════
print("\n📂 STEP 1: Loading labeled dataset...")

df = pd.read_csv(
    'data/processed/sensor_labeled.csv'
)
print(f"Loaded: {len(df)} rows, "
      f"{len(df.columns)} columns")

# Same features as Random Forest
features = [
    'accel_x', 'accel_y', 'moisture',
    'vibration', 'accel_magnitude',
    'moisture_avg5', 'moisture_change',
    'magnitude_change'
]
target = 'risk_label'

print(f"Features: {len(features)}")
print(f"Label distribution:")
counts = df[target].value_counts().sort_index()
print(f"  Safe:    {counts[0]}")
print(f"  Warning: {counts[1]}")
print(f"  Danger:  {counts[2]}")

# ════════════════════════════════════════════
# STEP 2: NORMALIZE FEATURES
# WHY: LSTM is sensitive to value ranges
#      All features must be 0-1 scale
#      Unlike Random Forest which doesn't need this
# ════════════════════════════════════════════
print("\n📌 STEP 2: Normalizing features...")

X_raw = df[features].values
y_raw = df[target].values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_raw)

print(f"Before: moisture range "
      f"{X_raw[:,2].min():.0f} to "
      f"{X_raw[:,2].max():.0f}")
print(f"After:  moisture range "
      f"{X_scaled[:,2].min():.3f} to "
      f"{X_scaled[:,2].max():.3f}")
print("✅ All features normalized to 0-1")

# Save scaler for later use in Flask API
joblib.dump(scaler, 'models/lstm_scaler.pkl')
print("✅ Scaler saved: models/lstm_scaler.pkl")

# ════════════════════════════════════════════
# STEP 3: CREATE SEQUENCES
# WHY: LSTM learns from sequences of readings
#      not individual readings
# ════════════════════════════════════════════
print("\n📌 STEP 3: Creating sequences...")

SEQUENCE_LENGTH = 10  # 10 readings = 5 minutes

def create_sequences(X, y, seq_length):
    """
    Converts flat data into overlapping sequences.

    Input:
    X shape: (2000, 8) - individual readings
    y shape: (2000,)   - individual labels

    Output:
    X_seq shape: (1990, 10, 8) - sequences
    y_seq shape: (1990,)       - label at sequence end
    """
    X_sequences = []
    y_sequences = []

    for i in range(len(X) - seq_length):
        # Take seq_length consecutive readings
        sequence = X[i:i + seq_length]
        # Label = risk at the END of sequence
        label = y[i + seq_length]

        X_sequences.append(sequence)
        y_sequences.append(label)

    return np.array(X_sequences), np.array(y_sequences)

X_seq, y_seq = create_sequences(
    X_scaled, y_raw, SEQUENCE_LENGTH
)

print(f"\nBefore sequencing:")
print(f"  X shape: {X_scaled.shape}")
print(f"  y shape: {y_raw.shape}")
print(f"\nAfter sequencing:")
print(f"  X_seq shape: {X_seq.shape}")
print(f"  → {X_seq.shape[0]} sequences")
print(f"  → {X_seq.shape[1]} timesteps each")
print(f"  → {X_seq.shape[2]} features per timestep")
print(f"  y_seq shape: {y_seq.shape}")

# Verify label distribution preserved
seq_counts = pd.Series(y_seq).value_counts().sort_index()
print(f"\nSequence label distribution:")
print(f"  Safe:    {seq_counts[0]}")
print(f"  Warning: {seq_counts[1]}")
print(f"  Danger:  {seq_counts[2]}")

# ════════════════════════════════════════════
# STEP 4: TRAIN/TEST SPLIT
# WHY: Must split BEFORE sequences overlap
#      to prevent data leakage
# ════════════════════════════════════════════
print("\n📌 STEP 4: Train/Test split...")

# Important: use shuffle=False for time series!
# Shuffling would mix future data into training
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq,
    test_size=0.2,
    random_state=42,
    shuffle=False  # ← CRITICAL for time series!
)

print(f"Training sequences: {len(X_train)} (80%)")
print(f"Testing sequences:  {len(X_test)} (20%)")
print(f"\nTraining labels:")
train_counts = pd.Series(y_train).value_counts().sort_index()
print(f"  Safe: {train_counts.get(0,0)}, "
      f"Warning: {train_counts.get(1,0)}, "
      f"Danger: {train_counts.get(2,0)}")
print(f"Testing labels:")
test_counts = pd.Series(y_test).value_counts().sort_index()
print(f"  Safe: {test_counts.get(0,0)}, "
      f"Warning: {test_counts.get(1,0)}, "
      f"Danger: {test_counts.get(2,0)}")

# ════════════════════════════════════════════
# STEP 5: VISUALIZE ONE SEQUENCE
# WHY: Verify sequences look correct
#      before training the model
# ════════════════════════════════════════════
print("\n📌 STEP 5: Visualizing sequences...")

# Find first danger sequence
danger_idx = np.where(y_seq == 2)[0]
if len(danger_idx) > 0:
    sample_seq = X_seq[danger_idx[0]]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        'Day 23 — Sample DANGER Sequence\n'
        '10 readings leading to DANGER event',
        fontsize=13, fontweight='bold'
    )

    # Moisture (feature index 2)
    moisture_seq = sample_seq[:, 2]
    # Reverse normalize for display
    moisture_raw = (moisture_seq *
                    (X_raw[:,2].max() -
                     X_raw[:,2].min()) +
                    X_raw[:,2].min())

    axes[0,0].plot(
        range(1, 11), moisture_raw,
        'g-o', linewidth=2, markersize=6
    )
    axes[0,0].axhline(
        y=3500, color='red',
        linestyle='--', label='Danger threshold'
    )
    axes[0,0].set_title('Moisture in Sequence')
    axes[0,0].set_xlabel('Reading (1=oldest, 10=latest)')
    axes[0,0].set_ylabel('Moisture Value')
    axes[0,0].legend()

    # Acceleration magnitude (feature index 4)
    mag_seq = sample_seq[:, 4]
    mag_raw = (mag_seq *
               (X_raw[:,4].max() -
                X_raw[:,4].min()) +
               X_raw[:,4].min())

    axes[0,1].plot(
        range(1, 11), mag_raw,
        'b-s', linewidth=2, markersize=6
    )
    axes[0,1].axhline(
        y=4000, color='red',
        linestyle='--', label='Danger threshold'
    )
    axes[0,1].set_title('Magnitude in Sequence')
    axes[0,1].set_xlabel('Reading (1=oldest, 10=latest)')
    axes[0,1].set_ylabel('Magnitude Value')
    axes[0,1].legend()

    # moisture_change (feature index 6)
    change_seq = sample_seq[:, 6]
    change_raw = (change_seq *
                  (X_raw[:,6].max() -
                   X_raw[:,6].min()) +
                  X_raw[:,6].min())

    axes[1,0].bar(
        range(1, 11), change_raw,
        color=['red' if c > 400 else 'green'
               for c in change_raw]
    )
    axes[1,0].axhline(
        y=0, color='black', linewidth=0.5
    )
    axes[1,0].set_title('Moisture Change per Reading')
    axes[1,0].set_xlabel('Reading')
    axes[1,0].set_ylabel('Change Value')

    # All features normalized heatmap
    im = axes[1,1].imshow(
        sample_seq.T,
        aspect='auto',
        cmap='RdYlGn_r'
    )
    axes[1,1].set_title(
        'All Features (Normalized)\n'
        'Red=High  Green=Low'
    )
    axes[1,1].set_xlabel('Timestep (reading)')
    axes[1,1].set_ylabel('Feature index')
    axes[1,1].set_yticks(range(len(features)))
    axes[1,1].set_yticklabels(
        [f[:10] for f in features], fontsize=7
    )
    plt.colorbar(im, ax=axes[1,1])

    plt.tight_layout()
    plt.savefig(
        'plots/day23_sequence_visualization.png',
        dpi=150, bbox_inches='tight'
    )
    plt.close()
    print("✅ Sequence visualization saved")

# ════════════════════════════════════════════
# STEP 6: SAVE SEQUENCES FOR DAY 24
# ════════════════════════════════════════════
print("\n📌 STEP 6: Saving sequences...")

np.save('data/processed/X_train_seq.npy', X_train)
np.save('data/processed/X_test_seq.npy', X_test)
np.save('data/processed/y_train_seq.npy', y_train)
np.save('data/processed/y_test_seq.npy', y_test)

print("✅ Saved:")
print(f"  X_train_seq.npy: {X_train.shape}")
print(f"  X_test_seq.npy:  {X_test.shape}")
print(f"  y_train_seq.npy: {y_train.shape}")
print(f"  y_test_seq.npy:  {y_test.shape}")

# ════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════
print("\n" + "="*55)
print("   DAY 23 COMPLETE — SEQUENCES READY!")
print("="*55)
print(f"\n📊 Summary:")
print(f"  Original rows    : 2000")
print(f"  Sequence length  : {SEQUENCE_LENGTH} readings")
print(f"  Total sequences  : {len(X_seq)}")
print(f"  Training seqs    : {len(X_train)}")
print(f"  Testing seqs     : {len(X_test)}")
print(f"  Input shape      : {X_train.shape}")
print(f"  (samples, timesteps, features)")
print(f"\n⚠️  Key Decision: shuffle=False")
print(f"  Time series data must stay")
print(f"  in chronological order!")
print(f"\n🚀 Next: Day 24 — Build LSTM Model!")
print("="*55)