# ============================================
# DAY 24 — Build First LSTM Model
# Date: 21 July 2026
# Author: Akshay
# Goal: Build and train LSTM that reads
#       10-reading sequences and predicts
#       SAFE / WARNING / DANGER
# ============================================
<<<<<<< HEAD
# ============================================
# DAY 24 OBSERVATIONS:
# 1. LSTM accuracy: 70.6% (vs RF 99.75%)
# 2. LSTM predicts SAFE for everything ❌
# 3. DANGER recall: 0% (critical problem!)
# 4. Training curve: unstable/oscillating
# 5. Root cause: only 32 danger sequences
#    not enough for LSTM to learn from
# 6. Fix planned for Day 25:
#    oversample danger + tune model
# ============================================
=======
>>>>>>> d028a6b5c48e2662f140d2d0665e9d79cfbbad1c

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*55)
print("   DAY 24 — LSTM MODEL TRAINING")
print("="*55)
print(f"TensorFlow version: {tf.__version__}")

# ════════════════════════════════════════════
# STEP 1: LOAD SEQUENCES FROM DAY 23
# ════════════════════════════════════════════
print("\n📂 STEP 1: Loading sequences...")

X_train = np.load(
    'data/processed/X_train_seq.npy'
)
X_test = np.load(
    'data/processed/X_test_seq.npy'
)
y_train = np.load(
    'data/processed/y_train_seq.npy'
)
y_test = np.load(
    'data/processed/y_test_seq.npy'
)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape:  {y_test.shape}")

# Extract dimensions
n_timesteps = X_train.shape[1]   # 10
n_features  = X_train.shape[2]   # 8
n_classes   = 3                   # SAFE/WARN/DANGER

print(f"\nModel will learn:")
print(f"  Timesteps : {n_timesteps} readings")
print(f"  Features  : {n_features} per reading")
print(f"  Classes   : {n_classes} risk levels")

# Check label distribution
print(f"\nTraining label distribution:")
unique, counts = np.unique(
    y_train, return_counts=True
)
labels = ['SAFE', 'WARNING', 'DANGER']
for label, count in zip(unique, counts):
    pct = count/len(y_train)*100
    print(f"  {labels[int(label)]}: "
          f"{count} ({pct:.1f}%)")

# ════════════════════════════════════════════
# STEP 2: CALCULATE CLASS WEIGHTS
# WHY: DANGER is rare (2%) but critical
#      Must force model to pay attention to it
# ════════════════════════════════════════════
print("\n📌 STEP 2: Calculating class weights...")

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(
    enumerate(class_weights)
)

print("Class weights (higher = more important):")
for i, weight in class_weight_dict.items():
    print(f"  {labels[i]:8s}: {weight:.2f}")

print("\nThis means LSTM will penalize")
print("DANGER mistakes ~34x more than SAFE!")

# ════════════════════════════════════════════
# STEP 3: BUILD LSTM MODEL
# ════════════════════════════════════════════
print("\n📌 STEP 3: Building LSTM architecture...")

model = models.Sequential([
    # Input shape: (10 timesteps, 8 features)
    layers.Input(shape=(n_timesteps, n_features)),

    # LSTM Layer 1: 64 units
    # return_sequences=True → passes ALL
    # timestep outputs to next LSTM layer
    layers.LSTM(64, return_sequences=True),

    # Dropout: randomly disable 20% of neurons
    # Prevents overfitting
    layers.Dropout(0.2),

    # LSTM Layer 2: 32 units
    # return_sequences=False (default)
    # → only returns final timestep output
    layers.LSTM(32),

    # Dropout again
    layers.Dropout(0.2),

    # Dense layer: combines LSTM output
    # relu: only passes positive values
    # "fires" when pattern is detected
    layers.Dense(16, activation='relu'),

    # Output layer: 3 neurons = 3 risk levels
    # softmax: converts to probabilities
    # that sum to 100%
    layers.Dense(n_classes, activation='softmax')
])

# Compile: set learning algorithm + metrics
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary
model.summary()

# ════════════════════════════════════════════
# STEP 4: SET UP CALLBACKS
# WHY: Smart training that stops at best point
# ════════════════════════════════════════════
print("\n📌 STEP 4: Setting up training callbacks...")

# Stop training if no improvement for 10 epochs
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# Save best model automatically during training
model_checkpoint = ModelCheckpoint(
    'models/lstm_best.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

print("✅ EarlyStopping: stops if no improvement")
print("   for 10 consecutive epochs")
print("✅ ModelCheckpoint: saves best model")
print("   automatically during training")

# ════════════════════════════════════════════
# STEP 5: TRAIN THE MODEL
# ════════════════════════════════════════════
print("\n📌 STEP 5: Training LSTM...")
print("(This may take 2-5 minutes)")
print("Watch val_accuracy — that's the real score")
print("-"*55)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[early_stopping, model_checkpoint],
    verbose=1
)

print("-"*55)
print("✅ Training complete!")

# ════════════════════════════════════════════
# STEP 6: EVALUATE MODEL
# ════════════════════════════════════════════
print("\n📌 STEP 6: Evaluating LSTM performance...")

# Predictions
y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)

# Overall accuracy
test_loss, test_acc = model.evaluate(
    X_test, y_test, verbose=0
)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")

# Detailed report — focus on DANGER recall!
print("\nDetailed Classification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=['SAFE', 'WARNING', 'DANGER']
))

# Critical check
danger_indices = np.where(y_test == 2)[0]
danger_predicted = y_pred[danger_indices]
danger_recall = (
    np.sum(danger_predicted == 2) /
    len(danger_indices)
) if len(danger_indices) > 0 else 0

print(f"🚨 DANGER Recall: "
      f"{danger_recall*100:.1f}%")
if danger_recall >= 0.90:
    print("✅ EXCELLENT! 90%+ DANGER recall")
elif danger_recall >= 0.70:
    print("⚠️  Good but needs improvement")
else:
    print("❌ Poor DANGER recall — dangerous!")

# ════════════════════════════════════════════
# STEP 7: VISUALIZE TRAINING + RESULTS
# ════════════════════════════════════════════
print("\n📌 STEP 7: Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    'Day 24 — LSTM Training Results',
    fontsize=14, fontweight='bold'
)

# Graph 1: Training history - accuracy
axes[0,0].plot(
    history.history['accuracy'],
    label='Training', color='steelblue'
)
axes[0,0].plot(
    history.history['val_accuracy'],
    label='Validation', color='orange'
)
axes[0,0].set_title('Model Accuracy over Epochs')
axes[0,0].set_xlabel('Epoch')
axes[0,0].set_ylabel('Accuracy')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Graph 2: Training history - loss
axes[0,1].plot(
    history.history['loss'],
    label='Training', color='steelblue'
)
axes[0,1].plot(
    history.history['val_loss'],
    label='Validation', color='orange'
)
axes[0,1].set_title('Model Loss over Epochs')
axes[0,1].set_xlabel('Epoch')
axes[0,1].set_ylabel('Loss')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Graph 3: Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm, annot=True, fmt='d',
    cmap='Blues', ax=axes[1,0],
    xticklabels=['SAFE','WARNING','DANGER'],
    yticklabels=['SAFE','WARNING','DANGER']
)
axes[1,0].set_title(
    f'Confusion Matrix\n'
    f'(Accuracy: {test_acc*100:.1f}%)'
)
axes[1,0].set_ylabel('Actual')
axes[1,0].set_xlabel('Predicted')

# Graph 4: RF vs LSTM comparison (preview)
models_comp = ['Random Forest', 'LSTM']
accuracies = [99.75, test_acc*100]
colors = ['steelblue', 'orange']
bars = axes[1,1].bar(
    models_comp, accuracies,
    color=colors, edgecolor='black',
    alpha=0.8
)
for bar, acc in zip(bars, accuracies):
    axes[1,1].text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.3,
        f'{acc:.2f}%',
        ha='center', fontweight='bold'
    )
axes[1,1].set_title('RF vs LSTM Accuracy')
axes[1,1].set_ylabel('Accuracy (%)')
axes[1,1].set_ylim(80, 102)

plt.tight_layout()
plt.savefig(
    'plots/day24_lstm_results.png',
    dpi=150, bbox_inches='tight'
)
plt.close()
print("✅ Visualization saved")

# ════════════════════════════════════════════
# STEP 8: SAVE FINAL MODEL
# ════════════════════════════════════════════
print("\n📌 STEP 8: Saving model...")

model.save('models/lstm_model.keras')
print("✅ Saved: models/lstm_model.keras")

# Save training history for Day 25 analysis
import json
history_dict = {
    'accuracy': history.history['accuracy'],
    'val_accuracy': history.history['val_accuracy'],
    'loss': history.history['loss'],
    'val_loss': history.history['val_loss']
}
with open('models/lstm_history.json', 'w') as f:
    json.dump(history_dict, f)
print("✅ Saved: models/lstm_history.json")

# ════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════
print("\n" + "="*55)
print("   DAY 24 COMPLETE — LSTM TRAINED!")
print("="*55)
print(f"\n📊 Results:")
print(f"  LSTM Accuracy     : {test_acc*100:.2f}%")
print(f"  Random Forest     : 99.75%")
print(f"  DANGER Recall     : {danger_recall*100:.1f}%")
print(f"  Epochs trained    : "
      f"{len(history.history['accuracy'])}")
print(f"\n💾 Saved:")
print(f"  models/lstm_model.keras")
print(f"  models/lstm_best.keras")
print(f"  models/lstm_history.json")
print(f"\n🚀 Next: Day 25 — Analyze and improve LSTM")
print("="*55)