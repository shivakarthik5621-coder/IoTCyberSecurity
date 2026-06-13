import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# GLOBAL MATPLOTLIB SETTINGS  (Figure Preparation Guide §1)
# =====================================================
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 100,        # on-screen preview
    'savefig.dpi': 300,       # final export
    'savefig.bbox': 'tight',
    'font.family': 'serif',   # matches Times in the paper body
})

from tensorflow.keras.layers import (
    Input, Conv1D, Dense, Add,
    TimeDistributed, Activation
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BENIGN_FILE = "final_training_set_benign.csv"
ATTACK_FILE = r"C:\Users\shiva\Downloads\attack_test_sampled.csv"

SEQ_LEN = 20
SEQ_STRIDE = 3
TARGET_ROWS = 200_000
LATENT_DIM = 32
BATCH_SIZE = 1024
EPOCHS = 32
THRESHOLD_PERCENTILE = 99.5


# =====================================================
# LOAD + CLEAN DATA
# =====================================================
print("Loading datasets...")
benign_df = pd.read_csv(BENIGN_FILE)
attack_df = pd.read_csv(ATTACK_FILE)

for df in [benign_df, attack_df]:
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

for col in ["label", "Label", "attack_label", "Attack_label"]:
    if col in attack_df.columns:
        attack_df.drop(columns=[col], inplace=True)

benign_num = benign_df.select_dtypes(include=[np.number])
attack_num = attack_df.select_dtypes(include=[np.number])

common_cols = sorted(set(benign_num.columns) & set(attack_num.columns))
benign_num = benign_num[common_cols]
attack_num = attack_num[common_cols]

if len(benign_num) > TARGET_ROWS:
    benign_num = benign_num.sample(n=TARGET_ROWS, random_state=SEED)

scaler = MinMaxScaler()
X_benign = scaler.fit_transform(benign_num)
X_attack = scaler.transform(attack_num)


# =====================================================
# SEQUENCE CREATION
# =====================================================
def create_sequences(X, L, stride):
    idxs = range(0, len(X) - L + 1, stride)
    return np.array([X[i:i + L] for i in idxs])

benign_seqs = create_sequences(X_benign, SEQ_LEN, SEQ_STRIDE)
attack_seqs = create_sequences(X_attack, SEQ_LEN, SEQ_STRIDE)

X_train, X_temp = train_test_split(
    benign_seqs, test_size=0.3, random_state=SEED
)
X_val, X_test_benign = train_test_split(
    X_temp, test_size=0.5, random_state=SEED
)

n_features = X_train.shape[2]


# =====================================================
# MODEL
# =====================================================
def tcn_block(x, filters, dilation):
    shortcut = x
    x = Conv1D(filters, 3, padding="causal",
               dilation_rate=dilation, activation="relu")(x)
    x = Conv1D(filters, 3, padding="causal",
               dilation_rate=dilation)(x)
    x = Add()([x, shortcut])
    x = Activation("relu")(x)
    return x

inputs = Input(shape=(SEQ_LEN, n_features))

x = Conv1D(64, 3, padding="causal", activation="relu")(inputs)
x = tcn_block(x, 64, 1)
x = tcn_block(x, 64, 2)
x = tcn_block(x, 64, 4)

latent = Conv1D(LATENT_DIM, 1, activation="relu")(x)

y = Conv1D(64, 1, activation="relu")(latent)
y = tcn_block(y, 64, 1)
y = tcn_block(y, 64, 2)
y = tcn_block(y, 64, 4)

outputs = TimeDistributed(Dense(n_features, activation="sigmoid"))(y)

model = Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="mse"
)

model.summary()


# =====================================================
# TRAINING
# =====================================================
train_ds = tf.data.Dataset.from_tensor_slices(
    (X_train, X_train)
).batch(BATCH_SIZE)

val_ds = tf.data.Dataset.from_tensor_slices(
    (X_val, X_val)
).batch(BATCH_SIZE)

callbacks = [
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-5
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    shuffle=False,
    callbacks=callbacks
)


# =====================================================
# THRESHOLD ESTIMATION
# =====================================================
pred_ben = model.predict(X_test_benign, batch_size=BATCH_SIZE)
ben_err = np.mean(np.abs(pred_ben - X_test_benign), axis=(1, 2))

threshold = np.percentile(ben_err, THRESHOLD_PERCENTILE)

print(f"Unsupervised threshold = {threshold:.6f}")


# =====================================================
# FIGURE 2 — TRAINING & VALIDATION LOSS  (Figure Preparation Guide §3)
# =====================================================
fig, ax = plt.subplots(figsize=(5.5, 3.5))
ax.plot(history.history['loss'],     label='Training loss',   linewidth=2)
ax.plot(history.history['val_loss'], label='Validation loss', linewidth=2, linestyle='--')
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE loss')
ax.set_title('Training vs Validation Loss')
ax.set_yscale('log')        # critical — exposes both convergence phases
ax.legend(frameon=True)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('fig2_loss.png', dpi=300, bbox_inches='tight')
plt.show()


# =====================================================
# EVALUATION
# =====================================================
pred_atk = model.predict(attack_seqs, batch_size=BATCH_SIZE)
atk_err = np.mean(np.abs(pred_atk - attack_seqs), axis=(1, 2))

y_true = np.concatenate([
    np.zeros_like(ben_err),
    np.ones_like(atk_err)
])

y_pred = np.concatenate([
    ben_err > threshold,
    atk_err > threshold
]).astype(int)

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = tp / (tp + fn + 1e-9)
fpr = fp / (fp + tn + 1e-9)
f1 = 2 * prec * rec / (prec + rec + 1e-9)

print("\n============ FINAL RESULTS ============")
print("Confusion Matrix:\n", cm)
print(f"Accuracy       : {acc:.4f}")
print(f"Precision      : {prec:.4f}")
print(f"Recall         : {rec:.4f}")
print(f"F1 Score       : {f1:.4f}")
print(f"False Pos Rate : {fpr:.4f}")
print("=====================================")


# =====================================================
# FIGURE 4 — CONFUSION MATRIX  (Figure Preparation Guide §5)
# =====================================================
cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
cm_labels = np.array([
    [f'{cm[i, j]:,}\n({cm_pct[i, j]:.2f}%)' for j in range(2)]
    for i in range(2)
])

fig, ax = plt.subplots(figsize=(4.5, 4))
sns.heatmap(
    cm, annot=cm_labels, fmt='', cmap='Blues', cbar=False,
    xticklabels=['Benign', 'Attack'],
    yticklabels=['Benign', 'Attack'],
    annot_kws={'size': 12}, ax=ax
)
ax.set_xlabel('Predicted label')
ax.set_ylabel('True label')
fig.tight_layout()
fig.savefig('fig4_confmat.png', dpi=300, bbox_inches='tight')
plt.show()

# =====================================================
# FIGURE 3 — RECONSTRUCTION ERROR DISTRIBUTION  (Figure Preparation Guide §4)
# =====================================================
fig, ax = plt.subplots(figsize=(5.5, 3.5))
ax.hist(ben_err, bins=80, density=True, alpha=0.6,
        label='Benign', color='steelblue', edgecolor='none')
ax.hist(atk_err, bins=80, density=True, alpha=0.6,
        label='Attack', color='darkorange', edgecolor='none')
ax.axvline(threshold, color='black', linestyle='--', linewidth=1.5,
           label=f'Threshold (\u03c4 = {threshold:.4f})')
ax.set_xlim(0, 0.15)          # clip — long attack tail isn't informative
ax.set_xlabel('Reconstruction error (MAE)')
ax.set_ylabel('Density')
ax.set_title('Reconstruction Error Distribution')
ax.set_yscale('log')          # makes the small benign distribution visible
ax.legend(frameon=True, loc='upper right')
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('fig3_error_dist.png', dpi=300, bbox_inches='tight')
plt.show()
