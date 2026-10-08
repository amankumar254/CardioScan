"""
CardioScan AI — CNN Model Training
=====================================
Trains a 1-D Convolutional Neural Network on the MIT-BIH preprocessed beats.

Architecture:
  Input (280,1)
  → Conv1D blocks with Batch Norm + Dropout
  → Global Average Pooling
  → Dense classifier (5 AAMI classes)

Usage:
    python data_preprocessing.py
    python train_model.py
"""

import os
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "..", "backend", "models")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
for d in [MODEL_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

BEAT_LEN = 280
N_CLASSES = 5
BATCH_SIZE = 128
EPOCHS = 60
LR_INITIAL = 1e-3
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42
AAMI_NAMES = ["Normal (N)", "Supraventricular (S)", "Ventricular (V)", "Fusion (F)", "Unknown (Q)"]

def conv_block(x, filters: int, kernel: int, dilation: int = 1, dropout: float = 0.2):
    shortcut = x
    x = layers.Conv1D(filters, kernel, padding="same", dilation_rate=dilation,
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters, kernel, padding="same", dilation_rate=dilation,
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, padding="same")(shortcut)
    x = layers.Add()([x, shortcut])
    return layers.Activation("relu")(x)

def build_cnn(input_len: int = BEAT_LEN, n_classes: int = N_CLASSES) -> keras.Model:
    inp = keras.Input(shape=(input_len, 1), name="ecg_beat")
    x = layers.Conv1D(32, 7, padding="same", kernel_regularizer=regularizers.l2(1e-4))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = conv_block(x, filters=64, kernel=5, dilation=1, dropout=0.2)
    x = layers.MaxPooling1D(2)(x)
    x = conv_block(x, filters=128, kernel=5, dilation=2, dropout=0.25)
    x = layers.MaxPooling1D(2)(x)
    x = conv_block(x, filters=256, kernel=3, dilation=4, dropout=0.3)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation="softmax", name="class_probs")(x)
    return keras.Model(inputs=inp, outputs=out, name="CardioScan_CNN")

class KerasArrhythmiaClassifier:
    def __init__(self, model: keras.Model, beat_len: int = BEAT_LEN):
        self.model = model
        self.beat_len = beat_len
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_3d = X.reshape(-1, self.beat_len, 1).astype(np.float32)
        return self.model.predict(X_3d, verbose=0)
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)
    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)
    @staticmethod
    def load(path: str) -> "KerasArrhythmiaClassifier":
        with open(path, "rb") as f:
            return pickle.load(f)

def load_data():
    X = np.load(os.path.join(DATA_DIR, "X_beats.npy"))
    y = np.load(os.path.join(DATA_DIR, "y_labels.npy")).astype(np.int32)
    print(f"Loaded dataset: {X.shape[0]} beats, {N_CLASSES} classes")
    return X, y

def train():
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    X, y = load_data()
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, stratify=y, random_state=RANDOM_SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=VAL_SPLIT / (1 - TEST_SPLIT),
        stratify=y_trainval, random_state=RANDOM_SEED)
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    X_train_3d = X_train[..., np.newaxis]
    X_val_3d = X_val[..., np.newaxis]
    X_test_3d = X_test[..., np.newaxis]

    classes = np.arange(N_CLASSES)
    cw = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight = {i: float(w) for i, w in enumerate(cw)}
    model = build_cnn()
    model.summary()

    total_steps = (len(X_train) // BATCH_SIZE) * EPOCHS
    lr_schedule = keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=LR_INITIAL,
        first_decay_steps=max(1, total_steps // 3),
        t_mul=1.5,
        m_mul=0.85,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(lr_schedule),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    cb_list = [
        callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, "best_weights.weights.h5"),
            monitor="val_accuracy", save_best_only=True,
            save_weights_only=True, verbose=1),
        callbacks.EarlyStopping(
            monitor="val_loss", patience=10,
            restore_best_weights=True, verbose=1),
        callbacks.TensorBoard(log_dir=os.path.join(BASE_DIR, "logs"), histogram_freq=0),
    ]

    history = model.fit(
        X_train_3d, y_train,
        validation_data=(X_val_3d, y_val),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        class_weight=class_weight, callbacks=cb_list, verbose=1)

    y_pred_proba = model.predict(X_test_3d, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    acc = accuracy_score(y_test, y_pred)
    f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_wtd = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_test, y_pred, target_names=AAMI_NAMES, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    keras_path = os.path.join(MODEL_DIR, "arrhythmia_model.keras")
    model.save(keras_path)
    KerasArrhythmiaClassifier(model, BEAT_LEN).save(
        os.path.join(MODEL_DIR, "arrhythmia_model.pkl"))

    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(os.path.join(MODEL_DIR, "training_history.json"), "w") as f:
        json.dump(hist_dict, f, indent=2)
    eval_report = {
        "test_accuracy": round(acc, 4),
        "macro_f1": round(f1_mac, 4),
        "weighted_f1": round(f1_wtd, 4),
        "confusion_matrix": cm.tolist(),
        "class_report": report,
        "n_test_samples": int(len(X_test)),
        "split_note": "Beat-level stratified split; not patient-independent.",
    }
    with open(os.path.join(MODEL_DIR, "evaluation_report.json"), "w") as f:
        json.dump(eval_report, f, indent=2)
    plot_history(history, acc)
    plot_confusion_matrix(cm, AAMI_NAMES)
    return model, eval_report

def plot_history(history, test_acc: float):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("CardioScan CNN — Training History", fontsize=13, fontweight="bold")
    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss", linestyle="--")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(history.history["accuracy"], label="Train Acc")
    axes[1].plot(history.history["val_accuracy"], label="Val Acc", linestyle="--")
    axes[1].axhline(test_acc, linestyle=":", label=f"Test Acc {test_acc*100:.1f}%")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "training_history.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_confusion_matrix(cm: np.ndarray, class_names: list):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="YlOrRd")
    ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Test Set", fontweight="bold")
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    train()
