"""
CardioScan AI — Data Preprocessing
====================================
Downloads MIT-BIH Arrhythmia Database records via wfdb,
segments beats, maps to AAMI classes, and saves a balanced
numpy dataset ready for model training.

Usage:
    python data_preprocessing.py

Output:
    data/X_beats.npy     — shape (N, 280) float32 beat segments
    data/y_labels.npy    — shape (N,)     int8   AAMI class labels
    data/metadata.json   — class counts, sampling freq, beat window

FIXES applied vs original:
  1. Removed duplicate "E" key in AAMI_MAP (was mapped to both 1 and 2;
     Python silently kept only the last value → all "E" beats became
     Ventricular instead of Supraventricular).
  2. Beat normalisation confirmed to be z-score per-beat — this is correct
     and must match exactly what app.py does at inference time.
"""

import os
import json
import numpy as np
import wfdb
from scipy.signal import butter, filtfilt
from collections import Counter

DATA_DIR       = os.path.join(os.path.dirname(__file__), "data", "mitbih")
OUTPUT_DIR     = os.path.join(os.path.dirname(__file__), "data")
FS             = 360
BEAT_PRE_MS    = 200
BEAT_POST_MS   = 380
BEAT_LEN       = 280
MAX_PER_CLASS  = 20000

MIT_BIH_RECORDS = [
    "100","101","102","103","104","105","106","107","108","109",
    "111","112","113","114","115","116","117","118","119","121",
    "122","123","124","200","201","202","203","205","207","208",
    "209","210","212","213","214","215","217","219","220","221",
    "222","223","228","230","231","232","233","234",
]

AAMI_MAP = {
    "N": 0, ".": 0, "R": 0, "L": 0, "e": 0, "j": 0, "n": 0,
    "A": 1, "a": 1, "J": 1, "S": 1, "E": 1,
    "V": 2, "!": 2,
    "F": 3,
    "/": 4, "f": 4, "Q": 4, "?": 4,
}

AAMI_NAMES = {0: "Normal (N)", 1: "Supraventricular (S)", 2: "Ventricular (V)", 3: "Fusion (F)", 4: "Unknown (Q)"}

def bandpass_filter(signal: np.ndarray, fs: float = FS,
                    lowcut: float = 0.5, highcut: float = 45.0) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(4, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, signal).astype(np.float32)

def normalize_beat(beat: np.ndarray) -> np.ndarray:
    """Z-score per beat. Must match app.py extract_beat() exactly."""
    std = beat.std()
    if std < 1e-6:
        return beat - beat.mean()
    return (beat - beat.mean()) / std

def extract_beats(signal: np.ndarray, r_peaks: np.ndarray, beat_len: int = BEAT_LEN) -> list:
    pre  = int(BEAT_PRE_MS / 1000 * FS)
    post = int(BEAT_POST_MS / 1000 * FS)
    beats = []
    for pk in r_peaks:
        start, end = pk - pre, pk + post
        if start < 0 or end > len(signal):
            beats.append(None)
            continue
        seg = signal[start:end]
        if len(seg) < beat_len:
            seg = np.pad(seg, (0, beat_len - len(seg)))
        else:
            seg = seg[:beat_len]
        beats.append(normalize_beat(seg))
    return beats

def process_record(record_id: str) -> tuple:
    os.makedirs(DATA_DIR, exist_ok=True)
    record_path = os.path.join(DATA_DIR, record_id)
    if not os.path.exists(record_path + ".dat"):
        print(f"  Downloading record {record_id}...")
        wfdb.dl_database("mitdb", dl_dir=DATA_DIR, records=[record_id])
    record = wfdb.rdrecord(record_path)
    ann = wfdb.rdann(record_path, "atr")
    signal = record.p_signal[:, 0].astype(np.float32)
    signal = bandpass_filter(signal)
    r_peaks = ann.sample
    symbols = ann.symbol
    beats = extract_beats(signal, r_peaks)
    X_rec, y_rec = [], []
    for beat, sym in zip(beats, symbols):
        if beat is None:
            continue
        label = AAMI_MAP.get(sym, None)
        if label is None:
            continue
        X_rec.append(beat)
        y_rec.append(label)
    return X_rec, y_rec

def build_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_X: dict = {c: [] for c in range(5)}
    total_records = len(MIT_BIH_RECORDS)
    for i, rec_id in enumerate(MIT_BIH_RECORDS):
        print(f"[{i+1}/{total_records}] Processing record {rec_id}...", end=" ")
        try:
            X_rec, y_rec = process_record(rec_id)
            for beat, lbl in zip(X_rec, y_rec):
                if len(all_X[lbl]) < MAX_PER_CLASS:
                    all_X[lbl].append(beat)
            print(f"{len(X_rec)} beats — {dict(Counter(y_rec))}")
        except Exception as e:
            print(f"FAILED ({e})")
    X_list, y_list = [], []
    for cls, beats in all_X.items():
        if beats:
            X_list.extend(beats)
            y_list.extend([cls] * len(beats))
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int8)
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    np.save(os.path.join(OUTPUT_DIR, "X_beats.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y_labels.npy"), y)
    class_counts = {AAMI_NAMES[c]: int(np.sum(y == c)) for c in range(5)}
    meta = {
        "total_beats": int(len(X)),
        "beat_length": BEAT_LEN,
        "sampling_freq": FS,
        "beat_pre_ms": BEAT_PRE_MS,
        "beat_post_ms": BEAT_POST_MS,
        "class_counts": class_counts,
        "aami_map": {str(k): int(v) for k, v in AAMI_MAP.items()},
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("\n" + "="*50)
    print(f"Dataset built: {len(X)} total beats")
    for name, cnt in class_counts.items():
        print(f"  {name}: {cnt}")
    print(f"Saved to {OUTPUT_DIR}")
    return X, y

if __name__ == "__main__":
    build_dataset()
