import os
import json
import pickle
import tempfile
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from scipy.signal import butter, filtfilt, find_peaks

try:
    import wfdb
except ImportError:
    wfdb = None

FS = 360
BEAT_LEN = 280
BEAT_PRE_MS = 200
BEAT_POST_MS = 380
DS_POINTS = 2000
CONF_THRESHOLD = 0.60

AAMI_NAMES = {
    0: "Normal (N)",
    1: "Supraventricular (S)",
    2: "Ventricular (V)",
    3: "Fusion (F)",
    4: "Unknown (Q)",
}
AAMI_SEVERITY = {0: "normal", 1: "warning", 2: "critical", 3: "warning", 4: "warning"}

app = Flask(__name__)
CORS(app)
MODEL = None
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "arrhythmia_model.pkl")

class KerasArrhythmiaClassifier:
    def __init__(self, model, beat_len=BEAT_LEN):
        self.model = model
        self.beat_len = beat_len

    def predict_proba(self, X):
        X_3d = X.reshape(-1, self.beat_len, 1).astype(np.float32)
        return self.model.predict(X_3d, verbose=0)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

def load_model():
    global MODEL
    if not os.path.exists(MODEL_PATH):
        print(f"[WARN] Model not found at {MODEL_PATH}.")
        return
    try:
        with open(MODEL_PATH, "rb") as f:
            MODEL = pickle.load(f)
        print("[INFO] Model loaded.")
    except Exception as exc:
        print(f"[ERROR] Could not load model: {exc}")

def bandpass_filter(signal, lowcut=0.5, highcut=45.0):
    nyq = 0.5 * FS
    b, a = butter(4, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, signal).astype(np.float32)

def detect_r_peaks(signal):
    diff = np.diff(signal.astype(np.float64))
    squared = diff ** 2
    window = max(1, int(0.15 * FS))
    smoothed = np.convolve(squared, np.ones(window) / window, mode="same")
    threshold = np.mean(smoothed) + 1.5 * np.std(smoothed)
    peaks, _ = find_peaks(smoothed, height=threshold, distance=int(0.3 * FS))
    return peaks

def extract_beat(signal, r_peak):
    pre = int(BEAT_PRE_MS / 1000 * FS)
    post = int(BEAT_POST_MS / 1000 * FS)
    start, end = r_peak - pre, r_peak + post
    if start < 0 or end > len(signal):
        return None
    seg = signal[start:end]
    if len(seg) < BEAT_LEN:
        seg = np.pad(seg, (0, BEAT_LEN - len(seg)))
    else:
        seg = seg[:BEAT_LEN]
    std = seg.std()
    return (seg - seg.mean()) / std if std > 1e-6 else seg - seg.mean()

def classify_rhythm(hr, events, rr_intervals):
    if len(rr_intervals) < 2:
        return "Undetermined"
    rr_cv = np.std(rr_intervals) / np.mean(rr_intervals)
    vent_frac = sum("Ventricular" in e["type"] for e in events) / max(1, len(rr_intervals))
    if vent_frac > 0.30:
        return "Ventricular Arrhythmia"
    if rr_cv > 0.15:
        return "Irregular Rhythm"
    if hr < 60:
        return "Sinus Bradycardia"
    if hr > 100:
        return "Sinus Tachycardia"
    return "Normal Sinus"

def derive_overall_status(rhythm, events, total_beats):
    if total_beats == 0:
        return "normal"
    fraction = len(events) / total_beats
    has_ventricular = any("Ventricular" in e["type"] for e in events)
    if rhythm == "Normal Sinus" and fraction < 0.05:
        return "normal"
    if has_ventricular and fraction > 0.05:
        return "critical"
    if fraction > 0.20:
        return "warning"
    return "normal"

def process_ecg(record_dir, record_name):
    record = wfdb.rdrecord(os.path.join(record_dir, record_name))
    signal = record.p_signal[:, 0].astype(np.float32)
    actual_fs = float(record.fs)
    if actual_fs != FS:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(FS, int(actual_fs))
        signal = resample_poly(signal, FS // g, int(actual_fs) // g).astype(np.float32)

    signal = bandpass_filter(signal)
    r_peaks = detect_r_peaks(signal)
    rr_intervals = np.diff(r_peaks) / FS
    heart_rate = int(60 / np.mean(rr_intervals)) if len(rr_intervals) else 0

    beats, valid_peaks = [], []
    for peak in r_peaks:
        beat = extract_beat(signal, peak)
        if beat is not None:
            beats.append(beat)
            valid_peaks.append(peak)

    events = []
    mean_confidence = 0.0
    model_warning = None

    if MODEL is not None and beats:
        X = np.asarray(beats, dtype=np.float32)
        proba = MODEL.predict_proba(X)
        preds = np.argmax(proba, axis=1)
        max_probs = np.max(proba, axis=1)
        mean_confidence = float(np.mean(max_probs))
        q_fraction = float(np.mean(preds == 4))
        if q_fraction > 0.80:
            model_warning = (
                f"{q_fraction*100:.0f}% of beats were classified as Unknown (Q). "
                "This may indicate a distribution mismatch, noise, different lead placement, "
                "or another signal condition outside the training data."
            )
        for peak, pred, max_p in zip(valid_peaks, preds, max_probs):
            pred = int(pred)
            if pred == 0:
                continue
            events.append({
                "sample_index": int(peak),
                "time_sec": round(float(peak / FS), 3),
                "type": AAMI_NAMES[pred],
                "severity": AAMI_SEVERITY[pred],
                "confidence": round(float(max_p), 4),
                "low_confidence": bool(max_p < CONF_THRESHOLD),
            })

    rhythm = classify_rhythm(heart_rate, events, rr_intervals)
    status = derive_overall_status(rhythm, events, len(valid_peaks))
    ds = max(1, len(signal) // DS_POINTS)
    result = {
        "ecg_signal": signal[::ds].tolist(),
        "r_peaks": [int(p // ds) for p in valid_peaks],
        "arrhythmia_events": events,
        "mean_confidence": round(mean_confidence, 4),
        "heart_rate": heart_rate,
        "total_beats": len(valid_peaks),
        "rhythm": rhythm,
        "overall_status": status,
    }
    for event in events:
        event["sample_index"] = int(event["sample_index"] // ds)
    if model_warning:
        result["model_warning"] = model_warning
    return result

def demo_result():
    x = np.linspace(0, 20, DS_POINTS)
    signal = (0.08 * np.sin(2 * np.pi * 1.2 * x) +
              0.02 * np.sin(2 * np.pi * 3.5 * x)).tolist()
    return {
        "ecg_signal": signal,
        "r_peaks": list(range(80, DS_POINTS, 83)),
        "arrhythmia_events": [],
        "mean_confidence": 0.0,
        "heart_rate": 72,
        "total_beats": 24,
        "rhythm": "Normal Sinus",
        "overall_status": "normal",
        "demo": True,
        "model_warning": "Demo mode: these values are synthetic and are not model predictions.",
    }

@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": MODEL is not None,
        "wfdb_available": wfdb is not None,
    })

@app.post("/api/analyze")
def analyze():
    if os.getenv("CARDIOSCAN_DEMO", "0") == "1":
        return jsonify(demo_result())

    if wfdb is None:
        return jsonify({"error": "wfdb is not installed"}), 500
    dat_file = request.files.get("dat")
    hea_file = request.files.get("hea")
    if dat_file is None:
        return jsonify({"error": "Upload a WFDB .dat signal file."}), 400

    with tempfile.TemporaryDirectory() as temp_dir:
        base = os.path.splitext(dat_file.filename or "record.dat")[0]
        dat_file.save(os.path.join(temp_dir, base + ".dat"))
        if hea_file:
            hea_file.save(os.path.join(temp_dir, base + ".hea"))
        else:
            return jsonify({"error": "A matching .hea header file is required for WFDB records."}), 400
        try:
            result = process_ecg(temp_dir, base)
            return jsonify(result)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

load_model()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
