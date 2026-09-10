import os
import tempfile
import subprocess
import numpy as np
import librosa
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from transformers import pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = 0 if torch.cuda.is_available() else -1
classifier = pipeline(
    "audio-classification",
    model="superb/wav2vec2-base-superb-er",
    device=device
)

TARGET_SR = 16000

def extract_pitch_features(y, sr):
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    valid_pitches = []
    for t in range(pitches.shape[1]):
        idx = magnitudes[:, t].argmax()
        p = pitches[idx, t]
        if 60 < p < 500:
            valid_pitches.append(float(p))

    if len(valid_pitches) < 2:
        return 0.0, 0.0

    pitch_std = float(np.std(valid_pitches))
    half = len(valid_pitches) // 2
    pitch_slope = float(np.mean(valid_pitches[half:]) - np.mean(valid_pitches[:half])) if half > 5 else 0.0

    return pitch_std, pitch_slope

def compute_confused_score(neu, sad, pitch_std, pitch_slope):
    pitch_std_norm = min(pitch_std / 50.0, 1.0)
    slope_norm = min(abs(pitch_slope) / 15.0, 1.0)
    raw = (neu * 0.35) + (sad * 0.15) + (pitch_std_norm * 0.30) + (slope_norm * 0.20)
    return float(max(0.0, min(raw, 1.0)))

@app.get("/")
async def root():
    return FileResponse("index.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    raw_path = None
    wav_path = None

    try:
        contents = await file.read()
        if not contents:
            return {"dominant_emotion": "neutral", "scores": {"neutral": 1.0}}

        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            tmp.write(contents)
            raw_path = tmp.name

        wav_path = raw_path + ".wav"

        subprocess.run(["ffmpeg", "-y", "-i", raw_path, "-ar", str(TARGET_SR), "-ac", "1", wav_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if not os.path.exists(wav_path):
            return {"dominant_emotion": "neutral", "scores": {"neutral": 1.0}}

        y, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)
        
        if y.size == 0 or np.max(np.abs(y)) < 1e-4:
            return {"dominant_emotion": "neutral", "scores": {"neutral": 1.0}}

        results = classifier(y, sampling_rate=TARGET_SR, top_k=5)
        scores_map = {r["label"].lower(): float(r["score"]) for r in results}

        hap = scores_map.get("hap", 0.0)
        sad = scores_map.get("sad", 0.0)
        ang = scores_map.get("ang", 0.0)
        neu = scores_map.get("neu", 0.0)

        pitch_std, pitch_slope = extract_pitch_features(y, sr)
        confused = compute_confused_score(neu, sad, pitch_std, pitch_slope)

        final_scores = {
            "happy": round(hap, 4),
            "sad": round(sad, 4),
            "angry": round(ang, 4),
            "confused": round(confused, 4),
            "neutral": round(neu, 4)
        }

        dominant = max(final_scores, key=final_scores.get)

        return {
            "dominant_emotion": dominant,
            "scores": final_scores,
            "raw_neutral_score": round(neu, 4),
            "pitch_std_hz": round(pitch_std, 3),
            "pitch_slope": round(pitch_slope, 5),
        }

    except Exception:
        return {"dominant_emotion": "neutral", "scores": {"neutral": 1.0}}
    finally:
        if raw_path and os.path.exists(raw_path):
            os.remove(raw_path)
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)
