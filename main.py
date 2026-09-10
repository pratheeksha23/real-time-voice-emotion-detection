import io
import tempfile
import os
import numpy as np
import librosa
from fastapi import FastAPI, File, UploadFile, HTTPException
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

classifier = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er")

TARGET_SR = 16000


def extract_pitch_features(y, sr):
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
        )
    except Exception:
        return 0.0, 0.0

    if f0 is None:
        return 0.0, 0.0

    valid_mask = ~np.isnan(f0)
    valid_count = int(np.sum(valid_mask))

    if valid_count < 2:
        return 0.0, 0.0

    times = np.arange(len(f0))[valid_mask].astype(np.float64)
    values = f0[valid_mask].astype(np.float64)

    pitch_std = float(np.std(values))

    try:
        slope = float(np.polyfit(times, values, 1)[0])
    except Exception:
        slope = 0.0

    return pitch_std, slope


def compute_confused_score(neu, sad, pitch_std, pitch_slope):
    pitch_std_norm = min(pitch_std / 50.0, 1.0)
    slope_norm = min(abs(pitch_slope) / 5.0, 1.0)

    raw = (neu * 0.35) + (sad * 0.15) + (pitch_std_norm * 0.30) + (slope_norm * 0.20)
    return float(max(0.0, min(raw, 1.0)))


@app.get("/")
async def root():
    return FileResponse("index.html")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "audio.webm")[1] or ".webm"
    tmp_path = None

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty audio file")

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        try:
            y, sr = librosa.load(tmp_path, sr=TARGET_SR, mono=True)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not decode audio: {str(e)}")

        if y.size == 0 or np.max(np.abs(y)) < 1e-4:
            raise HTTPException(status_code=400, detail="Audio too quiet or empty")

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
        }

        dominant = max(final_scores, key=final_scores.get)

        return {
            "dominant_emotion": dominant,
            "scores": final_scores,
            "raw_neutral_score": round(neu, 4),
            "pitch_std_hz": round(pitch_std, 3),
            "pitch_slope": round(pitch_slope, 5),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
