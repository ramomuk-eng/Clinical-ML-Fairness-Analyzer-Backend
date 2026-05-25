from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from data_processor import (
    load_and_process,
    get_race_distribution,
    get_feature_drift
)
from model import run_baseline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app_state = {}

@app.get("/")
def root():
    return {"status": "Clinical Fairness API running"}

@app.post("/analyse")
async def analyse(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        X, y, race, scaler = load_and_process(file_bytes)

        app_state['X'] = X
        app_state['y'] = y
        app_state['race'] = race

        distribution = get_race_distribution(race)
        drift = get_feature_drift(X, race)
        total = len(y)
        positive_rate = round(float(np.mean(y)) * 100, 1)

        clf_metrics, fairness, X_test, y_test, race_test, y_prob = \
            run_baseline(X, y, race)

        app_state['X_test'] = X_test
        app_state['y_test'] = y_test
        app_state['race_test'] = race_test

        return JSONResponse({
            "status": "success",
            "upload": {
                "total_patients": total,
                "positive_rate": positive_rate,
                "race_distribution": distribution,
                "feature_drift": drift,
                "num_features": X.shape[1],
                "num_groups": len(distribution)
            },
            "baseline": {
                "classification": clf_metrics,
                "fairness": fairness
            }
        })

    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )

@app.post("/mitigate")
async def run_mitigation(file: UploadFile = File(...)):
    try:
        from model import run_adversarial
        from fastapi import Form

        file_bytes = await file.read()
        X, y, race, scaler = load_and_process(file_bytes)

        clf_metrics, fairness = run_adversarial(
            X, y, race,
            alpha=0.5,
            epochs=50
        )

        return JSONResponse({
            "status": "success",
            "method": "adversarial",
            "classification": clf_metrics,
            "fairness": fairness
        })

    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )