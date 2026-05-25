from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from data_processor import (
    load_and_process,
    get_race_distribution,
    get_feature_drift
)
from model import run_baseline, run_adversarial

app = FastAPI()

# CORS - allows Bolt.new frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Store processed data between requests
app_state = {}

@app.get("/")
def root():
    return {"status": "Clinical Fairness API running"}

@app.post("/upload")
async def upload_data(file: UploadFile = File(...)):
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

        return JSONResponse({
            "status": "success",
            "total_patients": total,
            "positive_rate": positive_rate,
            "race_distribution": distribution,
            "feature_drift": drift,
            "num_features": X.shape[1],
            "num_groups": len(distribution)
        })

    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )

@app.post("/baseline")
async def run_baseline_model():
    try:
        if 'X' not in app_state:
            return JSONResponse(
                {"status": "error",
                 "message": "Upload data first"},
                status_code=400
            )

        X = app_state['X']
        y = app_state['y']
        race = app_state['race']

        clf_metrics, fairness, X_test, y_test, race_test, y_prob = \
            run_baseline(X, y, race)

        app_state['X_test'] = X_test
        app_state['y_test'] = y_test
        app_state['race_test'] = race_test

        return JSONResponse({
            "status": "success",
            "classification": clf_metrics,
            "fairness": fairness
        })

    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )

@app.post("/mitigate")
async def run_mitigation(
    method: str = Form(...),
    alpha: float = Form(1.0),
    epochs: int = Form(50)
):
    try:
        if 'X' not in app_state:
            return JSONResponse(
                {"status": "error",
                 "message": "Upload data first"},
                status_code=400
            )

        X = app_state['X']
        y = app_state['y']
        race = app_state['race']

        if method == "adversarial":
            clf_metrics, fairness = run_adversarial(
                X, y, race,
                alpha=alpha,
                epochs=epochs
            )
        else:
            clf_metrics, fairness = run_adversarial(
                X, y, race,
                alpha=0.5,
                epochs=30
            )

        return JSONResponse({
            "status": "success",
            "method": method,
            "classification": clf_metrics,
            "fairness": fairness
        })

    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )