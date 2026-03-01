from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import joblib
import numpy as np

app = FastAPI(title="Vehicle Insurance Risk UI")
from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(directory="static"), name="static")

# Load your pre-trained ML model
model = joblib.load("model/anomaly_model_vehicle.joblib")

# Setup templates folder
templates = Jinja2Templates(directory="service/templates")


# Root page: renders the input form
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Prediction route for UI form
@app.post("/predict-ui", response_class=HTMLResponse)
def predict_ui(
    request: Request,
    age: int = Form(...),
    income: float = Form(...),
    vehicle_age: int = Form(...),
    past_claims: int = Form(...),
    claim_amount: float = Form(...),
    accident_severity: int = Form(...)
):
    # Prepare features for the model
    features = np.array([[age, income, vehicle_age, past_claims, claim_amount, accident_severity]])
    risk_numeric = model.predict(features)[0]

    # Map numeric prediction to human-readable risk level
    if risk_numeric == -1:
        risk_level = "High"
        low, medium, high = 0, 0, 1
    elif risk_numeric == 0:
        risk_level = "Medium"
        low, medium, high = 0, 1, 0
    else:
        risk_level = "Low"
        low, medium, high = 1, 0, 0

    return templates.TemplateResponse("result.html", {
        "request": request,
        "risk_score": float(risk_numeric),
        "risk_level": risk_level,
        "low": low,
        "medium": medium,
        "high": high
    }) 