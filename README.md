# 🚗 Intelligent Vehicle Risk Assessment System

A Machine Learning powered web application that predicts vehicle insurance risk levels using anomaly detection techniques.

This system classifies risk as:

- 🟢 Low
- 🟡 Medium
- 🔴 High

It also provides:
- Dynamic risk visualization
- Color-coded risk badge
- Intelligent explanation summary
- Clean animated UI

---

## 🧠 How It Works

The model analyzes:

- Age
- Income
- Vehicle Age
- Past Claims
- Claim Amount
- Accident Severity

It then predicts whether the profile represents an anomaly (higher insurance risk) or normal behavior.

Anomaly Detection Logic:
- `-1` → High Risk
- `0` → Medium Risk
- `1` → Low Risk

---

## 🛠 Tech Stack

- **FastAPI**
- **Scikit-learn**
- **HTML / CSS**
- **Chart.js**
- **Jinja2 Templates**

---

## 🚀 Run Locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload