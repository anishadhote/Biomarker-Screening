# Biomarker Screening

A Streamlit web app that takes routine body and blood measurements for a patient and shows which group the profile sits closer to: **healthy control** or **patient**. It is a screening aid built on a small research dataset and is **not a medical diagnosis**.

## What it does

- Sign-in page with a password and an optional login ID
- Input form with nine measurements grouped as Patient, Glucose metabolism and Blood markers
- Each input box shows its normal range inside the box and the allowed range below it
- HOMA-IR is calculated automatically from glucose and insulin (`glucose × insulin / 405`)
- Result panel with a probability bar for both groups
- A warning when a value is far outside the range of the recorded cases, because the result is less reliable then

## Inputs

| Input | Unit | Allowed range |
|---|---|---|
| Age | years | 18–100 |
| BMI | kg/m² | 10–60 |
| Glucose | mg/dL | 30–500 |
| Insulin | µU/mL | 1–100 |
| Leptin | ng/mL | 1–120 |
| Adiponectin | µg/mL | 1–50 |
| Resistin | ng/mL | 1–100 |
| MCP-1 | pg/mL | 10–2000 |
| HOMA-IR | calculated | not entered |

Normal and typical ranges shown in the app are for adults and vary between laboratories.

## How the model works

The app trains its model from `datar2.csv` the first time it starts, so it always matches the installed scikit-learn version. The model is a soft-voting ensemble of five classifiers:

- Logistic regression
- Support vector machine
- k-nearest neighbours (k = 7, distance weighted)
- Random forest
- Gradient boosting

The first three use standardised features. Probabilities from all five are averaged, and a patient probability of 50% or more gives the "patient" result.

### Performance

Measured by repeated 5-fold cross-validation on the 116 recorded cases, so the numbers describe cases the model had not seen:

| Model | Accuracy | AUC |
|---|---|---|
| Ensemble used in this app | about 76% | about 0.85 |
| Earlier k-nearest neighbours model (k = 3) | about 71% | about 0.76 |

With only 116 cases, these figures have a wide margin of error. The model will be wrong for some patients.

## Run it locally

```bash
git clone <your-repository-url>
cd <your-repository-folder>
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create a new app and select `app.py` as the main file.
3. Open **Settings → Secrets** and set your own login details:

```toml
APP_USER = "your-login-id"
APP_PASSWORD = "your-password"
```

If these secrets are not set, the app falls back to default credentials written in `app.py`. Always set your own before sharing the app publicly.

## Project files

| File | Purpose |
|---|---|
| `app.py` | Streamlit app: styling, sign-in, form, model training and results |
| `datar2.csv` | Dataset of 116 cases with nine measurements and a group label |
| `requirements.txt` | Python dependencies |

The older `best_knn_model.pkl` and `scaler.pkl` files are no longer used and can be removed.

## Dataset

`datar2.csv` has 116 rows: 52 healthy controls (class 1) and 64 patients (class 2). Columns are `age`, `bmi`, `glucose`, `insulin`, `homa`, `leptin`, `adiponectin`, `resistin`, `mcp_1` and `classification`.

Dataset source: Breast Cancer Coimbra Dataset, UCI Machine Learning Repository.

## Limitations

- The model learns from a small dataset, so it is not suitable for clinical decisions.
- Results are less reliable for values outside the range of the recorded cases.
- The login uses one shared password. It keeps casual visitors out but is not a full account system.
- Normal ranges shown in the app are general adult values and may differ from your laboratory.

## Disclaimer

This project is for learning and demonstration. It does not diagnose, treat or prevent any condition. A qualified clinician should interpret any result.

