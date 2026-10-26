# Forensic Age Estimation Web App

Static web app for sex-specific forensic age estimation using the study workbook models.
The app also includes pooled sex-combined models trained on male and female rows together, with sex encoded as an additional predictor.
Kohler Method 2 supports partial input: at least two teeth are required, and the app uses separately trained models for the exact entered tooth combination.

## Files

- `index.html`, `styles.css`, `app.js`: practitioner-facing web app.
- `assets/forensic-age-models.js`: browser-ready trained model artifact.
- `assets/forensic-age-models.json`: same model artifact as JSON.
- `scripts/train_models.py`: retrains all models from the Excel workbook.

## Method Visibility

Method 1B (Demirjian) is currently hidden in the UI. To show it again, set `demirjian: true` in `METHOD_VISIBILITY` inside `app.js`.

## Retrain

```powershell
.\.venv\Scripts\python.exe scripts\train_models.py
```

## Run Locally

```powershell
.\.venv\Scripts\python.exe -m http.server 4173
```

Open `http://localhost:4173`.

## Deploy To Vercel

From this folder:

```powershell
npx vercel --prod
```

The app is static, so no patient or case measurements are sent to a server during use.
