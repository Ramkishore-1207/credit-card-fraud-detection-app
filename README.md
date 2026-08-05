# Credit Card Fraud Detection — Streamlit App

A Streamlit web app that serves a trained RandomForestClassifier for credit
card fraud detection (Kaggle Credit Card Fraud dataset schema: `Time`,
`V1`-`V28`, `Amount`).

## Files
- `app.py` — the Streamlit app
- `model.pkl` — the trained model (RandomForestClassifier, scikit-learn 1.6.1)
- `requirements.txt` — pinned dependencies

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy via GitHub + Streamlit Community Cloud

1. Push this folder to a new GitHub repo:
   ```bash
   git init
   git add .
   git commit -m "Fraud detection Streamlit app"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **New app** → select this repo/branch → set main file to `app.py`.
4. Click **Deploy**. You'll get a live URL like `https://<app-name>.streamlit.app`.

## Notes
- `requirements.txt` pins `scikit-learn==1.6.1` to match the version the
  model was trained/pickled with — using a different version can throw
  `InconsistentVersionWarning` or silently produce wrong predictions.
- `model.pkl` is ~2.9 MB, well under GitHub's file size limits, so no Git LFS
  needed.
