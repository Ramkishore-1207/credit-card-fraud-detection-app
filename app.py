import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="centered")

FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.title("💳 Credit Card Fraud Detection")
st.write(
    "This app uses a trained Random Forest model to flag potentially fraudulent "
    "credit card transactions. It expects the same 30 features used during "
    "training: `Time`, `V1`-`V28` (PCA components), and `Amount`."
)

tab1, tab2 = st.tabs(["🔢 Manual Entry", "📁 Upload CSV"])

# ---------- Manual entry ----------
with tab1:
    st.subheader("Enter a single transaction")

    col1, col2 = st.columns(2)
    with col1:
        time_val = st.number_input("Time (seconds since first transaction)", value=0.0, step=1.0)
    with col2:
        amount_val = st.number_input("Amount", value=100.0, min_value=0.0, step=1.0)

    st.caption("V1–V28 are anonymized PCA components from the original dataset. "
               "Default them to 0 if you don't have real values to test with.")

    v_values = {}
    cols = st.columns(4)
    for i in range(1, 29):
        with cols[(i - 1) % 4]:
            v_values[f"V{i}"] = st.number_input(f"V{i}", value=0.0, key=f"v_{i}", format="%.4f")

    if st.button("Predict transaction", type="primary"):
        row = {"Time": time_val, **v_values, "Amount": amount_val}
        input_df = pd.DataFrame([row])[FEATURES]

        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        st.divider()
        if prediction == 1:
            st.error(f"⚠️ **Likely FRAUDULENT** — fraud probability: {proba:.2%}")
        else:
            st.success(f"✅ **Likely legitimate** — fraud probability: {proba:.2%}")

# ---------- CSV upload ----------
with tab2:
    st.subheader("Batch predict from a CSV file")
    st.caption(f"CSV must contain these columns: {', '.join(FEATURES)}")

    uploaded_file = st.file_uploader("Upload CSV", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        missing = [c for c in FEATURES if c not in df.columns]

        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
        else:
            st.write("Preview:", df.head())

            if st.button("Run predictions on file", type="primary"):
                X = df[FEATURES]
                preds = model.predict(X)
                probs = model.predict_proba(X)[:, 1]

                results = df.copy()
                results["Fraud_Prediction"] = np.where(preds == 1, "Fraud", "Legitimate")
                results["Fraud_Probability"] = probs

                st.write(f"Found **{(preds == 1).sum()}** potentially fraudulent transactions out of {len(preds)}.")
                st.dataframe(results)

                st.download_button(
                    "Download results as CSV",
                    results.to_csv(index=False),
                    "fraud_predictions.csv",
                    "text/csv",
                )

st.divider()
st.caption("Model: RandomForestClassifier (class_weight='balanced') trained on the Kaggle Credit Card Fraud dataset.")
