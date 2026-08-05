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
    "This app uses a trained machine learning model to check whether a credit card "
    "transaction looks fraudulent or legitimate."
)

with st.expander("ℹ️ What is this model looking at? (read this first)"):
    st.markdown(
        """
        This model was trained on a well-known research dataset of real, anonymized
        European credit card transactions. To protect cardholder privacy, most of the
        original transaction details (merchant, location, card type, etc.) were mathematically
        transformed into 28 anonymous numeric features called **V1 to V28**. This means:

        - You **cannot** get V1-V28 values from a real receipt or card statement, they don't
          exist outside this dataset.
        - Only **Time** and **Amount** are "real world" numbers you can reason about.
        - **Simple mode** below lets you test with just Time and Amount (V1-V28 default to 0,
          representing an average/typical transaction pattern).
        - **Advanced mode** is for testing with exact V1-V28 values, e.g. rows copied from the
          dataset itself, useful for demos or verifying the model's accuracy.
        """
    )

mode = st.radio(
    "Choose how you want to test the model:",
    ["Simple mode (just amount & time)", "Advanced mode (all 30 features)", "Upload a CSV file"],
)

st.divider()

if mode.startswith("Simple"):
    st.subheader("Simple mode")
    st.caption(
        "Fill in the two real-world values below. The other 28 anonymized features "
        "will be set to 0 (a neutral/average value), so treat this as a rough demo "
        "rather than a precise fraud check."
    )

    amount_val = st.number_input(
        "Transaction amount ($)",
        min_value=0.0,
        value=100.0,
        step=1.0,
        help="The dollar amount of the transaction, e.g. 49.99 for a grocery purchase, "
             "or 1200 for a large electronics purchase.",
    )

    time_val = st.number_input(
        "Time (seconds since the first transaction in the dataset)",
        min_value=0.0,
        value=0.0,
        step=1.0,
        help="This is NOT a clock time (like 3:00 PM). In this dataset, 'Time' means "
             "how many seconds passed between this transaction and the very first "
             "transaction recorded. Leave at 0 if you're unsure, it barely affects the result.",
    )

    if st.button("Check this transaction", type="primary"):
        row = {"Time": time_val, **{f"V{i}": 0.0 for i in range(1, 29)}, "Amount": amount_val}
        input_df = pd.DataFrame([row])[FEATURES]

        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        st.divider()
        if prediction == 1:
            st.error(f"This looks like it could be FRAUDULENT - estimated fraud risk: {proba:.1%}")
        else:
            st.success(f"This looks legitimate - estimated fraud risk: {proba:.1%}")
        st.caption("Remember: with V1-V28 set to 0, this is only a rough estimate, not a real fraud check.")

elif mode.startswith("Advanced"):
    st.subheader("Advanced mode")
    st.caption(
        "Use this if you have exact values for all 30 features, for example, a row "
        "copied from the original dataset, or output from your own preprocessing pipeline."
    )

    col1, col2 = st.columns(2)
    with col1:
        time_val = st.number_input(
            "Time (seconds since first transaction)", value=0.0, step=1.0,
            help="Seconds elapsed since the first transaction in the dataset, not a clock time.",
        )
    with col2:
        amount_val = st.number_input(
            "Amount ($)", value=100.0, min_value=0.0, step=1.0,
            help="Dollar amount of the transaction.",
        )

    st.caption(
        "V1-V28 below are anonymized, PCA-transformed features from the original dataset. "
        "They don't correspond to anything you'd see on a receipt, only fill these in if "
        "you have exact values to test with (e.g. copied from a dataset row)."
    )

    v_values = {}
    cols = st.columns(4)
    for i in range(1, 29):
        with cols[(i - 1) % 4]:
            v_values[f"V{i}"] = st.number_input(f"V{i}", value=0.0, key=f"v_{i}", format="%.4f")

    if st.button("Check this transaction", type="primary"):
        row = {"Time": time_val, **v_values, "Amount": amount_val}
        input_df = pd.DataFrame([row])[FEATURES]

        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        st.divider()
        if prediction == 1:
            st.error(f"Likely FRAUDULENT - fraud probability: {proba:.2%}")
        else:
            st.success(f"Likely legitimate - fraud probability: {proba:.2%}")

else:
    st.subheader("Upload a CSV file")
    st.caption(
        f"Best for checking many transactions at once. Your file must have these exact "
        f"column names: {', '.join(FEATURES)}."
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        missing = [c for c in FEATURES if c not in df.columns]

        if missing:
            st.error(f"Your file is missing these required columns: {', '.join(missing)}")
        else:
            st.write("Preview of your file:")
            st.dataframe(df.head())

            if st.button("Check all transactions in this file", type="primary"):
                X = df[FEATURES]
                preds = model.predict(X)
                probs = model.predict_proba(X)[:, 1]

                results = df.copy()
                results["Prediction"] = np.where(preds == 1, "Fraud", "Legitimate")
                results["Fraud_Probability"] = probs.round(4)

                st.write(f"Found **{(preds == 1).sum()}** potentially fraudulent transaction(s) out of {len(preds)}.")
                st.dataframe(results)

                st.download_button(
                    "Download results as CSV",
                    results.to_csv(index=False),
                    "fraud_predictions.csv",
                    "text/csv",
                )

st.divider()
st.caption(
    "Model: Random forest classifier trained on the Kaggle Credit Card Fraud dataset. "
    "This is a demo tool for learning purposes, not a real fraud detection system."
)
