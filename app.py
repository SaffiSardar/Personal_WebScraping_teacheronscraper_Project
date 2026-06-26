import numpy as np
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="Tutor Subject Predictor", page_icon="🎓")

# ─── Load the trained model bundle ──────────────────────────
@st.cache_resource
def load_bundle():
    return joblib.load("tutor_model.pkl")

bundle = load_bundle()

# Pull the country / unit options from the trained dummy columns
countries = sorted(c.replace("country_", "") for c in bundle["cat_columns"] if c.startswith("country_"))
units = [u.replace("unit_", "") for u in bundle["cat_columns"] if u.startswith("unit_")]

# ─── UI: input fields ───────────────────────────────────────
st.title("Tutor Subject Predictor")
st.write("Fill in a tutor's profile and the model will predict their subject.")

col1, col2 = st.columns(2)
with col1:
    total_exp = st.number_input("Total experience (years)", 0.0, 40.0, 8.0, 0.5)
    online_exp = st.number_input("Online experience (years)", 0.0, 40.0, 5.0, 0.5)
    country = st.selectbox("Country", countries)
    price_unit = st.selectbox("Billing cycle", units,
                              index=units.index("hour") if "hour" in units else 0)
with col2:
    price_min = st.number_input("Min price (USD)", 0.0, 1000.0, 10.0, 1.0)
    price_max = st.number_input("Max price (USD)", 0.0, 1000.0, 30.0, 1.0)

subjects_text = st.text_input("Subjects taught (comma separated)",
                              "Algebra, Calculus, Trigonometry, Geometry")
description_text = st.text_area("Profile description",
                                "I am a mathematics teacher who loves solving "
                                "equations and helping students with exams.")

# ─── Predict ────────────────────────────────────────────────
if st.button("Predict Subject"):
    price_avg = (price_min + price_max) / 2
    num_subj = max(len([s for s in subjects_text.split(",") if s.strip()]), 1)
    desc_len = len(description_text)

    # 1) numeric features (same order as training)
    numeric = pd.DataFrame([[total_exp, online_exp, price_min, price_max,
                             price_avg, num_subj, desc_len]],
                           columns=bundle["num_cols"])

    # 2) categorical dummies (all 0, set the matching ones to 1)
    cat = pd.DataFrame(0, index=[0], columns=bundle["cat_columns"])
    c_col = f"country_{country}"
    cat[c_col if c_col in cat.columns else "country_Other"] = 1
    u_col = f"unit_{price_unit}"
    if u_col in cat.columns:
        cat[u_col] = 1

    # 3) scale numeric + categorical
    base_scaled = bundle["scaler"].transform(pd.concat([numeric, cat], axis=1))

    # 4) TF-IDF on the combined text
    text_vec = bundle["tfidf"].transform([subjects_text + " " + description_text]).toarray()

    # 5) combine and predict (exactly like training)
    X_new = np.hstack([base_scaled, text_vec])
    pred = bundle["le"].inverse_transform(bundle["model"].predict(X_new))[0]

    st.success(f"### Predicted subject: **{pred.title()}**")

    # Confidence breakdown
    if hasattr(bundle["model"], "predict_proba"):
        probs = bundle["model"].predict_proba(X_new)[0]
        prob_df = pd.DataFrame({"Subject": bundle["le"].classes_, "Probability": probs})
        prob_df = prob_df.sort_values("Probability", ascending=False).set_index("Subject")
        st.bar_chart(prob_df)
