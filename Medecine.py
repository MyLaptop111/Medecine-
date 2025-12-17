import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# =============================
# LOAD MODEL
# =============================
model = joblib.load('medical_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
le_drug = joblib.load('drug_encoder.pkl')
le_cond = joblib.load('condition_encoder.pkl')
le_target = joblib.load('decision_encoder.pkl')

# =============================
# CONFIG
# =============================
st.set_page_config(page_title="Medical AI CDS", layout="centered")

st.title("💊 Medical AI Decision Support System")
st.caption("Clinical-Grade | Advisory Only")

st.warning("⚠️ This system does NOT replace professional medical advice.")

# =============================
# SESSION
# =============================
if 'history' not in st.session_state:
    st.session_state.history = []

# =============================
# PATIENT INFO
# =============================
st.subheader("🧑‍⚕️ Patient Information")

age = st.number_input("Age", 0, 120, 30)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
weight = st.number_input("Weight (kg)", 1.0, 300.0, 70.0)
smoker = st.selectbox("Smoker", ["No", "Yes"])
chronic = st.multiselect(
    "Chronic Diseases",
    ["Diabetes", "Hypertension", "Heart Disease", "Kidney Disease", "None"]
)

# =============================
# MEDICATION INFO
# =============================
st.subheader("💊 Medication")

drug = st.selectbox("Drug", le_drug.classes_)
condition = st.selectbox("Condition", le_cond.classes_)

side_effects = st.text_area(
    "Describe side effects",
    placeholder="e.g. nausea, dizziness after 2 hours"
)

if not side_effects.strip():
    st.stop()

# =============================
# RULE-BASED SAFETY
# =============================
EMERGENCY = ['breathing', 'chest pain', 'seizure', 'unconscious', 'anaphylaxis']
HIGH_RISK = ['vomiting blood', 'black stool', 'severe rash', 'confusion']

text = side_effects.lower()

if any(k in text for k in EMERGENCY):
    st.error("🚨 EMERGENCY – Seek immediate medical help")
    st.stop()

# =============================
# RISK SCORE
# =============================
risk_score = 0
if age >= 65: risk_score += 2
if smoker == "Yes": risk_score += 1
if "Heart Disease" in chronic: risk_score += 2
if "Kidney Disease" in chronic: risk_score += 2

st.metric("Patient Risk Score", risk_score, "/10")

# =============================
# PREDICTION
# =============================
if st.button("🧠 Get Recommendation"):

    drug_enc = le_drug.transform([drug])[0]
    cond_enc = le_cond.transform([condition])[0]

    text_vec = vectorizer.transform([side_effects])
    X = np.hstack([text_vec.toarray(), [[drug_enc, cond_enc]]])

    probs = model.predict_proba(X)[0]
    idx = np.argmax(probs)
    decision = le_target.inverse_transform([idx])[0]
    confidence = probs[idx]

    # =============================
    # THRESHOLDS
    # =============================
    thresholds = {
        "Continue": 0.55,
        "See_Doctor": 0.55,
        "Emergency": 0.40
    }

    if confidence < thresholds[decision]:
        decision = "See_Doctor"

    if np.max(probs) < 0.45:
        decision = "See_Doctor"

    if risk_score >= 4:
        decision = "See_Doctor"

    # =============================
    # OUTPUT
    # =============================
    st.subheader("📊 Prediction Probabilities")
    st.json({
        le_target.inverse_transform([i])[0]: f"{p:.2%}"
        for i, p in enumerate(probs)
    })

    st.subheader("✅ Final Recommendation")

    if decision == "Continue":
        st.success("✅ Continue medication and monitor.")
    elif decision == "See_Doctor":
        st.warning("⚠️ Consult a doctor as soon as possible.")
    else:
        st.error("🚨 Seek emergency medical attention.")

    # =============================
    # LOGGING
    # =============================
    record = {
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Age": age,
        "Smoker": smoker,
        "RiskScore": risk_score,
        "Drug": drug,
        "Condition": condition,
        "Symptoms": side_effects,
        "Decision": decision,
        "Confidence": round(confidence, 3)
    }

    st.session_state.history.append(record)

    pd.DataFrame([record]).to_csv(
        "medical_logs.csv",
        mode="a",
        header=not pd.io.common.file_exists("medical_logs.csv"),
        index=False
    )

# =============================
# HISTORY
# =============================
if st.session_state.history:
    st.subheader("📜 Previous Decisions")
    st.dataframe(pd.DataFrame(st.session_state.history))
    csv = pd.DataFrame(st.session_state.history).to_csv(index=False).encode('utf-8')
