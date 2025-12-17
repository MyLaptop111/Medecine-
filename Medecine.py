import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# =============================
# LOAD ARTIFACTS
# =============================
model = joblib.load('medical_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
le_drug = joblib.load('drug_encoder.pkl')
le_cond = joblib.load('condition_encoder.pkl')
le_target = joblib.load('decision_encoder.pkl')

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title='Medical AI Decision Support',
    layout='centered'
)

st.title('💊 Medical AI Decision Support System')
st.caption('Model v1.1 | AI-assisted Clinical Decision Support')

st.warning(
    '⚠️ DISCLAIMER: Advisory system only. Always consult a healthcare professional.'
)

# =============================
# SESSION STATE
# =============================
if 'history' not in st.session_state:
    st.session_state.history = []

# =============================
# PATIENT INFORMATION
# =============================
st.subheader("🧑‍⚕️ Patient Information")

age = st.number_input("Age (years)", min_value=0, max_value=120, value=30)

gender = st.selectbox(
    "Gender",
    ["Male", "Female", "Other"]
)

weight = st.number_input(
    "Weight (kg)",
    min_value=1.0,
    max_value=300.0,
    value=70.0
)

smoker = st.selectbox(
    "Smoking Status",
    ["No", "Yes"]
)

chronic_diseases = st.multiselect(
    "Chronic Diseases (if any)",
    [
        "Diabetes",
        "Hypertension",
        "Heart Disease",
        "Kidney Disease",
        "Liver Disease",
        "Asthma",
        "None"
    ]
)

# =============================
# MEDICATION INFO
# =============================
st.subheader("💊 Medication Information")

drug = st.selectbox(
    'Select Drug',
    options=le_drug.classes_
)

condition = st.selectbox(
    'Select Medical Condition',
    options=le_cond.classes_
)

side_effects = st.text_area(
    'Describe side effects',
    placeholder='e.g. nausea, dizziness after 2 hours',
    height=120
)

# =============================
# RULE-BASED SAFETY ENGINE
# =============================
EMERGENCY = [
    'breathing', 'chest pain', 'seizure',
    'unconscious', 'anaphylaxis', 'swelling of face'
]

HIGH_RISK = [
    'vomiting blood', 'black stool',
    'severe rash', 'confusion'
]

text_lower = side_effects.lower()
rule_decision = None
risk_score = 0

# Symptom rules
if any(k in text_lower for k in EMERGENCY):
    st.error("🚨 EMERGENCY symptoms detected – Seek immediate medical attention")
    st.stop()

if any(k in text_lower for k in HIGH_RISK):
    risk_score += 2

# Patient risk factors
if age >= 65:
    risk_score += 1

if smoker == "Yes":
    risk_score += 1

if "Heart Disease" in chronic_diseases:
    risk_score += 2

if "Kidney Disease" in chronic_diseases or "Liver Disease" in chronic_diseases:
    risk_score += 2

if risk_score >= 4:
    rule_decision = "See_Doctor"

# =============================
# PREDICTION
# =============================
if st.button("🧠 Get AI Recommendation"):

    drug_enc = le_drug.transform([drug])[0]
    cond_enc = le_cond.transform([condition])[0]

    text_vec = vectorizer.transform([side_effects])

    X_input = np.hstack([
        text_vec.toarray(),
        [[drug_enc, cond_enc]]
    ])

    probs = model.predict_proba(X_input)[0]
    decision_idx = np.argmax(probs)
    model_decision = le_target.inverse_transform([decision_idx])[0]
    confidence = probs[decision_idx]

    # =============================
    # HYBRID FINAL DECISION
    # =============================
    final_decision = model_decision

    if rule_decision == "See_Doctor":
        final_decision = "See_Doctor"

    if confidence < 0.6:
        final_decision = "See_Doctor"

    # =============================
    # OUTPUT
    # =============================
    st.subheader("📊 Prediction Probabilities")

    st.json({
        le_target.inverse_transform([i])[0]: f"{p:.2%}"
        for i, p in enumerate(probs)
    })

    st.subheader("🧠 Explanation")

    st.write(f"""
    **Age:** {age}  
    **Gender:** {gender}  
    **Weight:** {weight} kg  
    **Smoker:** {smoker}  
    **Chronic Diseases:** {', '.join(chronic_diseases) if chronic_diseases else 'None'}  

    **Reported Symptoms:** {side_effects}  
    **Model Confidence:** {confidence:.2%}
    """)

    st.subheader("✅ Final Recommendation")

    if final_decision == "Continue":
        st.success("✅ Mild symptoms. Continue medication and monitor.")
    elif final_decision == "See_Doctor":
        st.warning("⚠️ Increased risk detected. Consult a doctor.")
    else:
        st.error("🚨 High risk detected. Seek medical attention.")

    # =============================
    # SAVE HISTORY
    # =============================
    record = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Age": age,
        "Gender": gender,
        "Weight": weight,
        "Smoker": smoker,
        "ChronicDiseases": ', '.join(chronic_diseases),
        "Drug": drug,
        "Condition": condition,
        "Symptoms": side_effects,
        "Decision": final_decision,
        "Confidence": round(confidence, 3)
    }

    st.session_state.history.append(record)

    pd.DataFrame([record]).to_csv(
        "medical_ai_logs.csv",
        mode="a",
        header=not pd.io.common.file_exists("medical_ai_logs.csv"),
        index=False
    )

# =============================
# HISTORY
# =============================
if st.session_state.history:
    st.subheader("📜 Previous Recommendations")
    st.dataframe(pd.DataFrame(st.session_state.history))

# =============================
# FOOTER
# =============================
st.caption(
    "⚕️ Educational Clinical Decision Support System – Not a diagnostic tool"
)

