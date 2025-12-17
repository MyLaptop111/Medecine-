import streamlit as st
import joblib
import numpy as np
import pandas as pd

# -----------------------------
# LOAD ARTIFACTS
# -----------------------------
model = joblib.load('medical_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
le_drug = joblib.load('drug_encoder.pkl')
le_cond = joblib.load('condition_encoder.pkl')
le_target = joblib.load('decision_encoder.pkl')

st.set_page_config(page_title='Medical AI Decision Support', layout='centered')
st.title('💊 Medical AI Decision Support')
st.warning('⚠️ This system is advisory only. Always consult a doctor.')

# -----------------------------
# USER INPUTS
# -----------------------------
drug = st.text_input('Drug name')
condition = st.text_input('Medical condition')
side_effects = st.text_area('Describe side effects')

# -----------------------------
# SAFETY RULE (Emergency Highest Priority)
# -----------------------------
emergency_keywords = ['breathing', 'chest pain', 'seizure', 'unconscious', 'swelling of face', 'anaphylaxis', 'fainting']
if any(k in side_effects.lower() for k in emergency_keywords):
    st.error('🚨 EMERGENCY detected in side effects! Seek immediate medical attention.')
    st.stop()

# -----------------------------
# PREDICTION
# -----------------------------
if st.button('Get Recommendation'):
    # Encode categorical features safely
    try:
        drug_enc = le_drug.transform([drug])[0]
    except:
        drug_enc = -1
    try:
        cond_enc = le_cond.transform([condition])[0]
    except:
        cond_enc = -1

    # TF-IDF vectorize side effects
    text_vec = vectorizer.transform([side_effects])
    X_input = np.hstack([text_vec.toarray(), [[drug_enc, cond_enc]]])

    # Prediction
    probs = model.predict_proba(X_input)[0]
    decision_idx = np.argmax(probs)
    decision = le_target.inverse_transform([decision_idx])[0]

    # Display probabilities
    st.write('Prediction Probabilities:', {le_target.inverse_transform([i])[0]: float(p) for i, p in enumerate(probs)})

    # Display recommendation
    if decision == 'Continue':
        st.success('✅ Symptoms appear mild. Continue medication and monitor.')
    elif decision == 'See_Doctor':
        st.warning('⚠️ Possible side effects. Please consult your doctor.')
    else:
        st.error('🚨 High risk detected. Seek medical help immediately.')

