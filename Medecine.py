
# PART 2: STREAMLIT WEB APP (app.py)
# =====================================================

import streamlit as st
import joblib
import numpy as np

# Load artifacts
model = joblib.load('xgb_medical_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
le_drug = joblib.load('drug_encoder.pkl')
le_cond = joblib.load('condition_encoder.pkl')
le_target = joblib.load('decision_encoder.pkl')

st.set_page_config(page_title='Medical AI Decision Support', layout='centered')

st.title('💊 Medical AI Decision Support')
st.warning('⚠️ This system is advisory only. Always consult a doctor.')

# User Inputs
drug = st.text_input('Drug name')
condition = st.text_input('Medical condition')
side_effects = st.text_area('Describe side effects')

# Safety rule (highest priority)
if 'breathing' in side_effects.lower():
    st.error('🚨 EMERGENCY: Breathing difficulty detected. Seek emergency care.')
    st.stop()

if st.button('Get Recommendation'):
    # ---------- Input validation ----------
    if drug.strip() == '' or condition.strip() == '' or side_effects.strip() == '':
        st.error('Please fill all fields.')
        st.stop()

    # ---------- Encode categorical safely ----------
    # ---------- Encode categorical safely with fallback ----------
    # If unseen drug/condition → use -1
    if drug.strip().lower() in le_drug.classes_:
        drug_enc = le_drug.transform([drug.strip().lower()])[0]
    else:
        drug_enc = -1

    if condition.strip().lower() in le_cond.classes_:
        cond_enc = le_cond.transform([condition.strip().lower()])[0]
    else:
        cond_enc = -1

    # ---------- Vectorize text ----------
    text_vec = vectorizer.transform([side_effects.lower()])

    if text_vec.shape[1] == 0:
        st.error('Text vectorization failed. Check TF-IDF vocabulary.')
        st.stop()

    X_input = np.hstack([
        text_vec.toarray(),
        [[drug_enc, cond_enc]]
    ])

    # ---------- Prediction ----------
    probs = model.predict_proba(X_input)[0]

    # Debug probabilities
    st.write('Prediction probabilities:', {
        'Continue': float(probs[0]),
        'See_Doctor': float(probs[1]),
        'Emergency': float(probs[2])
    })

    # ---------- Medical thresholds ----------
    if probs[2] > 0.35:
        decision = 'Emergency'
    elif probs[1] > 0.40:
        decision = 'See_Doctor'
    else:
        decision = 'Continue'

    st.subheader('AI Recommendation')

    if decision == 'Continue':
        st.success('✅ Symptoms appear mild. Continue medication and monitor.')
    elif decision == 'See_Doctor':
        st.warning('⚠️ Possible side effects. Please consult your doctor.')
    else:
        st.error('🚨 High risk detected. Seek medical help immediately.')

    st.caption('This is NOT a medical decision.')

