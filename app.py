import streamlit as st
import pandas as pd
import joblib

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Stroke Risk Predictor", layout="centered")

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

objects = load_model()

model = objects["model"]      
scaler = objects["scaler"]    
encoder = objects.get("encoder", None)  


model = load_model()

# ---------- BACKGROUND STYLE ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #667eea, #764ba2);
}
.block-container {
    background: rgba(255,255,255,0.08);
    padding: 2rem;
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.title("🧠 Stroke Risk Prediction System")
st.write("Enter patient health details below.")

# ---------- INPUT FORM ----------
with st.form("prediction_form"):

    age = st.slider("Age", 1, 100, 30)
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])
    avg_glucose_level = st.number_input("Average Glucose Level", 50.0, 300.0, 100.0)
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
    gender = st.selectbox("Gender", ["Male", "Female"])
    ever_married = st.selectbox("Ever Married", ["Yes", "No"])
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

    submit = st.form_submit_button("Predict Stroke Risk")

# ---------- PREDICTION ----------
if submit:

    # Convert categorical to numeric (MUST match training encoding)
    gender = 1 if gender == "Male" else 0
    ever_married = 1 if ever_married == "Yes" else 0
    residence_type = 1 if residence_type == "Urban" else 0

    work_map = {
        "Private": 0,
        "Self-employed": 1,
        "Govt_job": 2,
        "children": 3,
        "Never_worked": 4
    }

    smoke_map = {
        "formerly smoked": 0,
        "never smoked": 1,
        "smokes": 2,
        "Unknown": 3
    }

    input_data = pd.DataFrame([{
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "gender": gender,
        "ever_married": ever_married,
        "work_type": work_map[work_type],
        "Residence_type": residence_type,
        "smoking_status": smoke_map[smoking_status]
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("🩺 Prediction Result")

    if prediction == 1:
        st.error(f"⚠ High Stroke Risk ({probability:.2%} probability)")
    else:
        st.success(f"✅ Low Stroke Risk ({probability:.2%} probability)")
