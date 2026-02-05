import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Stroke Risk Predictor", layout="centered")

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    obj = joblib.load("model.joblib")

    if hasattr(obj, "predict"):
        return obj

    if isinstance(obj, dict):
        for key in obj:
            if hasattr(obj[key], "predict"):
                return obj[key]

    raise Exception("Model not found in model.joblib")

model = load_model()

# ---------- UI STYLE ----------
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #667eea, #764ba2); }
.block-container { background: rgba(255,255,255,0.08); padding: 2rem; border-radius: 15px; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Stroke Risk Prediction")

# ---------- FORM ----------
with st.form("form"):

    age = st.slider("Age", 1, 100, 30)
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])
    avg_glucose_level = st.number_input("Glucose Level", 50.0, 300.0, 100.0)
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0)

    gender = st.selectbox("Gender", ["Male", "Female"])
    ever_married = st.selectbox("Ever Married", ["Yes", "No"])
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

    submit = st.form_submit_button("Predict")

# ---------- PREDICTION ----------
if submit:

    input_data = pd.DataFrame([{
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "gender": gender,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": residence_type,
        "smoking_status": smoking_status
    }])

    # -------- FEATURE ENGINEERING (MATCH TRAINING) --------
    input_data["bmi_capped"] = input_data["bmi"].clip(upper=50)

    # One-Hot Encoding
    input_data = pd.get_dummies(input_data)

    # Add any missing training columns
    model_features = model.feature_names_in_

    for col in model_features:
        if col not in input_data.columns:
            input_data[col] = 0

    # Keep only columns model expects
    input_data = input_data[model_features]

    # -------- PREDICT --------
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("🩺 Result")

    if prediction == 1:
        st.error(f"⚠ High Stroke Risk ({probability:.2%})")
    else:
        st.success(f"✅ Low Stroke Risk ({probability:.2%})")
