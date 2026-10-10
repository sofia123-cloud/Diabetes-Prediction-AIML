import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ---------------------------------------
# 1. Load the trained model and scaler
# ---------------------------------------

model = joblib.load("diabetes_logistic_model.pkl")
scaler = joblib.load("diabetes_scaler.pkl")

# ---------------------------------------
# 2. Website configuration
# ---------------------------------------

st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="centered"
)

# ---------------------------------------
# 3. Custom website styling
# ---------------------------------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eaf4ff, #f8faff);
}

h1 {
    color: #123b63;
    text-align: center;
    font-weight: 800;
    padding-bottom: 10px;
}

h2, h3 {
    color: #176b87;
}

div[data-testid="stNumberInput"] input {
    border-radius: 10px;
    border: 1px solid #b8d7ee;
}

.stButton > button {
    background: linear-gradient(90deg, #087f8c, #176b87);
    color: white;
    border-radius: 12px;
    border: none;
    padding: 12px 24px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton > button:hover {
    background: #064e63;
    color: white;
    transform: scale(1.02);
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------
# 4. Website title and introduction
# ---------------------------------------

st.title("🩺 Diabetes Prediction System")

st.markdown(
    """
    <div style="text-align:center; color:#456477;">
        An educational Machine Learning project using
        <b>Logistic Regression</b> to predict diabetes outcomes.
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

st.info(
    "⚕️ This tool is for educational purposes only. "
    "It cannot diagnose diabetes or replace professional medical advice."
)

st.divider()

# ---------------------------------------
# 5. Input section
# ---------------------------------------

st.header("📋 Enter Health Measurements")

st.write(
    "Enter the requested values below and click Predict "
    "to see the model's output."
)

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=2
    )

    glucose = st.number_input(
        "Glucose Level",
        min_value=1,
        max_value=300,
        value=120
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=1,
        max_value=200,
        value=70
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0,
        max_value=100,
        value=20
    )

with col2:
    insulin = st.number_input(
        "Insulin Level",
        min_value=0,
        max_value=900,
        value=79
    )

    bmi = st.number_input(
        "BMI",
        min_value=1.0,
        max_value=80.0,
        value=25.0,
        step=0.1
    )

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.5,
        step=0.01
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30
    )

st.write("")

# ---------------------------------------
# 6. Prediction button
# ---------------------------------------

if st.button("🔍 Predict", type="primary", use_container_width=True):

    # Prepare input in the same feature order as training
    input_data = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree,
        "Age": age
    }])

    # Replace zero values for measurements where
    # zero was treated as missing during training
    reference_data = pd.read_csv("diabetes.csv")

    columns_with_zero = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    for column in columns_with_zero:
        valid_values = reference_data[column].replace(0, np.nan)
        median_value = valid_values.median()

        input_data[column] = input_data[column].replace(
            0, median_value
        )

    # Scale input values
    scaled_input = scaler.transform(input_data)

    # Generate prediction
    prediction = model.predict(scaled_input)[0]

    st.divider()
    st.header("📊 Prediction Result")

    if prediction == 1:
        st.warning(
            "The model predicts the diabetes-positive class."
        )

        st.write(
            "This result is not a diagnosis. Consult a qualified "
            "healthcare professional for appropriate testing and advice."
        )

    else:
        st.success(
            "The model predicts the diabetes-negative class."
        )

        st.write(
            "This result does not rule out diabetes. "
            "Consult a healthcare professional if you have concerns."
        )

    st.caption(
        "This prediction is generated by a machine learning model "
        "and may be incorrect. It should not be used to make medical decisions."
    )

# ---------------------------------------
# 7. Footer
# ---------------------------------------

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#607d8b; font-size:13px;">
        <b>Diabetes Prediction System</b><br>
        Built using Python, Streamlit and Logistic Regression.<br>
        Academic AIML Project | SDG 3: Good Health and Well-being
    </div>
    """,
    unsafe_allow_html=True
)