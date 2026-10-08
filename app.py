"""
app.py — Streamlit application for Student Exam Score Prediction.

Loads the saved Linear Regression model via src/predict.py and
provides an interactive UI to predict a student's exam score.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from src.predict import predict_exam_score

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Score Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for a polished look
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2d6a4f;
        border-left: 4px solid #52b788;
        padding-left: 10px;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
    .result-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        color: white;
        margin-top: 1rem;
    }
    .result-score {
        font-size: 3.5rem;
        font-weight: 700;
        color: #52b788;
    }
    .result-label {
        font-size: 1.1rem;
        color: #adb5bd;
        margin-top: 0.3rem;
    }
    .badge {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 50px;
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 0.8rem;
    }
    .disclaimer {
        font-size: 0.8rem;
        color: #888;
        font-style: italic;
        margin-top: 2rem;
        border-top: 1px solid #eee;
        padding-top: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="main-title">🎓 Student Score Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Predict a student\'s expected exam score using machine learning.</div>',
    unsafe_allow_html=True,
)

st.divider()

# ---------------------------------------------------------------------------
# Sidebar — Model information
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🤖 Model Information")
    st.markdown(
        """
        **Algorithm:** Linear Regression

        | Metric | Value |
        |--------|-------|
        | MAE    | 0.416 |
        | RMSE   | 1.521 |
        | R²     | 0.825 |
        """
    )
    st.info(
        "**MAE** = average prediction error in exam-score points.\n\n"
        "**R²** = proportion of variance in exam scores explained by the model (0–1, higher is better).",
        icon="ℹ️",
    )
    st.markdown("---")
    st.markdown(
        "**Dataset:** Student Performance Factors  \n"
        "**Rows:** ~6,607  \n"
        "**Features:** 19 academic, lifestyle & environmental variables"
    )

# ---------------------------------------------------------------------------
# Layout — two columns for the form
# ---------------------------------------------------------------------------
col1, col2 = st.columns(2, gap="large")

# ── Academic Information ────────────────────────────────────────────────────
with col1:
    st.markdown('<div class="section-header">📚 Academic Information</div>', unsafe_allow_html=True)

    hours_studied = st.slider("Hours Studied (per week)", min_value=1, max_value=44, value=20, step=1)
    attendance = st.slider("Attendance (%)", min_value=0, max_value=100, value=85, step=1)
    previous_scores = st.slider("Previous Scores (%)", min_value=0, max_value=100, value=75, step=1)
    tutoring_sessions = st.number_input(
        "Tutoring Sessions (per month)", min_value=0, max_value=8, value=2, step=1
    )

# ── Lifestyle ───────────────────────────────────────────────────────────────
with col2:
    st.markdown('<div class="section-header">🏃 Lifestyle</div>', unsafe_allow_html=True)

    sleep_hours = st.slider("Sleep Hours (per night)", min_value=4, max_value=10, value=7, step=1)
    physical_activity = st.slider(
        "Physical Activity (hours/week)", min_value=0, max_value=6, value=3, step=1
    )

# ── Student Environment ─────────────────────────────────────────────────────
st.markdown('<div class="section-header">🌍 Student Environment</div>', unsafe_allow_html=True)

env_col1, env_col2, env_col3 = st.columns(3, gap="medium")

with env_col1:
    parental_involvement = st.selectbox(
        "Parental Involvement", options=["Low", "Medium", "High"], index=2
    )
    access_to_resources = st.selectbox(
        "Access to Resources", options=["Low", "Medium", "High"], index=2
    )
    motivation_level = st.selectbox(
        "Motivation Level", options=["Low", "Medium", "High"], index=2
    )
    internet_access = st.radio("Internet Access", options=["Yes", "No"], horizontal=True)
    extracurricular = st.radio(
        "Extracurricular Activities", options=["Yes", "No"], horizontal=True
    )

with env_col2:
    family_income = st.selectbox(
        "Family Income", options=["Low", "Medium", "High"], index=1
    )
    teacher_quality = st.selectbox(
        "Teacher Quality", options=["Low", "Medium", "High"], index=2
    )
    school_type = st.radio("School Type", options=["Public", "Private"], horizontal=True)
    peer_influence = st.selectbox(
        "Peer Influence", options=["Negative", "Neutral", "Positive"], index=2
    )

with env_col3:
    learning_disabilities = st.radio(
        "Learning Disabilities", options=["No", "Yes"], horizontal=True
    )
    parental_education = st.selectbox(
        "Parental Education Level",
        options=["High School", "College", "Postgraduate"],
        index=2,
    )
    distance_from_home = st.selectbox(
        "Distance from Home", options=["Near", "Moderate", "Far"], index=0
    )
    gender = st.radio("Gender", options=["Male", "Female"], horizontal=True)

# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
validation_errors = []

if not (1 <= hours_studied <= 44):
    validation_errors.append("Hours Studied must be between 1 and 44.")
if not (0 <= attendance <= 100):
    validation_errors.append("Attendance must be between 0 and 100.")
if not (4 <= sleep_hours <= 10):
    validation_errors.append("Sleep Hours must be between 4 and 10.")
if not (0 <= previous_scores <= 100):
    validation_errors.append("Previous Scores must be between 0 and 100.")
if not (0 <= tutoring_sessions <= 8):
    validation_errors.append("Tutoring Sessions must be between 0 and 8.")
if not (0 <= physical_activity <= 6):
    validation_errors.append("Physical Activity must be between 0 and 6.")

if validation_errors:
    for err in validation_errors:
        st.error(err)

# ---------------------------------------------------------------------------
# Predict button
# ---------------------------------------------------------------------------
st.divider()
predict_clicked = st.button("🔮 Predict Exam Score", type="primary", use_container_width=True)

if predict_clicked:
    if validation_errors:
        st.warning("Please fix the validation errors above before predicting.")
    else:
        student_data = {
            "Hours_Studied": hours_studied,
            "Attendance": attendance,
            "Parental_Involvement": parental_involvement,
            "Access_to_Resources": access_to_resources,
            "Extracurricular_Activities": extracurricular,
            "Sleep_Hours": sleep_hours,
            "Previous_Scores": previous_scores,
            "Motivation_Level": motivation_level,
            "Internet_Access": internet_access,
            "Tutoring_Sessions": tutoring_sessions,
            "Family_Income": family_income,
            "Teacher_Quality": teacher_quality,
            "School_Type": school_type,
            "Peer_Influence": peer_influence,
            "Physical_Activity": physical_activity,
            "Learning_Disabilities": learning_disabilities,
            "Parental_Education_Level": parental_education,
            "Distance_from_Home": distance_from_home,
            "Gender": gender,
        }

        with st.spinner("Calculating prediction ..."):
            score = predict_exam_score(student_data)

        # Score interpretation
        if score >= 90:
            interpretation = "🏆 Excellent"
            badge_color = "#2d6a4f"
        elif score >= 80:
            interpretation = "🥇 Very Good"
            badge_color = "#40916c"
        elif score >= 70:
            interpretation = "👍 Good"
            badge_color = "#52b788"
        elif score >= 60:
            interpretation = "📊 Average"
            badge_color = "#f4a261"
        else:
            interpretation = "📉 Needs Improvement"
            badge_color = "#e63946"

        # Display result
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Predicted Exam Score</div>
                <div class="result-score">{score:.2f} <span style="font-size:1.8rem;color:#adb5bd;">/ 100</span></div>
                <div>
                    <span class="badge" style="background-color:{badge_color};">{interpretation}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Score breakdown expander
        with st.expander("📋 Input Summary"):
            import pandas as pd
            summary_df = pd.DataFrame(
                student_data.items(), columns=["Feature", "Value"]
            )
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Disclaimer
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="disclaimer">⚠️ This prediction is an estimate produced by a machine learning model and '
    "should not be treated as an official academic evaluation. Correlation does not imply causation.</div>",
    unsafe_allow_html=True,
)
