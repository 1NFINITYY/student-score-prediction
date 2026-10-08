"""
predict.py — Reusable prediction module for Student Score Prediction.

Loads the saved Linear Regression model and preprocessor from disk
and exposes a single predict_exam_score() function.
"""

import os
import joblib
import numpy as np
import pandas as pd

# Resolve paths relative to this file so the module works regardless of cwd
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_PATH = os.path.join(_BASE_DIR, "models", "linear_regression_model.pkl")
_PREPROCESSOR_PATH = os.path.join(_BASE_DIR, "models", "preprocessor.pkl")

# Feature column order — must match what the notebook used during training
FEATURE_COLUMNS = [
    "Hours_Studied",
    "Attendance",
    "Parental_Involvement",
    "Access_to_Resources",
    "Extracurricular_Activities",
    "Sleep_Hours",
    "Previous_Scores",
    "Motivation_Level",
    "Internet_Access",
    "Tutoring_Sessions",
    "Family_Income",
    "Teacher_Quality",
    "School_Type",
    "Peer_Influence",
    "Physical_Activity",
    "Learning_Disabilities",
    "Parental_Education_Level",
    "Distance_from_Home",
    "Gender",
]


def load_model():
    """Load and return the saved Linear Regression model."""
    return joblib.load(_MODEL_PATH)


def load_preprocessor():
    """Load and return the saved ColumnTransformer preprocessor."""
    return joblib.load(_PREPROCESSOR_PATH)


# Load once at import time for efficiency
_model = load_model()
_preprocessor = load_preprocessor()


def predict_exam_score(student_data: dict) -> float:
    """
    Predict the exam score for a single student.

    Parameters
    ----------
    student_data : dict
        A dictionary with keys matching FEATURE_COLUMNS.

    Returns
    -------
    float
        Predicted exam score, clipped to [0, 100].
    """
    # Convert to a single-row DataFrame with the correct column order
    df = pd.DataFrame([student_data])[FEATURE_COLUMNS]

    # Transform using the fitted preprocessor (fitted on training data only)
    X_transformed = _preprocessor.transform(df)

    # Generate prediction
    raw_prediction = _model.predict(X_transformed)[0]

    # Clip to valid exam-score range
    return float(np.clip(raw_prediction, 0, 100))
