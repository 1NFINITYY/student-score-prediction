"""
test_saved_model.py — Sanity-check that the saved model and preprocessor work correctly.

Expected prediction for the sample student: approximately 74.96
"""

import sys
import os

# Allow running from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import load_model, load_preprocessor, predict_exam_score

# ---------------------------------------------------------------------------
# Sample student (same as used in the notebook for validation)
# ---------------------------------------------------------------------------
sample_student = {
    "Hours_Studied": 20,
    "Attendance": 90,
    "Parental_Involvement": "High",
    "Access_to_Resources": "High",
    "Extracurricular_Activities": "Yes",
    "Sleep_Hours": 8,
    "Previous_Scores": 80,
    "Motivation_Level": "High",
    "Internet_Access": "Yes",
    "Tutoring_Sessions": 2,
    "Family_Income": "High",
    "Teacher_Quality": "High",
    "School_Type": "Private",
    "Peer_Influence": "Positive",
    "Physical_Activity": 3,
    "Learning_Disabilities": "No",
    "Parental_Education_Level": "Postgraduate",
    "Distance_from_Home": "Near",
    "Gender": "Male",
}

# ---------------------------------------------------------------------------
# Run prediction
# ---------------------------------------------------------------------------
print("Loading saved model and preprocessor ...")
model = load_model()
preprocessor = load_preprocessor()
print(f"  Model      : {type(model).__name__}")
print(f"  Preprocessor: {type(preprocessor).__name__}")

predicted_score = predict_exam_score(sample_student)
print(f"\nPredicted Exam Score: {predicted_score:.2f} / 100")
print("(Notebook baseline ~= 74.96 -- small floating-point differences are normal)")
