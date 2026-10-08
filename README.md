# 🎓 Student Score Prediction

A complete beginner-to-intermediate AI/ML portfolio project that predicts a student's exam score from 19 academic, lifestyle, and environmental features using Linear Regression — with a full EDA notebook and an interactive Streamlit web application.

---

## 📋 Overview

This project demonstrates the end-to-end machine learning lifecycle:

```
Raw Dataset → Data Cleaning → EDA → Preprocessing → Model Training
    → Model Evaluation → Model Comparison → Best Model Selection
    → Model Serialization → Prediction Module → Streamlit Application
```

---

## 🎯 Problem Statement

Given student-related features (study hours, attendance, parental involvement, etc.), predict the student's expected exam score as a continuous numerical value.

**Type:** Supervised Regression  
**Target variable:** `Exam_Score`

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source   | Student Performance Factors |
| Rows     | ~6,607 (after cleaning) |
| Columns  | 20 (19 features + 1 target) |
| Target   | `Exam_Score` (range 55–100) |

### Numerical Features
`Hours_Studied`, `Attendance`, `Sleep_Hours`, `Previous_Scores`, `Tutoring_Sessions`, `Physical_Activity`

### Categorical Features
`Parental_Involvement`, `Access_to_Resources`, `Extracurricular_Activities`, `Motivation_Level`, `Internet_Access`, `Family_Income`, `Teacher_Quality`, `School_Type`, `Peer_Influence`, `Learning_Disabilities`, `Parental_Education_Level`, `Distance_from_Home`, `Gender`

### Data Cleaning
- Missing categorical values → filled with most frequent value
- Invalid `Exam_Score` of `101` → converted to missing, row removed
- No duplicate rows found

---

## 🔍 Exploratory Data Analysis

Key findings from numerical correlation analysis:

| Feature | Correlation with Exam_Score |
|---|---:|
| Attendance | ≈ 0.58 |
| Hours_Studied | ≈ 0.45 |
| Previous_Scores | ≈ 0.17 |
| Tutoring_Sessions | ≈ 0.15 |
| Physical_Activity | ≈ 0.03 |
| Sleep_Hours | ≈ −0.02 |

**Attendance** and **Hours_Studied** showed the strongest numerical relationships with `Exam_Score`.

Categorical features such as `Parental_Involvement`, `Access_to_Resources`, `Motivation_Level`, `Family_Income`, and `Teacher_Quality` also showed notable differences in average exam scores across categories.

> ⚠️ Correlation does not imply causation.

---

## 🤖 Models Evaluated

- Linear Regression
- Decision Tree Regression
- Random Forest Regression

### Model Comparison

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| **Linear Regression** | **0.4160** | **1.5213** | **0.8250** |
| Random Forest | 1.0605 | 1.9230 | 0.7204 |
| Decision Tree | 1.5471 | 2.5168 | 0.5210 |

### ✅ Best Model: Linear Regression

Linear Regression outperformed both tree-based models on this dataset, achieving the lowest MAE and RMSE and the highest R² (0.825). This demonstrates that the relationships in this dataset are largely linear, and the simpler model generalizes better than more complex alternatives. Choosing Random Forest merely for its complexity would be a mistake here.

---

## 🛠 Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python |
| Data | Pandas, NumPy |
| ML | Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Model persistence | Joblib |
| App | Streamlit |
| Notebook | Jupyter |

---

## 📁 Project Structure

```
student-score-prediction/
│
├── data/
│   ├── StudentPerformanceFactors.csv          # Original dataset
│   └── StudentPerformanceFactors_Cleaned.csv  # Cleaned version
│
├── models/
│   ├── linear_regression_model.pkl            # Saved model
│   └── preprocessor.pkl                       # Saved ColumnTransformer
│
├── notebooks/
│   └── student_score_prediction.ipynb         # Full EDA & training notebook
│
├── src/
│   ├── __init__.py
│   ├── predict.py                             # Reusable prediction module
│   └── test_saved_model.py                    # Sanity-check script
│
├── app.py                                     # Streamlit application
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

```bash
git clone <repository-url>
cd student-score-prediction

# Create and activate a virtual environment
python -m venv .venv
```

**Windows:**
```bash
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

---

## 🚀 Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📓 Run the Notebook

Open the full analysis and training notebook:

```bash
jupyter notebook notebooks/student_score_prediction.ipynb
```

> **Note:** The notebook does not need to be re-run for the Streamlit app to work. The saved `.pkl` files in `models/` are ready to use.

---

## 🧪 Test the Saved Model

```bash
python src/test_saved_model.py
```

### Example Prediction

| Feature | Value |
|---------|-------|
| Hours Studied | 20 |
| Attendance | 90 % |
| Previous Scores | 80 |
| Sleep Hours | 8 |
| Tutoring Sessions | 2 |
| Physical Activity | 3 |
| Parental Involvement | High |
| Access to Resources | High |
| Motivation Level | High |
| School Type | Private |
| ... | ... |

**Predicted Exam Score: ~74.96 / 100**

---

## ⚠️ Limitations

- The model is trained on this specific dataset; performance on other populations may differ.
- Predictions are estimates, not official academic evaluations.
- Correlation does not imply causation — high attendance may correlate with high scores, but is not the sole cause.
- Dataset quality and representativeness directly affect model performance.
- The model should not be used for high-stakes academic decisions.

---

## 🔮 Future Improvements

- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
- Cross-validation for more robust evaluation
- Additional and more diverse datasets
- Advanced feature engineering
- Model explainability (SHAP values)
- Deployment to cloud (Streamlit Cloud, Heroku, GCP)
- Monitoring and model drift detection

---

## 📄 License

This project is for educational and portfolio purposes.
