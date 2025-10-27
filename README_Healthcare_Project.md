# Healthcare Risk Prediction & Recommendation System

## 📌 Project Overview
This project builds a **predictive healthcare recommendation system** using patient engagement and transaction history data.  
The goal is to classify patients into **risk categories (High, Medium, Low)** and generate **personalized non-clinical recommendations** to improve engagement and follow-ups.

Author: **Anshul Tiwari**  
Language: **Python 3.10+**  
Libraries: **scikit-learn, pandas, numpy, seaborn, matplotlib**  

---

## 📂 Dataset
- File: `blood.csv`
- Shape: 748 rows × 5 columns  
- Columns:
  - `Recency`: Time since last engagement (days)
  - `Frequency`: Number of past engagements
  - `Monetary`: Total spending/transaction value
  - `Time`: Duration (months)
  - `Class`: Target variable (0 = No follow-up needed, 1 = At-risk)

---

## 🔑 Workflow / Pipeline

### 1️⃣ Data Loading & Exploration
- Loaded patient dataset (`blood.csv`).
- Checked datatypes, null values, and class distribution.

### 2️⃣ Feature & Target Split
- Features (`Recency, Frequency, Monetary, Time`)  
- Target (`Class` → 0/1 classification)

### 3️⃣ Model Training
- **Logistic Regression** (baseline model with class balancing)
- **Random Forest Classifier** (ensemble model with class balancing)

### 4️⃣ Model Evaluation
Metrics used:
- **Confusion Matrix**
- **Classification Report (Precision, Recall, F1)**
- **ROC-AUC Score**

Results:
- Logistic Regression performed better for **recall on Class=1** (critical in healthcare).  
- Random Forest achieved higher precision, but lower recall.  

### 5️⃣ Threshold Tuning
- Default threshold = 0.50  
- Optimized threshold = **0.565** to achieve recall ≈ 0.80.  

### 6️⃣ Feature Importance
Top predictors (Logistic Regression):
- **Frequency** (0.20)
- **Recency** (0.07)
- **Time** (0.02)

### 7️⃣ Risk Banding & Recommendations
- **High Risk** → Schedule follow-up within 1–2 weeks  
- **Medium Risk** → Send reminders, follow-up in 1–3 months  
- **Low Risk** → Continue routine monitoring  

### 8️⃣ Save Recommendations
- All predictions exported to **`recommendations.csv`**
- Columns:
  - `patient_id`
  - `pred_prob_class1`
  - `risk_band`
  - `recommendations`

---

## 📊 Key Visuals
- Confusion Matrix at optimized threshold
- Feature Importance plot

---

## ⚙️ Installation & Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the pipeline:
```bash
python run_recommender.py
```

---

## 📈 Results Summary
- Logistic Regression selected for **best recall** on Class=1.  
- Achieved **R² Score (Recall focus)** = 0.806 on minority class.  
- Generated **patient-specific recommendations** saved to CSV.  

---

## 🚀 Future Improvements
- Add clinical features (BP, glucose, cholesterol).  
- Use advanced models (XGBoost, LightGBM).  
- Deploy via Flask/Django for real-time recommendations.  

---

✅ **Project Completed Successfully by Anshul Tiwari**
