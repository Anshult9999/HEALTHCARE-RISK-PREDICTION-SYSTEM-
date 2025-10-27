
# ==========================================================
# Healthcare Risk Prediction & Recommendation System
# ----------------------------------------------------------
# Author: Anshul Tiwari
# Description:
#   End-to-end pipeline for predicting patient risk using
#   Logistic Regression and Random Forest classifiers.
#   Includes:
#   - Data loading & inspection
#   - Model training & evaluation
#   - Threshold optimization for recall (Class=1)
#   - Feature importance analysis
#   - Personalized recommendations engine
#   - Export of results (recommendations.csv)
# ==========================================================

# ================================
# 1. Import Libraries
# ================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve
)

# ================================
# 2. Load Dataset
# ================================
print("📥 Loading dataset...")
df = pd.read_csv("blood.csv")
print("✅ Dataset loaded. Shape:", df.shape)
print(df.head())

# Ensure 'Class' column exists
assert "Class" in df.columns, "Expected a 'Class' column (0/1)."

# Add patient_id if missing
if "patient_id" not in df.columns:
    df["patient_id"] = np.arange(1, len(df) + 1)

# ================================
# 3. Data Summary
# ================================
print("\n--- Dataset Info ---")
print(df.info())
print(df.describe())

print("\n--- Class Distribution ---")
print(df["Class"].value_counts())

# ================================
# 4. Feature / Target Split
# ================================
X = df.drop(columns=["Class"])
y = df["Class"]
feature_names = X.columns.tolist()

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ================================
# 5. Model Training (Logistic Regression & Random Forest)
# ================================
models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
}

results = []
print("\n=== MODEL COMPARISON ===")
for name, model in models.items():
    model.fit(X_train, y_train)
    proba_test = model.predict_proba(X_test)[:, 1]
    y_pred_default = (proba_test >= 0.50).astype(int)

    print(f"\n--- {name} ---")
    print("Confusion Matrix @0.50:\n", confusion_matrix(y_test, y_pred_default))
    print("Classification Report @0.50:\n", classification_report(y_test, y_pred_default, digits=3))
    roc_auc = roc_auc_score(y_test, proba_test)
    print("ROC-AUC:", round(roc_auc, 3))

    report = classification_report(y_test, y_pred_default, output_dict=True)
    results.append({
        "model": name,
        "roc_auc": roc_auc,
        "recall_class_1": report["1"]["recall"],
        "precision_class_1": report["1"]["precision"],
        "f1_class_1": report["1"]["f1-score"]
    })

# Pick best model by recall (Class=1)
best = max(results, key=lambda r: (r["recall_class_1"], r["roc_auc"]))
best_name = best["model"]
print(f"\n>>> Selected model for recommendations: {best_name}")
best_model = models[best_name]

# ================================
# 6. Threshold Optimization
# ================================
TARGET_RECALL = 0.80
proba_test = best_model.predict_proba(X_test)[:, 1]
prec, rec, thresh = precision_recall_curve(y_test, proba_test)
candidates = thresh[rec[:-1] >= TARGET_RECALL]
cut = float(np.max(candidates)) if len(candidates) > 0 else 0.50

print(f"\nChosen probability threshold: {round(cut, 3)} (target recall={TARGET_RECALL})")
y_pred_cut = (proba_test >= cut).astype(int)

print("\nConfusion Matrix @chosen threshold:\n", confusion_matrix(y_test, y_pred_cut))
print("Classification Report @chosen threshold:\n", classification_report(y_test, y_pred_cut, digits=3))
print("ROC-AUC:", round(roc_auc_score(y_test, proba_test), 3))

# ================================
# 7. Feature Importance
# ================================
def simple_feature_importance(model, X_cols):
    if isinstance(model, LogisticRegression):
        coef = np.abs(model.coef_[0])
        return pd.DataFrame({"feature": X_cols, "importance": coef}).sort_values("importance", ascending=False)
    if isinstance(model, RandomForestClassifier):
        imp = model.feature_importances_
        return pd.DataFrame({"feature": X_cols, "importance": imp}).sort_values("importance", ascending=False)
    return pd.DataFrame({"feature": X_cols, "importance": np.nan})

fi = simple_feature_importance(best_model, feature_names)
print("\nTop features driving predictions:")
print(fi.head(10))

# ================================
# 8. Risk Banding & Recommendations
# ================================
def risk_band(p, low=0.30, high=None):
    high = cut if high is None else high
    if p >= high:
        return "High"
    if p >= low:
        return "Medium"
    return "Low"

def build_recommendations(row, p, cols):
    recs = []
    band = risk_band(p)

    if band == "High":
        recs.append("⚠️ High risk predicted: schedule follow-up within 1–2 weeks.")
        recs.append("Review history & ensure adherence to care plan.")
    elif band == "Medium":
        recs.append("🔔 Medium risk: suggest check-in within 1–3 months.")
    else:
        recs.append("✅ Low risk: continue routine monitoring.")

    if "Recency" in cols and pd.notnull(row.get("Recency")) and row["Recency"] >= 14:
        recs.append("Long time since last engagement: send personalized reminder.")
    if "Frequency" in cols and pd.notnull(row.get("Frequency")) and row["Frequency"] <= 2:
        recs.append("Low engagement: offer onboarding/education session.")
    if "Time" in cols and pd.notnull(row.get("Time")) and row["Time"] < 12:
        recs.append("Newly onboarded: share starter tips and schedule review.")

    return " | ".join(recs[:4])

all_proba = best_model.predict_proba(X)[:, 1]
out = df[["patient_id"]].copy()
out["pred_prob_class1"] = all_proba
out["risk_band"] = out["pred_prob_class1"].apply(risk_band)
out["recommendations"] = [
    build_recommendations(row, p, set(df.columns))
    for (_, row), p in zip(df.iterrows(), out["pred_prob_class1"])
]

# ================================
# 9. Save Results
# ================================
out.to_csv("recommendations.csv", index=False)
print("\n💾 Saved: recommendations.csv")
print(out.head(10))

# ================================
# 10. Visualizations
# ================================
try:
    cm = confusion_matrix(y_test, y_pred_cut)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title(f"Confusion Matrix ({best_name}) @ threshold={round(cut,3)}")
    plt.show()

    if not fi.empty and fi["importance"].notna().any():
        plt.figure()
        sns.barplot(data=fi.head(10), x="importance", y="feature")
        plt.title(f"Top Features - {best_name}")
        plt.tight_layout()
        plt.show()
except Exception as e:
    print("Plotting skipped:", e)

print("\n✅ Project Completed Successfully!")
