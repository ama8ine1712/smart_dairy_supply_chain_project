"""
AI-Driven Smart Dairy Supply Chain Optimization
Traditional ML Training & Evaluation Pipeline (CSV -> Preprocessing -> Scikit-Learn Models)
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, r2_score, mean_absolute_error, root_mean_squared_error
)

def main():
    # 1. Load CSV Dataset
    df = pd.read_csv("dairy_supply_chain_dataset.csv")
    print(f"[INFO] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # One-hot encode categorical supply-chain hubs & corridors
    df_encoded = pd.get_dummies(df, columns=["Collection_Center_BMC", "Route_Corridor"], drop_first=False)

    # =========================================================================
    # MODEL 1: Cold-Chain Spoilage Risk Classifier
    # =========================================================================
    clf_features = [
        "Season_Index", "Raw_Milk_Volume_Liters", "Fat_Content_Pct", "SNF_Content_Pct",
        "Initial_Chilling_Temp_C", "Reefer_Chamber_Temp_C", "Ambient_External_Temp_C",
        "Transit_Distance_Km", "Transit_Duration_Hrs", "Reefer_Door_Open_Count",
        "Thermal_Excursion_Mins", "Milk_pH_Level", "MBRT_Hours"
    ]
    X_clf = df_encoded[clf_features]
    y_clf = df_encoded["Cold_Chain_Spoilage_Risk"]

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_clf, y_clf, test_size=0.20, random_state=42, stratify=y_clf
    )

    scaler_c = StandardScaler()
    X_train_c_sc = scaler_c.fit_transform(X_train_c)
    X_test_c_sc = scaler_c.transform(X_test_c)

    # Baseline Logistic Regression
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train_c_sc, y_train_c)

    # Primary Random Forest Classifier
    rf_clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf_clf.fit(X_train_c, y_train_c)
    rf_preds = rf_clf.predict(X_test_c)
    rf_probs = rf_clf.predict_proba(X_test_c)[:, 1]

    print("\n" + "="*65)
    print("MODEL 1: COLD-CHAIN SPOILAGE RISK CLASSIFICATION (RANDOM FOREST)")
    print("="*65)
    print(f"Accuracy  : {accuracy_score(y_test_c, rf_preds):.4f}")
    print(f"Precision : {precision_score(y_test_c, rf_preds):.4f}")
    print(f"Recall    : {recall_score(y_test_c, rf_preds):.4f}")
    print(f"F1-Score  : {f1_score(y_test_c, rf_preds):.4f}")
    print(f"ROC-AUC   : {roc_auc_score(y_test_c, rf_probs):.4f}")
    print("\nConfusion Matrix:\n", confusion_matrix(y_test_c, rf_preds))
    print("\nClassification Report:\n", classification_report(y_test_c, rf_preds, target_names=["Safe (0)", "Spoilage Risk (1)"]))

    # =========================================================================
    # MODEL 2: Next-Day Regional Demand Forecaster
    # =========================================================================
    bmc_cols = [c for c in df_encoded.columns if c.startswith("Collection_Center_BMC_")]
    reg_features = [
        "Day_Of_Week", "Season_Index", "Raw_Milk_Volume_Liters",
        "Fat_Content_Pct", "Ambient_External_Temp_C", "Transit_Distance_Km"
    ] + bmc_cols

    X_reg = df_encoded[reg_features]
    y_reg = df_encoded["Next_Day_Demand_Liters"]

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg, y_reg, test_size=0.20, random_state=42
    )

    gbr_reg = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.06, random_state=42)
    gbr_reg.fit(X_train_r, y_train_r)
    gbr_preds = gbr_reg.predict(X_test_r)

    print("="*65)
    print("MODEL 2: NEXT-DAY MILK DEMAND FORECASTING (GRADIENT BOOSTING)")
    print("="*65)
    print(f"R2 Score : {r2_score(y_test_r, gbr_preds):.4f}")
    print(f"MAE      : {mean_absolute_error(y_test_r, gbr_preds):.2f} Liters")
    print(f"RMSE     : {root_mean_squared_error(y_test_r, gbr_preds):.2f} Liters")

    # Save trained models & preprocessing artifacts
    joblib.dump({
        "spoilage_classifier": rf_clf,
        "spoilage_scaler": scaler_c,
        "logistic_baseline": log_reg,
        "demand_forecaster": gbr_reg,
        "clf_features": clf_features,
        "reg_features": reg_features
    }, "dairy_supply_chain_models.joblib")
    print("\n[SUCCESS] Saved trained models to dairy_supply_chain_models.joblib")

if __name__ == "__main__":
    main()
