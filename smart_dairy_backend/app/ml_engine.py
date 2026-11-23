import os
import joblib
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, r2_score, mean_absolute_error
)
from app.models import MilkBatchRecord, MLModelRun
from app.config import settings


CLF_FEATURES = [
    "season_index", "raw_milk_volume_liters", "fat_content_pct", "snf_content_pct",
    "initial_chilling_temp_c", "reefer_chamber_temp_c", "ambient_external_temp_c",
    "transit_distance_km", "transit_duration_hrs", "reefer_door_open_count",
    "thermal_excursion_mins", "milk_ph_level", "mbrt_hours"
]

REG_BASE_FEATURES = [
    "day_of_week", "season_index", "raw_milk_volume_liters",
    "fat_content_pct", "ambient_external_temp_c", "transit_distance_km"
]


def train_models_from_db(db: Session) -> dict:
    """Fetches all MilkBatchRecord rows from the relational database, trains ML models,
    persists the joblib artifact, and logs metrics into ml_model_runs."""
    records = db.query(MilkBatchRecord).all()
    if len(records) < 50:
        raise ValueError(f"Insufficient records in database ({len(records)}). Need at least 50 rows.")

    rows = []
    for r in records:
        rows.append({
            "collection_center_bmc": r.collection_center_bmc,
            "day_of_week": r.day_of_week,
            "season_index": r.season_index,
            "raw_milk_volume_liters": r.raw_milk_volume_liters,
            "fat_content_pct": r.fat_content_pct,
            "snf_content_pct": r.snf_content_pct,
            "initial_chilling_temp_c": r.initial_chilling_temp_c,
            "reefer_chamber_temp_c": r.reefer_chamber_temp_c,
            "ambient_external_temp_c": r.ambient_external_temp_c,
            "transit_distance_km": r.transit_distance_km,
            "transit_duration_hrs": r.transit_duration_hrs,
            "reefer_door_open_count": r.reefer_door_open_count,
            "thermal_excursion_mins": r.thermal_excursion_mins,
            "milk_ph_level": r.milk_ph_level,
            "mbrt_hours": r.mbrt_hours,
            "cold_chain_spoilage_risk": r.cold_chain_spoilage_risk,
            "next_day_demand_liters": r.next_day_demand_liters,
        })

    df = pd.DataFrame(rows)
    df_enc = pd.get_dummies(df, columns=["collection_center_bmc"], drop_first=False)

    # 1. Cold-Chain Spoilage Classification
    X_clf = df_enc[CLF_FEATURES]
    y_clf = df_enc["cold_chain_spoilage_risk"]
    X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
        X_clf, y_clf, test_size=0.20, random_state=42, stratify=y_clf
    )

    scaler_c = StandardScaler()
    X_tr_c_sc = scaler_c.fit_transform(X_tr_c)
    X_te_c_sc = scaler_c.transform(X_te_c)

    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_tr_c_sc, y_tr_c)
    lr_preds = log_reg.predict(X_te_c_sc)

    rf_clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf_clf.fit(X_tr_c, y_tr_c)
    rf_preds = rf_clf.predict(X_te_c)
    rf_probs = rf_clf.predict_proba(X_te_c)[:, 1]

    # 2. Next-Day Regional Demand Forecasting
    bmc_cols = sorted([c for c in df_enc.columns if c.startswith("collection_center_bmc_")])
    reg_features = REG_BASE_FEATURES + bmc_cols
    X_reg = df_enc[reg_features]
    y_reg = df_enc["next_day_demand_liters"]

    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
        X_reg, y_reg, test_size=0.20, random_state=42
    )

    scaler_r = StandardScaler()
    X_tr_r_sc = scaler_r.fit_transform(X_tr_r)
    X_te_r_sc = scaler_r.transform(X_te_r)

    ridge_reg = Ridge(alpha=1.0)
    ridge_reg.fit(X_tr_r_sc, y_tr_r)
    ridge_preds = ridge_reg.predict(X_te_r_sc)

    gbr_reg = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.06, random_state=42)
    gbr_reg.fit(X_tr_r, y_tr_r)
    gbr_preds = gbr_reg.predict(X_te_r)

    metrics = {
        "training_rows": int(len(X_tr_c)),
        "test_rows": int(len(X_te_c)),
        "rf_accuracy": round(float(accuracy_score(y_te_c, rf_preds)), 4),
        "rf_f1_score": round(float(f1_score(y_te_c, rf_preds)), 4),
        "rf_roc_auc": round(float(roc_auc_score(y_te_c, rf_probs)), 4),
        "logistic_accuracy": round(float(accuracy_score(y_te_c, lr_preds)), 4),
        "gbr_r2_score": round(float(r2_score(y_te_r, gbr_preds)), 4),
        "gbr_mae_liters": round(float(mean_absolute_error(y_te_r, gbr_preds)), 2),
        "ridge_r2_score": round(float(r2_score(y_te_r, ridge_preds)), 4),
        "feature_importances": {
            feat: round(float(imp), 4)
            for feat, imp in sorted(zip(CLF_FEATURES, rf_clf.feature_importances_), key=lambda x: -x[1])
        }
    }

    os.makedirs(os.path.dirname(settings.MODEL_ARTIFACT_PATH), exist_ok=True)
    bundle = {
        "rf_clf": rf_clf,
        "log_reg": log_reg,
        "scaler_c": scaler_c,
        "gbr_reg": gbr_reg,
        "ridge_reg": ridge_reg,
        "scaler_r": scaler_r,
        "clf_features": CLF_FEATURES,
        "reg_features": reg_features,
        "bmc_cols": bmc_cols,
        "metrics": metrics,
    }
    joblib.dump(bundle, settings.MODEL_ARTIFACT_PATH)

    run_log = MLModelRun(
        training_rows=metrics["training_rows"],
        test_rows=metrics["test_rows"],
        rf_accuracy=metrics["rf_accuracy"],
        rf_f1_score=metrics["rf_f1_score"],
        rf_roc_auc=metrics["rf_roc_auc"],
        logistic_accuracy=metrics["logistic_accuracy"],
        gbr_r2_score=metrics["gbr_r2_score"],
        gbr_mae_liters=metrics["gbr_mae_liters"],
        ridge_r2_score=metrics["ridge_r2_score"],
        artifact_path=settings.MODEL_ARTIFACT_PATH,
    )
    db.add(run_log)
    db.commit()

    return metrics


def load_or_train_bundle(db: Session) -> dict:
    if not os.path.exists(settings.MODEL_ARTIFACT_PATH):
        train_models_from_db(db)
    return joblib.load(settings.MODEL_ARTIFACT_PATH)


def predict_batch_risk_and_demand(db: Session, payload: dict) -> dict:
    bundle = load_or_train_bundle(db)

    X_clf_df = pd.DataFrame([{feat: payload[feat] for feat in bundle["clf_features"]}])
    X_clf_sc = bundle["scaler_c"].transform(X_clf_df)

    prob_rf = float(bundle["rf_clf"].predict_proba(X_clf_df)[0, 1])
    prob_lr = float(bundle["log_reg"].predict_proba(X_clf_sc)[0, 1])

    reg_row = {feat: payload[feat] for feat in REG_BASE_FEATURES}
    target_col = f"collection_center_bmc_{payload['collection_center_bmc']}"
    for col in bundle["bmc_cols"]:
        reg_row[col] = 1 if col == target_col else 0

    X_reg_df = pd.DataFrame([reg_row])[bundle["reg_features"]]
    pred_demand = float(bundle["gbr_reg"].predict(X_reg_df)[0])

    if prob_rf >= 0.70:
        label = "CRITICAL_SPOILAGE_RISK"
        action = "Divert tanker to nearest BMC chill-station immediately & boost reefer compressor to MAX."
    elif prob_rf >= 0.40:
        label = "ELEVATED_THERMAL_WARNING"
        action = "Prioritize unload bay at Mother Dairy dock & lock rear reefer doors."
    else:
        label = "OPTIMAL_COLD_CHAIN"
        action = "Continue scheduled VRPTW milk collection corridor."

    return {
        "spoilage_probability_rf": round(prob_rf, 4),
        "spoilage_probability_logistic": round(prob_lr, 4),
        "spoilage_risk_label": label,
        "recommended_action": action,
        "predicted_next_day_demand_liters": round(pred_demand, 1),
        "feature_contributions": bundle["metrics"]["feature_importances"],
    }
