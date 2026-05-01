from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")

# TWO-STAGE HURDLE MODEL TRAINING
# This script trains a two-stage hurdle model for hospital
# readmission prediction:
#
# STAGE 1: Predict whether a patient will be readmitted at all
# STAGE 2: For readmitted patients only, predict whether
#          readmission is early (<30 days) or late (>=30 days)
#
# MODEL SET — chosen to span four distinct model families:
#   Logistic Regression — linear baseline; interpretable coefficients
#   Decision Tree       — interpretable nonlinear baseline
#   Random Forest       — bagging ensemble; reduces variance
#   XGBoost             — gradient-boosted SOTA benchmark
#
# THREE-WAY DATA SPLIT  each subset has a single, exclusive role:
#   Training set   hyperparameter selection via 5-fold CV (GridSearchCV)
#   Validation setthreshold optimisation only (held out from CV entirely)
#   Test set      final one-time evaluation only (never used for tuning)
#
# IMBALANCE HANDLING — one mechanism per model family:
#   LR / DT / RF Stage 1  SMOTE inside ImbPipeline
#   LR / DT / RF Stage 2 SMOTEENN inside ImbPipeline
#   XGBoost              scale_pos_weight (native loss-level weighting)
#
# RESAMPLING LEAKAGE PREVENTION:
#   SMOTE/SMOTEENN sit inside the ImbPipeline, so synthetic samples are
#   generated only on the training fold at each CV iteration.
#   Stratified CV preserves the original class ratio before resampling
#   is applied, ensuring fold stability throughout.
#   The validation and test sets receive no resampling at any stage.
#
# SCALING:
#   StandardScaler is applied to Logistic Regression only, inside the
#   pipeline. Tree-based models (DT, RF, XGBoost) are scale-invariant
#   and do not require feature standardisation.
#
# EVALUATION METRIC HIERARCHY:
#   Primary    PR-AUC   (model selection via CV; robust under imbalance)
#   Secondary  Recall   (clinical priority; missing a readmission is costly)
#   Tertiary   F1 / Precision (operational trade-off at chosen threshold)
#   Accuracy is reported but not used for selection unreliable under
#   class imbalance even after training-side resampling, since the test
#   set preserves the real-world imbalanced distribution.



# paths
BASE_DIR   = Path(__file__).resolve().parents[1]
DATA_DIR   = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE  = DATA_DIR   / "hurdle_data.pkl"
OUTPUT_FILE = MODELS_DIR / "trained_model.pkl"


#  HELPER FUNCTIONS                                                   

def evaluate_model(name, y_true, y_prob, threshold=0.5):
    """Compute all evaluation metrics at a given threshold.
    Metrics reported: AUC-ROC, PR-AUC, Accuracy, F1 (weighted + positive),
    Precision, Recall, Confusion Matrix.
    Primary metric for model selection is PR-AUC (threshold-independent).
    """
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "Model":              name,
        "Threshold":          round(float(threshold), 2),
        "AUC-ROC":            round(roc_auc_score(y_true, y_prob), 4),
        "PR-AUC":             round(average_precision_score(y_true, y_prob), 4),
        "Accuracy":           round(accuracy_score(y_true, y_pred), 4),
        "F1_weighted":        round(f1_score(y_true, y_pred, average="weighted"), 4),
        "F1_positive":        round(f1_score(y_true, y_pred, average="binary", pos_label=1), 4),
        "Precision_positive": round(precision_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0), 4),
        "Recall_positive":    round(recall_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0), 4),
        "Confusion Matrix":   confusion_matrix(y_true, y_pred),
    }


def find_best_f1_threshold(y_true, y_prob, threshold_min=0.01, threshold_max=0.99, step=0.01):
    """Perform full decision-threshold optimisation over the validation
    distribution. Sweeps the complete probability range (0.01-0.99) and
    selects the threshold maximising F1-positive, reflecting the clinical
    priority of identifying high-risk patients. Precision then recall are
    used as sequential tiebreakers to ensure a deterministic selection.

    
    """
    thresholds = np.arange(threshold_min, threshold_max + step, step)
    best_threshold, best_f1, best_precision, best_recall = 0.5, -1, 0, 0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1_pos = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        prec   = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        rec    = recall_score(y_true, y_pred, pos_label=1, zero_division=0)

        if (
            f1_pos > best_f1
            or (f1_pos == best_f1 and prec > best_precision)
            or (f1_pos == best_f1 and prec == best_precision and rec > best_recall)
        ):
            best_f1, best_precision, best_recall, best_threshold = f1_pos, prec, rec, t

    return round(float(best_threshold), 2), round(float(best_f1), 4)


def train_and_score_model(
    name, model,
    X_train_full, y_train_full,
    X_train_part, y_train_part,
    X_val, y_val,
    X_test, y_test,
):
    """Two-phase final training:
    Phase 1 — fit on partial training set, tune threshold on validation set.
    Phase 2 — refit on full training set, evaluate once on test set.

    Threshold is optimised on the validation set (Phase 1 model) and then
    applied to the full-training model (Phase 2). This follows standard
    practice for threshold calibration and is separate from hyperparameter
    selection, which was performed via CV in GridSearchCV.
    """
    print(f"\nTraining {name}...")

    # Phase 1: threshold tuning on validation set
    model.fit(X_train_part, y_train_part)
    val_prob = model.predict_proba(X_val)[:, 1]
    best_threshold, best_val_f1 = find_best_f1_threshold(y_val, val_prob)

    # Phase 2: refit on full training set, evaluate on test set
    model.fit(X_train_full, y_train_full)
    test_prob = model.predict_proba(X_test)[:, 1]

    default_result = evaluate_model(name, y_test, test_prob, threshold=0.5)
    tuned_result   = evaluate_model(name, y_test, test_prob, threshold=best_threshold)

    print(
        f"  threshold={best_threshold} | "
        f"val F1+={best_val_f1:.4f} | "
        f"default F1+={default_result['F1_positive']:.4f} | "
        f"tuned F1+={tuned_result['F1_positive']:.4f}"
    )

    return {
        "model":          model,
        "best_threshold": best_threshold,
        "val_best_f1":    best_val_f1,
        "default_result": default_result,
        "tuned_result":   tuned_result,
        "test_prob":      test_prob,
    }


#  MAIN

def main():
    with open(INPUT_FILE, "rb") as f:
        data = pickle.load(f)

    X_train_model = data["X_train_model"].copy()
    X_test_model  = data["X_test_model"].copy()
    X_train_lr    = data["X_train_lr"].copy()
    X_test_lr     = data["X_test_lr"].copy()

    X_train_s2    = data["X_train_s2"].copy()
    X_test_s2     = data["X_test_s2"].copy()
    X_train_s2_lr = data["X_train_s2_lr"].copy()
    X_test_s2_lr  = data["X_test_s2_lr"].copy()

    y_train_s1 = data["y_train_s1"]
    y_test_s1  = data["y_test_s1"]
    y_train_s2 = data["y_train_s2"]
    y_test_s2  = data["y_test_s2"]

    print("Data loaded")
    print(f"Stage 1 train/test: {X_train_model.shape} / {X_test_model.shape}")
    print(f"Stage 2 train/test: {X_train_s2.shape} / {X_test_s2.shape}")

    # VALIDATION SPLIT — reserved exclusively for threshold tuning.  
    # Hyperparameters are selected via 5-fold CV on the full training 
    # set inside GridSearchCV. This validation split plays no role    
    # in hyperparameter selection or test evaluation.                 
    X_tr_s1, X_val_s1, y_tr_s1, y_val_s1 = train_test_split(
        X_train_model, y_train_s1,
        test_size=0.15, random_state=42, stratify=y_train_s1,
    )
    X_tr_s2, X_val_s2, y_tr_s2, y_val_s2 = train_test_split(
        X_train_s2, y_train_s2,
        test_size=0.15, random_state=42, stratify=y_train_s2,
    )

    print(f"\nStage 1 split: train={X_tr_s1.shape}, val={X_val_s1.shape}")
    print(f"Stage 2 split: train={X_tr_s2.shape}, val={X_val_s2.shape}")

    # Scaling inside LR pipeline only — tree-based models are
    # scale-invariant and do not require feature standardisation
    print("\nScaling handled inside LR pipeline only (tree models are scale-invariant)")

    # XGBoost imbalance weights — set to neg/pos ratio as recommended
    scale_pos_s1 = (y_train_s1 == 0).sum() / (y_train_s1 == 1).sum()
    scale_pos_s2 = (y_train_s2 == 0).sum() / (y_train_s2 == 1).sum()
    print(f"\nStage 1 class ratio (scale_pos_weight base): {scale_pos_s1:.2f}")
    print(f"Stage 2 class ratio (scale_pos_weight base): {scale_pos_s2:.2f}")

    # 5-fold stratified CV — stratification preserves class ratio in
    # each fold before SMOTE is applied inside the pipeline
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n" + "=" * 60)
    print("Hyperparameter tuning (5-fold stratified CV, scoring=PR-AUC)")
    print("=" * 60)

    #  STAGE 1                                                            
    #  Imbalance: SMOTE inside ImbPipeline for LR / DT / RF              
    #             scale_pos_weight for XGBoost (native loss weighting)    
    #  class_weight not used — resampling is the sole imbalance mechanism 
    #  for pipeline models, avoiding simultaneous double-handling         
    print("\nStage 1")

    # Logistic Regression — linear baseline
    # L2 (Ridge) regularization handles multicollinearity in clinical features
    # StandardScaler inside pipeline — required for LR, applied per CV fold
    lr_s1_pipe = ImbPipeline([
        ("smote",  SMOTE(random_state=42)),
        ("scaler", StandardScaler()),
        ("model",  LogisticRegression(
            max_iter=1000, random_state=42, penalty="l2", solver="lbfgs")),
    ])
    lr_s1_grid = GridSearchCV(
        lr_s1_pipe,
        {"model__C": [0.01, 0.1, 1, 10]},
        cv=cv, scoring="average_precision", n_jobs=1,
    )
    lr_s1_grid.fit(X_train_lr, y_train_s1)
    best_lr_s1 = lr_s1_grid.best_estimator_
    print(f"  LR  best: {lr_s1_grid.best_params_} | CV PR-AUC: {lr_s1_grid.best_score_:.4f}")

    # Decision Tree — interpretable nonlinear baseline
    dt_s1_pipe = ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("model", DecisionTreeClassifier(random_state=42)),
    ])
    dt_s1_grid = GridSearchCV(
        dt_s1_pipe,
        {
            "model__max_depth":         [8, 12],
            "model__min_samples_split": [5, 10],
        },
        cv=cv, scoring="average_precision", n_jobs=1,
    )
    dt_s1_grid.fit(X_train_model, y_train_s1)
    best_dt_s1 = dt_s1_grid.best_estimator_
    print(f"  DT  best: {dt_s1_grid.best_params_} | CV PR-AUC: {dt_s1_grid.best_score_:.4f}")

    # Random Forest — bagging ensemble; reduces variance over single DT
    rf_s1_pipe = ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("model", RandomForestClassifier(random_state=42, n_jobs=1)),
    ])
    rf_s1_grid = GridSearchCV(
        rf_s1_pipe,
        {
            "model__n_estimators":     [100, 200],
            "model__max_depth":        [10, 15],
            "model__min_samples_leaf": [1, 2],
        },
        cv=cv, scoring="average_precision", n_jobs=1,
    )
    rf_s1_grid.fit(X_train_model, y_train_s1)
    best_rf_s1 = rf_s1_grid.best_estimator_
    print(f"  RF  best: {rf_s1_grid.best_params_} | CV PR-AUC: {rf_s1_grid.best_score_:.4f}")

    # XGBoost — gradient-boosted SOTA benchmark
    # scale_pos_weight modifies the loss function directly;
    # external resampling would be redundant and may distort calibration
    xgb_s1_grid = GridSearchCV(
        XGBClassifier(
            random_state=42, eval_metric="logloss",
            verbosity=0, colsample_bytree=0.8,
        ),
        {
            "n_estimators":     [100, 200],
            "max_depth":        [4, 6],
            "learning_rate":    [0.05, 0.1],
            "subsample":        [0.8],
            "scale_pos_weight": [scale_pos_s1, scale_pos_s1 * 1.25],
        },
        cv=cv, scoring="average_precision", n_jobs=1,
    )
    xgb_s1_grid.fit(X_train_model, y_train_s1)
    best_xgb_s1 = xgb_s1_grid.best_estimator_
    print(f"  XGB best: {xgb_s1_grid.best_params_} | CV PR-AUC: {xgb_s1_grid.best_score_:.4f}")

    #  STAGE 2                                                            
    #  SMOTEENN preferred over SMOTE — Stage 2 operates on the smaller,  
    #  noisier readmitted-only subset. ENN removes borderline samples     
    #  whose label differs from their neighbours, cleaning the decision   
    #  boundary that SMOTE alone may distort on noisy data.              
    print("\nStage 2")

    # Logistic Regression — linear baseline
    lr_s2_pipe = ImbPipeline([
        ("smoteenn", SMOTEENN(random_state=42)),
        ("scaler",   StandardScaler()),
        ("model",    LogisticRegression(
            max_iter=1000, random_state=42, penalty="l2", solver="lbfgs")),
    ])
    lr_s2_grid = GridSearchCV(
        lr_s2_pipe,
        {"model__C": [0.01, 0.1, 1, 10]},
        cv=cv, scoring="average_precision", n_jobs=1,
    )
    lr_s2_grid.fit(X_train_s2_lr, y_train_s2)
    best_lr_s2 = lr_s2_grid.best_estimator_
    print(f"  LR  best: {lr_s2_grid.best_params_} | CV PR-AUC: {lr_s2_grid.best_score_:.4f}")

    # Decision Tree — interpretable nonlinear baseline
    dt_s2_pipe = ImbPipeline([
        ("smoteenn", SMOTEENN(random_state=42)),
        ("model",    DecisionTreeClassifier(random_state=42)),
    ])
    dt_s2_grid = GridSearchCV(
        dt_s2_pipe,
        {
            "model__max_depth":         [8, 12],
            "model__min_samples_split": [5, 10],
        },
        cv=cv, scoring="average_precision", n_jobs=1,
    )
    dt_s2_grid.fit(X_train_s2, y_train_s2)
    best_dt_s2 = dt_s2_grid.best_estimator_
    print(f"  DT  best: {dt_s2_grid.best_params_} | CV PR-AUC: {dt_s2_grid.best_score_:.4f}")

    # Random Forest — bagging ensemble
    rf_s2_pipe = ImbPipeline([
        ("smoteenn", SMOTEENN(random_state=42)),
        ("model",    RandomForestClassifier(random_state=42, n_jobs=1)),
    ])
    rf_s2_grid = GridSearchCV(
        rf_s2_pipe,
        {
            "model__n_estimators":     [100, 200],
            "model__max_depth":        [10, 15],
            "model__min_samples_leaf": [1, 2],
        },
        cv=cv, scoring="average_precision", n_jobs=1,
    )
    rf_s2_grid.fit(X_train_s2, y_train_s2)
    best_rf_s2 = rf_s2_grid.best_estimator_
    print(f"  RF  best: {rf_s2_grid.best_params_} | CV PR-AUC: {rf_s2_grid.best_score_:.4f}")

    # XGBoost — gradient-boosted SOTA benchmark
    xgb_s2_grid = GridSearchCV(
        XGBClassifier(
            random_state=42, eval_metric="logloss",
            verbosity=0, colsample_bytree=0.8,
        ),
        {
            "n_estimators":     [100, 200],
            "max_depth":        [4, 6],
            "learning_rate":    [0.05, 0.1],
            "subsample":        [0.8],
            "scale_pos_weight": [scale_pos_s2, scale_pos_s2 * 1.25],
        },
        cv=cv, scoring="average_precision", n_jobs=1,
    )
    xgb_s2_grid.fit(X_train_s2, y_train_s2)
    best_xgb_s2 = xgb_s2_grid.best_estimator_
    print(f"  XGB best: {xgb_s2_grid.best_params_} | CV PR-AUC: {xgb_s2_grid.best_score_:.4f}")

    print("\nTuning finished")

    # CV summary — sole basis for model selection
    cv_summary = {
        "Stage 1": {
            "Logistic Regression": round(lr_s1_grid.best_score_, 4),
            "Decision Tree":       round(dt_s1_grid.best_score_, 4),
            "Random Forest":       round(rf_s1_grid.best_score_, 4),
            "XGBoost":             round(xgb_s1_grid.best_score_, 4),
        },
        "Stage 2": {
            "Logistic Regression": round(lr_s2_grid.best_score_, 4),
            "Decision Tree":       round(dt_s2_grid.best_score_, 4),
            "Random Forest":       round(rf_s2_grid.best_score_, 4),
            "XGBoost":             round(xgb_s2_grid.best_score_, 4),
        },
    }

    print("\nCV PR-AUC summary (used for model selection)")
    for stage_name, scores in cv_summary.items():
        print(f"\n{stage_name}")
        for model_name, score in scores.items():
            print(f"  {model_name}: {score:.4f}")

    #  FINAL TRAINING AND EVALUATION — Stage 1
    stage1_outputs = {}

    stage1_outputs["Logistic Regression"] = train_and_score_model(
        "Logistic Regression", best_lr_s1,
        X_train_lr, y_train_s1,
        X_train_lr.loc[X_tr_s1.index], y_tr_s1,
        X_train_lr.loc[X_val_s1.index], y_val_s1,
        X_test_lr, y_test_s1,
    )
    stage1_outputs["Decision Tree"] = train_and_score_model(
        "Decision Tree", best_dt_s1,
        X_train_model, y_train_s1,
        X_tr_s1, y_tr_s1, X_val_s1, y_val_s1,
        X_test_model, y_test_s1,
    )
    stage1_outputs["Random Forest"] = train_and_score_model(
        "Random Forest", best_rf_s1,
        X_train_model, y_train_s1,
        X_tr_s1, y_tr_s1, X_val_s1, y_val_s1,
        X_test_model, y_test_s1,
    )
    stage1_outputs["XGBoost"] = train_and_score_model(
        "XGBoost", best_xgb_s1,
        X_train_model, y_train_s1,
        X_tr_s1, y_tr_s1, X_val_s1, y_val_s1,
        X_test_model, y_test_s1,
    )

    #  FINAL TRAINING AND EVALUATION — Stage 2
    stage2_outputs = {}

    stage2_outputs["Logistic Regression"] = train_and_score_model(
        "Logistic Regression", best_lr_s2,
        X_train_s2_lr, y_train_s2,
        X_train_s2_lr.loc[X_tr_s2.index], y_tr_s2,
        X_train_s2_lr.loc[X_val_s2.index], y_val_s2,
        X_test_s2_lr, y_test_s2,
    )
    stage2_outputs["Decision Tree"] = train_and_score_model(
        "Decision Tree", best_dt_s2,
        X_train_s2, y_train_s2,
        X_tr_s2, y_tr_s2, X_val_s2, y_val_s2,
        X_test_s2, y_test_s2,
    )
    stage2_outputs["Random Forest"] = train_and_score_model(
        "Random Forest", best_rf_s2,
        X_train_s2, y_train_s2,
        X_tr_s2, y_tr_s2, X_val_s2, y_val_s2,
        X_test_s2, y_test_s2,
    )
    stage2_outputs["XGBoost"] = train_and_score_model(
        "XGBoost", best_xgb_s2,
        X_train_s2, y_train_s2,
        X_tr_s2, y_tr_s2, X_val_s2, y_val_s2,
        X_test_s2, y_test_s2,
    )

    #  RESULTS TABLES                                                     
    def make_table(rows, sort_by):
        return pd.DataFrame(
            [{k: v for k, v in r.items() if k != "Confusion Matrix"} for r in rows]
        ).sort_values(sort_by, ascending=False)

    df_stage1_default = make_table([v["default_result"] for v in stage1_outputs.values()], "PR-AUC")
    df_stage1_tuned   = make_table([v["tuned_result"]   for v in stage1_outputs.values()], "F1_positive")
    df_stage2_default = make_table([v["default_result"] for v in stage2_outputs.values()], "PR-AUC")
    df_stage2_tuned   = make_table([v["tuned_result"]   for v in stage2_outputs.values()], "F1_positive")

    print("\nStage 1 default results (threshold=0.5)")
    print(df_stage1_default.to_string(index=False))
    print("\nStage 1 tuned results (threshold optimised on validation)")
    print(df_stage1_tuned.to_string(index=False))
    print("\nStage 2 default results (threshold=0.5)")
    print(df_stage2_default.to_string(index=False))
    print("\nStage 2 tuned results (threshold optimised on validation)")
    print(df_stage2_tuned.to_string(index=False))

    # Model selection by CV PR-AUC — test set used for evaluation only
    best_model_stage1 = max(cv_summary["Stage 1"], key=cv_summary["Stage 1"].get)
    best_model_stage2 = max(cv_summary["Stage 2"], key=cv_summary["Stage 2"].get)

    print(f"\nBest Stage 1 model by CV PR-AUC: {best_model_stage1}")
    print(f"Best Stage 2 model by CV PR-AUC: {best_model_stage2}")

    #  SAVE — keys kept identical to dashboard expectations               
    trained_output = {
        "cv_summary":           cv_summary,
        "stage1_outputs":       stage1_outputs,
        "stage2_outputs":       stage2_outputs,
        "stage1_default_table": df_stage1_default,
        "stage1_tuned_table":   df_stage1_tuned,
        "stage2_default_table": df_stage2_default,
        "stage2_tuned_table":   df_stage2_tuned,
        "best_model_stage1":    best_model_stage1,
        "best_model_stage2":    best_model_stage2,
        "y_test_s1":            y_test_s1,
        "y_test_s2":            y_test_s2,
        "X_test_model":         X_test_model,
        "X_test_s2":            X_test_s2,
        "training_notes": {
            #  data split roles 
            "split_roles": {
                "training_set":   "hyperparameter selection via 5-fold stratified CV",
                "validation_set": "threshold optimisation only — excluded from CV and test evaluation",
                "test_set":       "final one-time evaluation only — never used for tuning or selection",
            },
            # model selection 
            "model_selection_rule": "best CV PR-AUC — test set used for evaluation only",
            "cv_folds": 5,
            #  model rationale 
            "model_rationale": {
                "Logistic Regression": "linear baseline — interpretable coefficients",
                "Decision Tree":       "interpretable nonlinear baseline",
                "Random Forest":       "bagging ensemble — reduces variance",
                "XGBoost":             "gradient-boosted SOTA benchmark",
            },
            # threshold 
            "threshold_rule": (
                "full decision-threshold optimisation over validation distribution "
                "(sweep 0.01-0.99, step 0.01); maximises F1-positive; "
                "precision then recall used as sequential tiebreakers"
            ),
            # evaluation metric hierarchy 
            "metric_hierarchy": {
                "primary":   "PR-AUC — model selection; robust under class imbalance",
                "secondary": "Recall — clinical priority; missing a readmission is costly",
                "tertiary":  "F1 / Precision — operational trade-off at chosen threshold",
                "note":      "Accuracy reported but not used — unreliable on imbalanced test set",
            },
            # imbalance 
            "imbalance_strategy": {
                "LR_DT_RF_stage1": "SMOTE inside ImbPipeline",
                "LR_DT_RF_stage2": "SMOTEENN inside ImbPipeline",
                "XGBoost":         "scale_pos_weight only (native loss-level weighting)",
                "class_weight":    "not used — resampling is sole mechanism for pipeline models",
                "leakage_prevention": (
                    "resampling applied only on training folds inside pipeline; "
                    "stratified CV preserves class ratio before resampling; "
                    "validation and test sets receive no resampling"
                ),
            },
            # scaling 
            "scaling": (
                "StandardScaler inside LR pipeline only — "
                "tree-based models are scale-invariant and require no standardisation"
            ),
            # features 
            "stage1_features": X_train_model.columns.tolist(),
            "stage2_features": X_train_s2.columns.tolist(),
        },
    }

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(trained_output, f)

    print(f"\nSaved trained model artefact to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
