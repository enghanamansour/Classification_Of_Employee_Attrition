"""
Employee Attrition Classification - Training & Evaluation Pipeline
==================================================================
This script trains and benchmarks machine learning classification models
on the IBM HR Analytics Employee Attrition dataset.

Author: Antigravity Machine Learning Engineer
"""

import os
# Ensure thread constraints for OpenBLAS / OMP stability on Windows
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import sys
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, cross_val_predict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, balanced_accuracy_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
import xgboost as xgb

def load_and_clean_data(filepath='WA_Fn-UseC_-HR-Employee-Attrition.csv'):
    """Loads dataset, removes non-informative columns, and encodes target."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"[+] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Invariant and identifier columns that add no generalizable signal
    cols_to_drop = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
    existing_drops = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing_drops)
    print(f"[+] Dropped non-informative columns: {existing_drops}")
    
    # Map target: 'Yes' -> 1, 'No' -> 0
    if df['Attrition'].dtype == object:
        df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
        
    attrition_rate = df['Attrition'].mean()
    stay_count = (df['Attrition'] == 0).sum()
    leave_count = (df['Attrition'] == 1).sum()
    print(f"[+] Class Balance: Stay={stay_count} ({1-attrition_rate:.1%}), Leave={leave_count} ({attrition_rate:.1%})")
    
    return df

def build_preprocessor(X):
    """Creates a ColumnTransformer for numerical scaling and categorical one-hot encoding."""
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"[+] Preprocessing {len(num_cols)} numerical and {len(cat_cols)} categorical features")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols)
        ]
    )
    return preprocessor, num_cols, cat_cols

def train_and_evaluate(data_path='WA_Fn-UseC_-HR-Employee-Attrition.csv', output_dir='.'):
    """Main training workflow."""
    os.makedirs(output_dir, exist_ok=True)
    static_dir = os.path.join(output_dir, 'static')
    os.makedirs(static_dir, exist_ok=True)
    
    df = load_and_clean_data(data_path)
    X = df.drop(columns=['Attrition'])
    y = df['Attrition']
    
    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[+] Split: Train={len(X_train)} samples, Test={len(X_test)} samples")
    
    preprocessor, num_cols, cat_cols = build_preprocessor(X)
    
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    
    # Define candidate model architectures
    lr_clf = LogisticRegression(C=0.35, class_weight='balanced', max_iter=1000, random_state=42)
    rf_clf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_split=6, class_weight='balanced', random_state=42)
    xgb_clf = xgb.XGBClassifier(
        n_estimators=160, max_depth=3, learning_rate=0.05,
        scale_pos_weight=scale_pos, subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric='logloss', n_jobs=1
    )
    
    ensemble_clf = VotingClassifier(
        estimators=[('lr', lr_clf), ('rf', rf_clf), ('xgb', xgb_clf)],
        voting='soft',
        weights=[2.0, 1.0, 1.0]
    )
    
    candidate_models = {
        'Logistic Regression (Balanced)': lr_clf,
        'Random Forest (Balanced)': rf_clf,
        'XGBoost (Weighted)': xgb_clf,
        'Soft Voting Ensemble': ensemble_clf
    }
    
    print("\n" + "="*65)
    print("5-FOLD STRATIFIED CROSS-VALIDATION ON TRAINING DATA")
    print("="*65)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_summary = {}
    
    for name, clf in candidate_models.items():
        pipe = Pipeline([('prep', preprocessor), ('clf', clf)])
        scores = cross_validate(
            pipe, X_train, y_train, cv=cv,
            scoring=['roc_auc', 'average_precision', 'f1', 'recall', 'precision', 'accuracy'],
            n_jobs=1
        )
        cv_summary[name] = {
            'roc_auc_mean': float(scores['test_roc_auc'].mean()),
            'roc_auc_std': float(scores['test_roc_auc'].std()),
            'pr_auc_mean': float(scores['test_average_precision'].mean()),
            'pr_auc_std': float(scores['test_average_precision'].std()),
            'f1_mean': float(scores['test_f1'].mean()),
            'recall_mean': float(scores['test_recall'].mean()),
            'precision_mean': float(scores['test_precision'].mean()),
            'accuracy_mean': float(scores['test_accuracy'].mean()),
        }
        print(f"[{name}]")
        print(f"  ROC-AUC:  {scores['test_roc_auc'].mean():.4f} +/- {scores['test_roc_auc'].std():.4f}")
        print(f"  PR-AUC:   {scores['test_average_precision'].mean():.4f} +/- {scores['test_average_precision'].std():.4f}")
        print(f"  Recall:   {scores['test_recall'].mean():.4f} | Precision: {scores['test_precision'].mean():.4f} | F1: {scores['test_f1'].mean():.4f}")
    
    print("\n" + "="*65)
    print("HOLDOUT TEST SET EVALUATION (294 SAMPLES)")
    print("="*65)
    
    test_results = {}
    test_probs = {}
    test_preds_default = {}
    trained_pipes = {}
    
    for name, clf in candidate_models.items():
        pipe = Pipeline([('prep', preprocessor), ('clf', clf)])
        pipe.fit(X_train, y_train)
        trained_pipes[name] = pipe
        
        prob = pipe.predict_proba(X_test)[:, 1]
        pred_def = pipe.predict(X_test)
        
        test_probs[name] = prob
        test_preds_default[name] = pred_def
        
        auc = roc_auc_score(y_test, prob)
        pr_auc = average_precision_score(y_test, prob)
        acc = accuracy_score(y_test, pred_def)
        rec = recall_score(y_test, pred_def)
        prec = precision_score(y_test, pred_def, zero_division=0)
        f1 = f1_score(y_test, pred_def)
        bal_acc = balanced_accuracy_score(y_test, pred_def)
        cm = confusion_matrix(y_test, pred_def).tolist()
        
        test_results[name] = {
            'roc_auc': float(auc),
            'pr_auc': float(pr_auc),
            'accuracy': float(acc),
            'balanced_accuracy': float(bal_acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1),
            'confusion_matrix': cm
        }
        
        print(f"\nModel: {name}")
        print(f"  Accuracy: {acc:.4f} | Balanced Acc: {bal_acc:.4f}")
        print(f"  ROC-AUC:  {auc:.4f} | PR-AUC (Average Precision): {pr_auc:.4f}")
        print(f"  Precision:{prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
        print(f"  Confusion Matrix (TN, FP / FN, TP):\n  {cm}")
        
    # Primary Production Model is the Soft Voting Ensemble (or Logistic Regression)
    best_model_name = 'Soft Voting Ensemble'
    best_pipe = trained_pipes[best_model_name]
    
    # Optimal Threshold Selection based on CV Precision-Recall trade-off
    cv_prob_best = cross_val_predict(best_pipe, X_train, y_train, cv=5, method='predict_proba')[:, 1]
    prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_train, cv_prob_best)
    f1_curve = 2 * (prec_arr * rec_arr) / (prec_arr + rec_arr + 1e-10)
    best_thresh_idx = np.argmax(f1_curve)
    optimal_threshold = float(thresh_arr[best_thresh_idx]) if best_thresh_idx < len(thresh_arr) else 0.40
    # Bound threshold to reasonable business range [0.30, 0.50]
    optimal_threshold = float(np.clip(optimal_threshold, 0.30, 0.50))
    print(f"\n[+] Recommended Decision Threshold for Attrition Risk Alert: {optimal_threshold:.2f}")
    
    opt_pred = (test_probs[best_model_name] >= optimal_threshold).astype(int)
    opt_cm = confusion_matrix(y_test, opt_pred).tolist()
    opt_rec = recall_score(y_test, opt_pred)
    opt_prec = precision_score(y_test, opt_pred)
    opt_f1 = f1_score(y_test, opt_pred)
    opt_acc = accuracy_score(y_test, opt_pred)
    
    print(f"[+] Best Model at Tuned Threshold ({optimal_threshold:.2f}):")
    print(f"    Recall: {opt_rec:.2%} (Catches {opt_cm[1][1]} out of {sum(y_test==1)} at-risk employees)")
    print(f"    Precision: {opt_prec:.2%}")
    print(f"    F1-Score: {opt_f1:.4f} | Accuracy: {opt_acc:.2%}")

    # Extract Feature Importances from trained components
    preprocessor_fitted = best_pipe.named_steps['prep']
    cat_feature_names = preprocessor_fitted.named_transformers_['cat'].get_feature_names_out(cat_cols).tolist()
    all_feature_names = num_cols + cat_feature_names
    
    lr_fitted = trained_pipes['Logistic Regression (Balanced)'].named_steps['clf']
    rf_fitted = trained_pipes['Random Forest (Balanced)'].named_steps['clf']
    
    lr_coeffs = pd.Series(lr_fitted.coef_[0], index=all_feature_names)
    rf_importances = pd.Series(rf_fitted.feature_importances_, index=all_feature_names)
    
    top_risk_drivers = lr_coeffs.sort_values(ascending=False).head(10).to_dict()
    top_protective_factors = lr_coeffs.sort_values(ascending=True).head(10).to_dict()
    top_rf_features = rf_importances.sort_values(ascending=False).head(10).to_dict()

    # Generate Evaluation Diagnostic Charts
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.25)
    
    # 1. ROC Curves
    ax_roc = axes[0, 0]
    for m_name in candidate_models:
        fpr, tpr, _ = roc_curve(y_test, test_probs[m_name])
        auc_val = test_results[m_name]['roc_auc']
        ax_roc.plot(fpr, tpr, lw=2, label=f"{m_name} (AUC={auc_val:.3f})")
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6, label='Random Chance')
    ax_roc.set_title("ROC Curves (Hold-out Test Set)", fontsize=13, fontweight='bold', pad=10)
    ax_roc.set_xlabel("False Positive Rate", fontsize=11)
    ax_roc.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax_roc.legend(loc="lower right", fontsize=9)
    ax_roc.grid(True, linestyle=':', alpha=0.6)
    
    # 2. Precision-Recall Curves
    ax_pr = axes[0, 1]
    for m_name in candidate_models:
        precision_c, recall_c, _ = precision_recall_curve(y_test, test_probs[m_name])
        pr_auc_val = test_results[m_name]['pr_auc']
        ax_pr.plot(recall_c, precision_c, lw=2, label=f"{m_name} (PR-AUC={pr_auc_val:.3f})")
    baseline_pr = sum(y_test == 1) / len(y_test)
    ax_pr.axhline(baseline_pr, color='k', linestyle='--', lw=1.5, alpha=0.6, label=f'Baseline ({baseline_pr:.2%})')
    ax_pr.set_title("Precision-Recall Curves (Hold-out Test Set)", fontsize=13, fontweight='bold', pad=10)
    ax_pr.set_xlabel("Recall", fontsize=11)
    ax_pr.set_ylabel("Precision", fontsize=11)
    ax_pr.legend(loc="upper right", fontsize=9)
    ax_pr.grid(True, linestyle=':', alpha=0.6)
    
    # 3. Confusion Matrix of Production Model (at optimal threshold)
    ax_cm = axes[1, 0]
    cm_arr = np.array(opt_cm)
    im = ax_cm.imshow(cm_arr, interpolation='nearest', cmap=plt.cm.Blues)
    ax_cm.figure.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    ax_cm.set(
        xticks=np.arange(2), yticks=np.arange(2),
        xticklabels=['Pred: Stay (0)', 'Pred: Leave (1)'],
        yticklabels=['Actual: Stay (0)', 'Actual: Leave (1)'],
        title=f"Confusion Matrix ({best_model_name}\nThreshold = {optimal_threshold:.2f})"
    )
    ax_cm.title.set_fontsize(13)
    ax_cm.title.set_fontweight('bold')
    thresh = cm_arr.max() / 2.
    for i in range(2):
        for j in range(2):
            ax_cm.text(j, i, f"{cm_arr[i, j]}\n({cm_arr[i, j]/len(y_test):.1%})",
                       ha="center", va="center",
                       color="white" if cm_arr[i, j] > thresh else "black",
                       fontsize=11, fontweight='bold')
                       
    # 4. Top Feature Importances (Random Forest)
    ax_imp = axes[1, 1]
    top10_imp = rf_importances.sort_values(ascending=True).tail(10)
    # clean up feature names for display
    clean_labels = [name.replace('_', ' ') for name in top10_imp.index]
    y_pos = np.arange(len(top10_imp))
    ax_imp.barh(y_pos, top10_imp.values, color='#3b82f6', align='center', alpha=0.85)
    ax_imp.set_yticks(y_pos)
    ax_imp.set_yticklabels(clean_labels, fontsize=9.5)
    ax_imp.set_xlabel("Relative Importance Score", fontsize=11)
    ax_imp.set_title("Top 10 Feature Importances (Random Forest)", fontsize=13, fontweight='bold', pad=10)
    ax_imp.grid(True, axis='x', linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plots_path = os.path.join(output_dir, 'evaluation_plots.png')
    static_plots_path = os.path.join(static_dir, 'evaluation_plots.png')
    fig.savefig(plots_path, dpi=200, bbox_inches='tight')
    fig.savefig(static_plots_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved diagnostic plots to {plots_path} and {static_plots_path}")
    
    # Save the trained model pipeline
    model_save_path = os.path.join(output_dir, 'attrition_model.joblib')
    joblib.dump({
        'pipeline': best_pipe,
        'model_name': best_model_name,
        'optimal_threshold': optimal_threshold,
        'feature_names': all_feature_names,
        'num_cols': num_cols,
        'cat_cols': cat_cols
    }, model_save_path)
    print(f"[+] Saved trained model artifact to {model_save_path}")
    
    # Save model metadata and report
    metadata = {
        'model_name': best_model_name,
        'optimal_threshold': optimal_threshold,
        'dataset_summary': {
            'total_samples': int(len(df)),
            'train_samples': int(len(X_train)),
            'test_samples': int(len(X_test)),
            'total_features': len(X.columns),
            'preprocessed_features': len(all_feature_names),
            'attrition_rate': float(df['Attrition'].mean())
        },
        'cross_validation_5fold': cv_summary,
        'test_set_performance': test_results,
        'tuned_threshold_metrics': {
            'threshold': optimal_threshold,
            'accuracy': float(opt_acc),
            'precision': float(opt_prec),
            'recall': float(opt_rec),
            'f1_score': float(opt_f1),
            'confusion_matrix': opt_cm
        },
        'key_risk_drivers': {k: float(v) for k, v in top_risk_drivers.items()},
        'key_protective_factors': {k: float(v) for k, v in top_protective_factors.items()},
        'top_random_forest_features': {k: float(v) for k, v in top_rf_features.items()},
        'numerical_features': num_cols,
        'categorical_features': cat_cols
    }
    
    meta_path = os.path.join(output_dir, 'model_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"[+] Saved detailed model metadata to {meta_path}")
    
    print("\n" + "="*65)
    print("TRAINING AND BENCHMARKING COMPLETED SUCCESSFULLY!")
    print("="*65)
    return metadata

if __name__ == '__main__':
    train_and_evaluate()
