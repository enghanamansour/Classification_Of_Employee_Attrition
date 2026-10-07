import os
# Ensure thread constraints for stability on Windows
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, precision_score, recall_score,
    accuracy_score, classification_report, confusion_matrix
)
import xgboost as xgb

# 1. Load data
data_path = 'WA_Fn-UseC_-HR-Employee-Attrition.csv'
df = pd.read_csv(data_path)

drop_cols = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Target encoding
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
X = df.drop(columns=['Attrition'])
y = df['Attrition']

cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"Dataset shape: {df.shape}")
print(f"Class distribution: 0={sum(y==0)}, 1={sum(y==1)} ({y.mean():.2%})")
print(f"Numerical features ({len(num_cols)}): {num_cols}")
print(f"Categorical features ({len(cat_cols)}): {cat_cols}")

# Train/Test Split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols)
    ]
)

scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, max_depth=8, min_samples_split=5),
    'HistGradientBoosting': HistGradientBoostingClassifier(class_weight='balanced', random_state=42, max_iter=150, min_samples_leaf=15),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        n_jobs=1
    )
}

print("\n--- 5-Fold Stratified Cross-Validation on Training Set ---")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, clf in models.items():
    pipe = Pipeline([('prep', preprocessor), ('clf', clf)])
    scores = cross_validate(
        pipe, X_train, y_train, cv=cv,
        scoring=['roc_auc', 'average_precision', 'f1', 'recall', 'precision'],
        n_jobs=1
    )
    print(f"\n{name}:")
    print(f"  ROC-AUC:  {scores['test_roc_auc'].mean():.4f} (+/- {scores['test_roc_auc'].std():.4f})")
    print(f"  PR-AUC:   {scores['test_average_precision'].mean():.4f} (+/- {scores['test_average_precision'].std():.4f})")
    print(f"  F1-Score: {scores['test_f1'].mean():.4f} (+/- {scores['test_f1'].std():.4f})")
    print(f"  Recall:   {scores['test_recall'].mean():.4f} (+/- {scores['test_recall'].std():.4f})")
    print(f"  Precision:{scores['test_precision'].mean():.4f} (+/- {scores['test_precision'].std():.4f})")

# Evaluate each model on test set
print("\n" + "="*50)
print("--- Hold-out Test Set Evaluation (294 employees) ---")
print("="*50)

results = []
trained_pipelines = {}

for name, clf in models.items():
    pipe = Pipeline([('prep', preprocessor), ('clf', clf)])
    pipe.fit(X_train, y_train)
    trained_pipelines[name] = pipe
    
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'ROC-AUC': auc,
        'PR-AUC': pr_auc,
        'F1': f1,
        'Recall': rec,
        'Precision': prec,
        'Accuracy': acc
    })
    
    print(f"\nModel: {name}")
    print(f"Accuracy: {acc:.4f} | ROC-AUC: {auc:.4f} | PR-AUC: {pr_auc:.4f}")
    print(f"Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

res_df = pd.DataFrame(results).sort_values(by='ROC-AUC', ascending=False)
print("\n" + "="*50)
print("Summary Comparison on Test Set:")
print(res_df.to_string(index=False))
