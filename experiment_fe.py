import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, precision_score, recall_score,
    accuracy_score, precision_recall_curve, confusion_matrix, classification_report
)
import xgboost as xgb

def engineer_features(data):
    df = data.copy()
    
    # Ratios and Interactions
    df['TenureRatio'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1)
    df['RoleTenureRatio'] = df['YearsInCurrentRole'] / (df['YearsAtCompany'] + 1)
    df['PromotionLagRatio'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)
    df['ManagerStabilityRatio'] = df['YearsWithCurrManager'] / (df['YearsAtCompany'] + 1)
    df['IncomePerWorkYear'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] + 1)
    
    # Combined Satisfaction index
    df['OverallSatisfaction'] = (
        df['EnvironmentSatisfaction'] +
        df['JobSatisfaction'] +
        df['RelationshipSatisfaction'] +
        df['WorkLifeBalance']
    ) / 4.0
    
    # Interaction flags
    df['YoungAndSingle'] = ((df['Age'] < 30) & (df['MaritalStatus'] == 'Single')).astype(int)
    df['OverTime_Single'] = ((df['OverTime'] == 'Yes') & (df['MaritalStatus'] == 'Single')).astype(int)
    df['FarAndLowPay'] = ((df['DistanceFromHome'] > 15) & (df['MonthlyIncome'] < 3000)).astype(int)
    
    return df

raw_df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
drop_cols = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
raw_df = raw_df.drop(columns=[c for c in drop_cols if c in raw_df.columns])
raw_df['Attrition'] = raw_df['Attrition'].map({'Yes': 1, 'No': 0})

df_feat = engineer_features(raw_df)
X = df_feat.drop(columns=['Attrition'])
y = df_feat['Attrition']

cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64', 'int32']).columns.tolist()

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
    'Logistic Regression': LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42),
    'Random Forest (Balanced)': RandomForestClassifier(n_estimators=300, max_depth=7, min_samples_split=4, class_weight='balanced', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=180, learning_rate=0.06, max_depth=3, subsample=0.8, random_state=42),
    'XGBoost (Weighted)': xgb.XGBClassifier(
        n_estimators=160, max_depth=3, learning_rate=0.05,
        scale_pos_weight=scale_pos, subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric='logloss', n_jobs=1
    )
}

print("="*60)
print("EVALUATING MODELS WITH FEATURE ENGINEERING & THRESHOLD OPTIMIZATION")
print("="*60)

for name, clf in models.items():
    pipe = Pipeline([('prep', preprocessor), ('clf', clf)])
    pipe.fit(X_train, y_train)
    
    # 5-fold CV probabilities on training set to find optimal threshold
    cv_prob = cross_val_predict(pipe, X_train, y_train, cv=5, method='predict_proba')[:, 1]
    cv_auc = roc_auc_score(y_train, cv_prob)
    cv_pr_auc = average_precision_score(y_train, cv_prob)
    
    # Find best threshold for F1 on CV
    precisions, recalls, thresholds = precision_recall_curve(y_train, cv_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_cv_f1 = f1_scores[best_idx]
    
    # Test set evaluation
    test_prob = pipe.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_prob)
    test_pr_auc = average_precision_score(y_test, test_prob)
    
    # Metrics with default threshold (0.5)
    pred_def = (test_prob >= 0.5).astype(int)
    f1_def = f1_score(y_test, pred_def)
    rec_def = recall_score(y_test, pred_def)
    prec_def = precision_score(y_test, pred_def)
    acc_def = accuracy_score(y_test, pred_def)
    
    # Metrics with optimal threshold
    pred_opt = (test_prob >= best_thresh).astype(int)
    f1_opt = f1_score(y_test, pred_opt)
    rec_opt = recall_score(y_test, pred_opt)
    prec_opt = precision_score(y_test, pred_opt)
    acc_opt = accuracy_score(y_test, pred_opt)
    
    print(f"\n--- {name} ---")
    print(f"CV ROC-AUC: {cv_auc:.4f} | CV PR-AUC: {cv_pr_auc:.4f} | Optimal Thresh: {best_thresh:.3f} (CV F1: {best_cv_f1:.4f})")
    print(f"Test ROC-AUC: {test_auc:.4f} | Test PR-AUC: {test_pr_auc:.4f}")
    print(f"Default Thresh (0.50): Acc={acc_def:.4f}, Prec={prec_def:.4f}, Rec={rec_def:.4f}, F1={f1_def:.4f}")
    print(f"Optimal Thresh ({best_thresh:.2f}): Acc={acc_opt:.4f}, Prec={prec_opt:.4f}, Rec={rec_opt:.4f}, F1={f1_opt:.4f}")
    print("Confusion Matrix (Optimal):\n", confusion_matrix(y_test, pred_opt))
