"""
Employee Attrition Classification - Inference & Risk Scoring Engine
====================================================================
This script performs inference on single employee profiles or batch CSV files
using the trained employee attrition classification model.

Usage Examples:
  # Run built-in sample predictions:
  python predict.py --sample

  # Predict on a CSV dataset:
  python predict.py --input WA_Fn-UseC_-HR-Employee-Attrition.csv --output attrition_predictions.csv

  # Run custom interactive prediction:
  python predict.py --interactive
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import sys
import argparse
import joblib
import json
import pandas as pd
import numpy as np

MODEL_FILE = 'attrition_model.joblib'
METADATA_FILE = 'model_metadata.json'

def load_model(model_path=MODEL_FILE):
    """Loads the trained model pipeline and configuration."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model file '{model_path}' not found. Please run 'python train.py' first."
        )
    model_data = joblib.load(model_path)
    return model_data

def get_risk_tier(prob):
    """Categorizes attrition probability into actionable HR risk tiers."""
    if prob < 0.20:
        return "Low Risk", "#10b981", "Employee shows strong retention indicators. Standard engagement."
    elif prob < 0.40:
        return "Moderate Risk", "#f59e0b", "Monitor engagement, check-in on workload and growth path."
    elif prob < 0.65:
        return "High Risk", "#f97316", "Active attrition warning. Immediate 1-on-1 manager review recommended."
    else:
        return "Critical Risk", "#ef4444", "Urgent intervention needed. High probability of departure."

def explain_employee_risk(employee_row, model_meta=None):
    """Identifies the key contributing risk factors for a specific employee."""
    factors = []
    
    # Overtime
    if str(employee_row.get('OverTime', '')).strip().lower() in ['yes', 'y']:
        factors.append("Working frequent Overtime (strongest single driver of burnout)")
        
    # Travel
    if 'Frequently' in str(employee_row.get('BusinessTravel', '')):
        factors.append("High travel frequency (frequent business travel increases attrition)")
        
    # Compensation vs Level
    income = float(employee_row.get('MonthlyIncome', 5000))
    job_level = int(employee_row.get('JobLevel', 2))
    if income < 3000:
        factors.append(f"Low monthly salary (${income:,.0f}/mo)")
    elif job_level >= 2 and income < 4000:
        factors.append(f"Below-market compensation for Job Level {job_level} (${income:,.0f}/mo)")
        
    # Stock Options
    stock = int(employee_row.get('StockOptionLevel', 0))
    if stock == 0:
        factors.append("Zero stock options (lower long-term equity attachment)")
        
    # Commute
    dist = float(employee_row.get('DistanceFromHome', 5))
    if dist >= 15:
        factors.append(f"Long commute distance ({dist:.0f} miles from home)")
        
    # Satisfaction scores
    job_sat = int(employee_row.get('JobSatisfaction', 3))
    env_sat = int(employee_row.get('EnvironmentSatisfaction', 3))
    work_life = int(employee_row.get('WorkLifeBalance', 3))
    
    if job_sat <= 2:
        factors.append(f"Low Job Satisfaction rating ({job_sat}/4)")
    if env_sat <= 2:
        factors.append(f"Low Environment Satisfaction rating ({env_sat}/4)")
    if work_life <= 2:
        factors.append(f"Poor Work-Life Balance rating ({work_life}/4)")
        
    # Management & Tenure
    years_curr_mgr = float(employee_row.get('YearsWithCurrManager', 3))
    years_since_promo = float(employee_row.get('YearsSinceLastPromotion', 1))
    years_at_co = float(employee_row.get('YearsAtCompany', 3))
    
    if years_since_promo >= 5 and years_at_co >= 5:
        factors.append(f"Promotion stagnation ({years_since_promo:.0f} years without promotion)")
    if years_curr_mgr < 1:
        factors.append("New direct manager (<1 year working relationship)")
    if str(employee_row.get('MaritalStatus', '')).strip().lower() == 'single':
        factors.append("Single marital status (statistically higher job mobility)")
        
    if not factors:
        factors.append("Balanced employee profile with positive retention indicators.")
        
    return factors

def predict_single(employee_dict, model_data=None):
    """Predicts attrition risk and provides reasoning for a single employee record."""
    if model_data is None:
        model_data = load_model()
        
    pipeline = model_data['pipeline']
    thresh = model_data.get('optimal_threshold', 0.50)
    
    # Filter or fill missing columns
    df_row = pd.DataFrame([employee_dict])
    
    drop_cols = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours', 'Attrition']
    for c in drop_cols:
        if c in df_row.columns:
            df_row = df_row.drop(columns=[c])
            
    prob = float(pipeline.predict_proba(df_row)[0, 1])
    pred = int(prob >= thresh)
    tier, tier_color, action_rec = get_risk_tier(prob)
    factors = explain_employee_risk(employee_dict)
    
    return {
        'attrition_prediction': 'Yes' if pred == 1 else 'No',
        'attrition_probability': round(prob * 100, 2),
        'risk_tier': tier,
        'tier_color': tier_color,
        'decision_threshold': thresh,
        'recommended_action': action_rec,
        'key_risk_factors': factors
    }

def predict_batch(input_csv, output_csv, model_data=None):
    """Predicts attrition for a full CSV dataset and writes enriched results."""
    if model_data is None:
        model_data = load_model()
        
    pipeline = model_data['pipeline']
    thresh = model_data.get('optimal_threshold', 0.50)
    
    print(f"[+] Loading input file: {input_csv}")
    df = pd.read_csv(input_csv)
    
    # Store ID if present
    id_col = None
    for cand in ['EmployeeNumber', 'EmployeeID', 'ID', 'Id']:
        if cand in df.columns:
            id_col = df[cand]
            break
            
    # Prepare features
    drop_cols = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours', 'Attrition']
    features_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    print(f"[+] Running inference on {len(features_df)} employees...")
    probs = pipeline.predict_proba(features_df)[:, 1]
    preds = (probs >= thresh).astype(int)
    
    df_out = df.copy()
    df_out['Predicted_Attrition'] = np.where(preds == 1, 'Yes', 'No')
    df_out['Attrition_Probability_Pct'] = (probs * 100).round(2)
    df_out['Risk_Tier'] = [get_risk_tier(p)[0] for p in probs]
    df_out['Recommended_Action'] = [get_risk_tier(p)[2] for p in probs]
    
    df_out.to_csv(output_csv, index=False)
    print(f"[+] Saved predictions to: {output_csv}")
    
    # Summary
    at_risk = (preds == 1).sum()
    print(f"\n[+] Batch Inference Summary:")
    print(f"    Total Evaluated: {len(df)}")
    print(f"    Flagged At-Risk: {at_risk} ({at_risk/len(df):.1%})")
    print(f"    Risk Breakdown:")
    print(df_out['Risk_Tier'].value_counts().to_string())
    return df_out

def run_sample_demos():
    """Runs demonstration predictions for archetypal employee profiles."""
    model_data = load_model()
    
    # Sample 1: High Risk Profile (Young, Overtime, Single, Low Income, High Travel)
    high_risk_employee = {
        'Age': 24,
        'BusinessTravel': 'Travel_Frequently',
        'DailyRate': 400,
        'Department': 'Sales',
        'DistanceFromHome': 22,
        'Education': 2,
        'EducationField': 'Marketing',
        'EnvironmentSatisfaction': 1,
        'Gender': 'Female',
        'HourlyRate': 45,
        'JobInvolvement': 1,
        'JobLevel': 1,
        'JobRole': 'Sales Representative',
        'JobSatisfaction': 1,
        'MaritalStatus': 'Single',
        'MonthlyIncome': 2100,
        'MonthlyRate': 12000,
        'NumCompaniesWorked': 3,
        'OverTime': 'Yes',
        'PercentSalaryHike': 11,
        'PerformanceRating': 3,
        'RelationshipSatisfaction': 2,
        'StockOptionLevel': 0,
        'TotalWorkingYears': 2,
        'TrainingTimesLastYear': 2,
        'WorkLifeBalance': 1,
        'YearsAtCompany': 1,
        'YearsInCurrentRole': 0,
        'YearsSinceLastPromotion': 0,
        'YearsWithCurrManager': 0
    }
    
    # Sample 2: Low Risk Profile (Senior, Stable, Good Pay, High Stock, No Overtime)
    low_risk_employee = {
        'Age': 48,
        'BusinessTravel': 'Non-Travel',
        'DailyRate': 1200,
        'Department': 'Research & Development',
        'DistanceFromHome': 2,
        'Education': 4,
        'EducationField': 'Medical',
        'EnvironmentSatisfaction': 4,
        'Gender': 'Male',
        'HourlyRate': 85,
        'JobInvolvement': 3,
        'JobLevel': 4,
        'JobRole': 'Research Director',
        'JobSatisfaction': 4,
        'MaritalStatus': 'Married',
        'MonthlyIncome': 16500,
        'MonthlyRate': 22000,
        'NumCompaniesWorked': 1,
        'OverTime': 'No',
        'PercentSalaryHike': 18,
        'PerformanceRating': 3,
        'RelationshipSatisfaction': 4,
        'StockOptionLevel': 2,
        'TotalWorkingYears': 24,
        'TrainingTimesLastYear': 3,
        'WorkLifeBalance': 4,
        'YearsAtCompany': 18,
        'YearsInCurrentRole': 12,
        'YearsSinceLastPromotion': 1,
        'YearsWithCurrManager': 10
    }
    
    print("\n" + "="*60)
    print("DEMO CASE 1: JUNIOR EMPLOYEE WITH BURNOUT SIGNALS")
    print("="*60)
    res1 = predict_single(high_risk_employee, model_data)
    print(f"Prediction:            {res1['attrition_prediction']} (Risk Score: {res1['attrition_probability']}%)")
    print(f"Risk Tier:             {res1['risk_tier']}")
    print(f"Action Recommendation: {res1['recommended_action']}")
    print("Key Drivers:")
    for f in res1['key_risk_factors']:
        print(f"  * {f}")
        
    print("\n" + "="*60)
    print("DEMO CASE 2: SENIOR LEADER WITH HIGH RETENTION SIGNALS")
    print("="*60)
    res2 = predict_single(low_risk_employee, model_data)
    print(f"Prediction:            {res2['attrition_prediction']} (Risk Score: {res2['attrition_probability']}%)")
    print(f"Risk Tier:             {res2['risk_tier']}")
    print(f"Action Recommendation: {res2['recommended_action']}")
    print("Key Drivers:")
    for f in res2['key_risk_factors']:
        print(f"  * {f}")
    print("="*60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Employee Attrition Classification Predictor")
    parser.add_argument('--sample', action='store_true', help="Run demonstration profiles")
    parser.add_argument('--input', type=str, help="Path to input CSV for batch prediction")
    parser.add_argument('--output', type=str, default='attrition_predictions.csv', help="Path to output CSV")
    
    args = parser.parse_args()
    
    if args.input:
        predict_batch(args.input, args.output)
    else:
        run_sample_demos()
