# Employee Attrition Classification System

An end-to-end Machine Learning classification and risk diagnostic platform built on the IBM HR Analytics Employee Attrition dataset (1,470 employee records, 35 raw features).

---

## 🎓 الإشارة إلى البرنامج التدريبي (Training Program Acknowledgement)

تم تنفيذ وتطوير هذا المشروع ضمن متطلبات **برنامج أساسيات تعلم الآلة** المقدم من **أكاديمية سدايا (SDAIA Academy)** للهيئة السعودية للبيانات والذكاء الاصطناعي، وذلك بهدف تطبيق أفضل الممارسات في مجال تعلم الآلة وبناء نماذج التصنيف.

This project was developed as part of the **Fundamentals of Machine Learning** program offered by **SDAIA Academy**.

| | |
| :--- | :--- |
| **البرنامج / Program** | أساسيات تعلم الآلة (Fundamentals of Machine Learning) |
| **المدربة / Instructor** | حنين المعيوف (Haneen Almaiouf) |
| **فريق العمل / Team** | هناء الشمراني (Hanaa Alshamrani) · رشا السالمي (Rasha Alsalmi) · سارة عسيري (Sarah Asiri) |

---

## 📊 1. Dataset & Problem Summary

- **Objective**: Predict whether an employee will leave the organization (`Attrition` = `'Yes'` vs `'No'`).
- **Class Balance**: 
  - `No` (Stay): **1,233** (83.88%)
  - `Yes` (Attrition): **237** (16.12%)
  - *Class Imbalance Ratio*: ~5.2 : 1
- **Feature Overview**:
  - **7 Categorical Features**: `BusinessTravel`, `Department`, `EducationField`, `Gender`, `JobRole`, `MaritalStatus`, `OverTime`.
  - **23 Numerical & Rating Features**: `Age`, `DailyRate`, `DistanceFromHome`, `Education`, `EnvironmentSatisfaction`, `HourlyRate`, `JobInvolvement`, `JobLevel`, `JobSatisfaction`, `MonthlyIncome`, `MonthlyRate`, `NumCompaniesWorked`, `PercentSalaryHike`, `PerformanceRating`, `RelationshipSatisfaction`, `StockOptionLevel`, `TotalWorkingYears`, `TrainingTimesLastYear`, `WorkLifeBalance`, `YearsAtCompany`, `YearsInCurrentRole`, `YearsSinceLastPromotion`, `YearsWithCurrManager`.
  - **Excluded Invariant / ID Columns**: `EmployeeCount` (constant 1), `StandardHours` (constant 80), `Over18` (constant 'Y'), `EmployeeNumber` (identifier).

---

## 🏆 2. Model Benchmarks & Comparison

Evaluated using **5-Fold Stratified Cross-Validation** on training data (1,176 samples) and validated on a held-out test set (294 samples, 20% stratified split):

| Model Architecture | 5-Fold CV ROC-AUC | 5-Fold CV PR-AUC | Test ROC-AUC | Test PR-AUC | Attrition Recall | Test Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Soft Voting Ensemble (Champion)** | **0.8312 ± 0.030** | **0.6494 ± 0.071** | **0.8075** | **0.5763** | **53.2% - 67.4%** | **80.95%** |
| **Logistic Regression (Balanced)** | 0.8276 ± 0.030 | 0.6103 ± 0.083 | 0.8055 | 0.5802 | 68.09% | 76.19% |
| **XGBoost (Weighted)** | 0.8043 ± 0.028 | 0.6067 ± 0.062 | 0.7735 | 0.4994 | 51.06% | 80.27% |
| **Random Forest (Balanced)** | 0.7977 ± 0.030 | 0.5499 ± 0.075 | 0.7876 | 0.4158 | 26.84% | 81.97% |

> **Key takeaway**: Because the dataset is imbalanced (16.1% positive class), **ROC-AUC** and **PR-AUC (Average Precision)** are the primary ranking metrics. The **Soft Voting Ensemble** combines the calibrated linear ranking of Logistic Regression with the interaction-capture capabilities of Random Forest and XGBoost.

---

## 🔍 3. Key Drivers of Employee Attrition

### ⚠️ Top Risk Factors (Increasing Attrition Risk)
1. **OverTime (Yes)** (`+1.59` log-odds): The single strongest driver of burnout and employee departure.
2. **BusinessTravel (Frequently)** (`+1.25` log-odds): High travel load creates strain on work-life balance.
3. **MaritalStatus (Single)** (`+0.98` log-odds): Statistically higher geographical and career mobility.
4. **JobRole (Laboratory Technician / Sales Representative)** (`+0.71` to `+0.88` log-odds): High turnover in entry/stress-heavy roles.
5. **Promotion Stagnation** (`+0.48` log-odds): Employees with 5+ years without promotion exhibit elevated flight risk.
6. **Commute Distance** (>15 miles from home): Commuting strain compounds daily burnout.

### 🛡️ Top Protective Factors (Decreasing Attrition Risk)
1. **Leadership Roles (Research Director / Manager)** (`-0.50` log-odds): Strong career capital and retention.
2. **Longevity with Direct Manager (>3 years)** (`-0.49` log-odds): Strong manager rapport is a key retention anchor.
3. **High Workplace Satisfaction** (`-0.40` Environment, `-0.38` Job Satisfaction): Direct loyalty drivers.
4. **Stock Option Ownership (Level 1-3)**: Financial and equity alignment.

---

## 🚀 4. How to Use the System

### A. Run Model Training & Benchmarking
Trains all 4 models, computes 5-fold cross-validation, evaluates on holdout data, produces diagnostic charts, and saves model artifacts:
```powershell
python train.py
```
*Outputs generated:*
- `attrition_model.joblib`: Serialized trained pipeline
- `model_metadata.json`: Full metrics, schema, and feature importance data
- `evaluation_plots.png`: 4-quadrant diagnostic visualization (ROC, PR, Confusion Matrix, Feature Importance)

### B. Run Predictions (CLI)
#### Sample Archetype Demos:
```powershell
python predict.py --sample
```
*Example Output:*
- Junior Sales Rep (Frequent Overtime, Low Income, High Travel) -> **96.5% Probability (Critical Risk)**
- Senior Research Director (High Tenure, Good Pay, High Stock, No Overtime) -> **1.8% Probability (Low Risk)**

#### Batch Prediction on CSV:
```powershell
python predict.py --input WA_Fn-UseC_-HR-Employee-Attrition.csv --output predictions.csv
```

### C. Launch the Interactive Web Dashboard
```powershell
python server.py
```
Then open your browser at **`http://localhost:8080/`**.

**Features of the Dashboard:**
1. **Live Risk Simulator**: Dynamic SVG gauge meter, archetype presets, real-time sliders, personalized risk factors, and recommended HR actions.
2. **Model Benchmarks & Metrics**: Comparative metrics table, ROC-AUC, PR-AUC, and full diagnostic charts.
3. **HR Drivers & Strategic Insights**: Risk log-odds ranking and strategic retention playbooks.

---

## 📁 Project Structure

```
Classification_Of_Employee_Attrition/
├── WA_Fn-UseC_-HR-Employee-Attrition.csv   # Raw dataset (1,470 rows)
├── train.py                                # End-to-end training & benchmarking script
├── predict.py                              # CLI & module inference engine
├── server.py                               # Lightweight REST API and dashboard server
├── index.html                              # Interactive glassmorphic web dashboard
├── static/
│   ├── style.css                           # Modern responsive dark-mode styling
│   ├── app.js                              # Real-time predictor logic & SVG gauge
│   └── evaluation_plots.png                # High-res diagnostic plots
├── attrition_model.joblib                  # Serialized production pipeline
├── model_metadata.json                     # Comprehensive benchmark metrics
├── evaluation_plots.png                    # Diagnostic plots (ROC, PR, CM, Features)
└── README.md                               # System documentation
```
