// Interactive Employee Attrition Classification Client Application

let debounceTimer = null;

// Tab Switching
function switchTab(tabName) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.nav-tab-btn').forEach(el => el.classList.remove('active'));

  const targetTab = document.getElementById(`tab-${tabName}`);
  const targetBtn = document.getElementById(`tab-btn-${tabName}`);

  if (targetTab && targetBtn) {
    targetTab.classList.add('active');
    targetBtn.classList.add('active');
  }
}

// Slider live value update
function updateVal(name, prefix = '', suffix = '') {
  const inp = document.getElementById(`inp-${name}`);
  const badge = document.getElementById(`val-${name}`);
  if (inp && badge) {
    let val = inp.value;
    if (name === 'income' || name === 'monthlyrate' || name === 'dailyrate') {
      val = Number(val).toLocaleString();
    }
    badge.textContent = `${prefix}${val}${suffix}`;
  }
}

// Gather form data
function getFormData() {
  return {
    Age: parseInt(document.getElementById('inp-age').value),
    Gender: document.getElementById('inp-gender').value,
    MaritalStatus: document.getElementById('inp-marital').value,
    Department: document.getElementById('inp-dept').value,
    JobRole: document.getElementById('inp-role').value,
    EducationField: document.getElementById('inp-educationfield').value,
    JobLevel: parseInt(document.getElementById('inp-joblevel').value),
    Education: parseInt(document.getElementById('inp-education').value),

    MonthlyIncome: parseFloat(document.getElementById('inp-income').value),
    DistanceFromHome: parseFloat(document.getElementById('inp-distance').value),
    StockOptionLevel: parseInt(document.getElementById('inp-stock').value),
    PercentSalaryHike: parseFloat(document.getElementById('inp-hike').value),
    DailyRate: parseFloat(document.getElementById('inp-dailyrate').value),
    HourlyRate: parseFloat(document.getElementById('inp-hourlyrate').value),
    MonthlyRate: parseFloat(document.getElementById('inp-monthlyrate').value),
    NumCompaniesWorked: parseInt(document.getElementById('inp-companies').value),

    OverTime: document.getElementById('inp-overtime').value,
    BusinessTravel: document.getElementById('inp-travel').value,
    JobSatisfaction: parseInt(document.getElementById('inp-jobsat').value),
    EnvironmentSatisfaction: parseInt(document.getElementById('inp-envsat').value),
    WorkLifeBalance: parseInt(document.getElementById('inp-worklife').value),
    JobInvolvement: parseInt(document.getElementById('inp-jobinvolve').value),
    RelationshipSatisfaction: parseInt(document.getElementById('inp-relsat').value),
    PerformanceRating: parseInt(document.getElementById('inp-perf').value),

    TotalWorkingYears: parseFloat(document.getElementById('inp-totalworking').value),
    YearsAtCompany: parseFloat(document.getElementById('inp-yearsco').value),
    YearsInCurrentRole: parseFloat(document.getElementById('inp-yearsrole').value),
    YearsSinceLastPromotion: parseFloat(document.getElementById('inp-yearspromo').value),
    YearsWithCurrManager: parseFloat(document.getElementById('inp-yearsmgr').value),
    TrainingTimesLastYear: parseInt(document.getElementById('inp-training').value)
  };
}

// Trigger real-time prediction
function triggerPrediction() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(async () => {
    const payload = getFormData();
    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (!response.ok) throw new Error('Prediction API failed');
      const res = await response.json();
      updateUIWithResult(res);
    } catch (err) {
      console.warn('API error, falling back to local simulation heuristic:', err);
      simulateLocalPrediction(payload);
    }
  }, 120);
}

// Update UI with model prediction response
function updateUIWithResult(res) {
  const prob = res.attrition_probability;
  const tier = res.risk_tier;
  const color = res.tier_color || '#10b981';

  // Update text & gauge
  const probEl = document.getElementById('res-prob');
  probEl.textContent = `${prob.toFixed(1)}%`;
  probEl.style.color = color;

  // SVG Gauge calculations: Arc length is ~251.3
  const gaugeFill = document.getElementById('gauge-path-fill');
  // Perimeter of semi-circle arc with r=80: PI * 80 ~= 251.32
  const maxOffset = 251.32;
  const offset = maxOffset - (maxOffset * Math.min(prob, 100) / 100);
  gaugeFill.style.strokeDasharray = `${maxOffset}`;
  gaugeFill.style.strokeDashoffset = `${offset}`;
  gaugeFill.style.stroke = color;

  // Badge box
  const tierBadge = document.getElementById('res-tier-badge');
  tierBadge.textContent = tier;
  tierBadge.style.color = color;
  tierBadge.style.backgroundColor = `${color}22`;
  tierBadge.style.border = `1px solid ${color}44`;

  document.getElementById('res-explanation').textContent = res.recommended_action || '';
  document.getElementById('res-action').textContent = res.recommended_action || '';

  // Drivers
  const driversList = document.getElementById('res-drivers-list');
  driversList.innerHTML = '';
  if (res.key_risk_factors && res.key_risk_factors.length > 0) {
    res.key_risk_factors.forEach(f => {
      const li = document.createElement('li');
      const isRisk = f.includes('burnout') || f.includes('Overtime') || f.includes('Low') || f.includes('distance') || f.includes('stagnation') || f.includes('Zero stock');
      li.className = `driver-item ${isRisk ? 'risk' : 'safe'}`;
      li.textContent = f;
      driversList.appendChild(li);
    });
  }
}

// Fallback in case network is disconnected
function simulateLocalPrediction(p) {
  let score = 0.15;
  if (p.OverTime === 'Yes') score += 0.35;
  if (p.BusinessTravel === 'Travel_Frequently') score += 0.18;
  if (p.MaritalStatus === 'Single') score += 0.14;
  if (p.MonthlyIncome < 3000) score += 0.16;
  if (p.StockOptionLevel === 0) score += 0.12;
  if (p.DistanceFromHome > 15) score += 0.08;
  if (p.JobSatisfaction <= 2) score += 0.10;
  if (p.YearsWithCurrManager > 3) score -= 0.15;
  if (p.JobRole === 'Research Director' || p.JobRole === 'Manager') score -= 0.18;
  if (p.YearsAtCompany > 8) score -= 0.12;

  score = Math.max(0.01, Math.min(0.98, score));
  const probPct = score * 100;
  let tier = 'Low Risk', color = '#10b981', action = 'Standard retention path.';
  if (probPct >= 65) {
    tier = 'Critical Risk'; color = '#ef4444'; action = 'Urgent intervention needed. High probability of departure.';
  } else if (probPct >= 40) {
    tier = 'High Risk'; color = '#f97316'; action = 'Active attrition warning. Immediate 1-on-1 manager review recommended.';
  } else if (probPct >= 20) {
    tier = 'Moderate Risk'; color = '#f59e0b'; action = 'Monitor engagement, check-in on workload and growth path.';
  }

  updateUIWithResult({
    attrition_probability: probPct,
    risk_tier: tier,
    tier_color: color,
    recommended_action: action,
    key_risk_factors: [
      p.OverTime === 'Yes' ? 'Working frequent Overtime (high burnout factor)' : 'Standard work hours without excessive overtime',
      p.MonthlyIncome < 3500 ? `Below-average salary ($${p.MonthlyIncome.toLocaleString()}/mo)` : `Competitive compensation ($${p.MonthlyIncome.toLocaleString()}/mo)`,
      p.YearsWithCurrManager <= 1 ? 'Recent manager reassignment (<1 year)' : `Long-term manager rapport (${p.YearsWithCurrManager} years)`
    ]
  });
}

// Load Predefined Archetype Presets
const PRESETS = {
  junior_burnout: {
    Age: 24, Gender: 'Female', MaritalStatus: 'Single', Department: 'Sales', JobRole: 'Sales Representative',
    EducationField: 'Marketing', JobLevel: 1, Education: 2, MonthlyIncome: 2200, DistanceFromHome: 24,
    StockOptionLevel: 0, PercentSalaryHike: 11, DailyRate: 420, HourlyRate: 45, MonthlyRate: 12000,
    NumCompaniesWorked: 3, OverTime: 'Yes', BusinessTravel: 'Travel_Frequently', JobSatisfaction: 1,
    EnvironmentSatisfaction: 1, WorkLifeBalance: 1, JobInvolvement: 1, RelationshipSatisfaction: 2,
    PerformanceRating: 3, TotalWorkingYears: 2, YearsAtCompany: 1, YearsInCurrentRole: 0,
    YearsSinceLastPromotion: 0, YearsWithCurrManager: 0, TrainingTimesLastYear: 1
  },
  mid_stagnant: {
    Age: 34, Gender: 'Male', MaritalStatus: 'Single', Department: 'Research & Development',
    JobRole: 'Laboratory Technician', EducationField: 'Medical', JobLevel: 2, Education: 3,
    MonthlyIncome: 3800, DistanceFromHome: 18, StockOptionLevel: 0, PercentSalaryHike: 12,
    DailyRate: 750, HourlyRate: 60, MonthlyRate: 16000, NumCompaniesWorked: 4, OverTime: 'Yes',
    BusinessTravel: 'Travel_Rarely', JobSatisfaction: 2, EnvironmentSatisfaction: 2, WorkLifeBalance: 2,
    JobInvolvement: 2, RelationshipSatisfaction: 2, PerformanceRating: 3, TotalWorkingYears: 8,
    YearsAtCompany: 6, YearsInCurrentRole: 5, YearsSinceLastPromotion: 5, YearsWithCurrManager: 1,
    TrainingTimesLastYear: 2
  },
  senior_stable: {
    Age: 49, Gender: 'Female', MaritalStatus: 'Married', Department: 'Research & Development',
    JobRole: 'Research Director', EducationField: 'Life Sciences', JobLevel: 4, Education: 4,
    MonthlyIncome: 17500, DistanceFromHome: 3, StockOptionLevel: 2, PercentSalaryHike: 19,
    DailyRate: 1350, HourlyRate: 88, MonthlyRate: 21000, NumCompaniesWorked: 2, OverTime: 'No',
    BusinessTravel: 'Non-Travel', JobSatisfaction: 4, EnvironmentSatisfaction: 4, WorkLifeBalance: 4,
    JobInvolvement: 4, RelationshipSatisfaction: 4, PerformanceRating: 4, TotalWorkingYears: 26,
    YearsAtCompany: 19, YearsInCurrentRole: 12, YearsSinceLastPromotion: 1, YearsWithCurrManager: 11,
    TrainingTimesLastYear: 3
  },
  average_employee: {
    Age: 36, Gender: 'Male', MaritalStatus: 'Married', Department: 'Sales', JobRole: 'Sales Executive',
    EducationField: 'Life Sciences', JobLevel: 2, Education: 3, MonthlyIncome: 5800, DistanceFromHome: 8,
    StockOptionLevel: 1, PercentSalaryHike: 14, DailyRate: 850, HourlyRate: 68, MonthlyRate: 14000,
    NumCompaniesWorked: 2, OverTime: 'No', BusinessTravel: 'Travel_Rarely', JobSatisfaction: 3,
    EnvironmentSatisfaction: 3, WorkLifeBalance: 3, JobInvolvement: 3, RelationshipSatisfaction: 3,
    PerformanceRating: 3, TotalWorkingYears: 10, YearsAtCompany: 6, YearsInCurrentRole: 4,
    YearsSinceLastPromotion: 1, YearsWithCurrManager: 4, TrainingTimesLastYear: 3
  }
};

function loadPreset(key) {
  const p = PRESETS[key];
  if (!p) return;

  document.getElementById('inp-age').value = p.Age;
  document.getElementById('inp-gender').value = p.Gender;
  document.getElementById('inp-marital').value = p.MaritalStatus;
  document.getElementById('inp-dept').value = p.Department;
  document.getElementById('inp-role').value = p.JobRole;
  document.getElementById('inp-educationfield').value = p.EducationField;
  document.getElementById('inp-joblevel').value = p.JobLevel;
  document.getElementById('inp-education').value = p.Education;

  document.getElementById('inp-income').value = p.MonthlyIncome;
  document.getElementById('inp-distance').value = p.DistanceFromHome;
  document.getElementById('inp-stock').value = p.StockOptionLevel;
  document.getElementById('inp-hike').value = p.PercentSalaryHike;
  document.getElementById('inp-dailyrate').value = p.DailyRate;
  document.getElementById('inp-hourlyrate').value = p.HourlyRate;
  document.getElementById('inp-monthlyrate').value = p.MonthlyRate;
  document.getElementById('inp-companies').value = p.NumCompaniesWorked;

  document.getElementById('inp-overtime').value = p.OverTime;
  document.getElementById('inp-travel').value = p.BusinessTravel;
  document.getElementById('inp-jobsat').value = p.JobSatisfaction;
  document.getElementById('inp-envsat').value = p.EnvironmentSatisfaction;
  document.getElementById('inp-worklife').value = p.WorkLifeBalance;
  document.getElementById('inp-jobinvolve').value = p.JobInvolvement;
  document.getElementById('inp-relsat').value = p.RelationshipSatisfaction;
  document.getElementById('inp-perf').value = p.PerformanceRating;

  document.getElementById('inp-totalworking').value = p.TotalWorkingYears;
  document.getElementById('inp-yearsco').value = p.YearsAtCompany;
  document.getElementById('inp-yearsrole').value = p.YearsInCurrentRole;
  document.getElementById('inp-yearspromo').value = p.YearsSinceLastPromotion;
  document.getElementById('inp-yearsmgr').value = p.YearsWithCurrManager;
  document.getElementById('inp-training').value = p.TrainingTimesLastYear;

  // Sync badges
  updateVal('age');
  updateVal('joblevel');
  updateVal('education');
  updateVal('income', '$');
  updateVal('distance', '', ' mi');
  updateVal('stock');
  updateVal('hike', '', '%');
  updateVal('dailyrate', '$');
  updateVal('hourlyrate', '$');
  updateVal('monthlyrate', '$');
  updateVal('companies');
  updateVal('jobsat', '', '/4');
  updateVal('envsat', '', '/4');
  updateVal('worklife', '', '/4');
  updateVal('jobinvolve', '', '/4');
  updateVal('relsat', '', '/4');
  updateVal('totalworking', '', ' yrs');
  updateVal('yearsco', '', ' yrs');
  updateVal('yearsrole', '', ' yrs');
  updateVal('yearspromo', '', ' yr');
  updateVal('yearsmgr', '', ' yrs');
  updateVal('training');

  triggerPrediction();
}

function resetForm() {
  loadPreset('average_employee');
}

// Initialize on page load
window.addEventListener('DOMContentLoaded', () => {
  resetForm();
});
