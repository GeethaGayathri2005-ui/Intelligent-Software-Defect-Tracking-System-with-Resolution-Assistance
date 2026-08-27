# Intelligent Software Defect Tracking System with Resolution Assistance

**Bug Lifecycle Analytics • Software Quality Intelligence**

An interactive dashboard for monitoring, analyzing, and improving the software defect lifecycle using data-driven insights.

`Python` • `Streamlit` • `Pandas` • `Plotly` | **License:** MIT

---

## 📍 Navigation
[Overview](#overview) | [Objectives](#objectives) | [Features](#features) | [Technology Stack](#technology-stack) | [Dashboard Components](#dashboard-components) | [KPIs](#kpis) | [Dataset Structure](#dataset-structure) | [Run Locally](#run-locally) | [Future Enhancements](#future-enhancements) | [Author](#author) | [License](#license)

---

## 📌 Project Overview
Software development teams continuously generate bug reports during development, testing, and maintenance. When these records are analyzed manually, it becomes difficult to identify:

* High-defect modules
* Recurring root causes
* High-severity bugs
* Priority distribution
* Increasing bug trends
* Long resolution times
* Memory-related problems

This project provides a centralized interactive dashboard for analyzing software defects and supporting software-quality decisions.

---

## 🔄 System Workflow

```text
       [ 🐞 BUG REPORTS ]
                │
                ▼
      [ 📊 DATA ANALYSIS ]
         ┌──────┼──────┐
         ▼      ▼      ▼
       KPIs   Trends  Root Causes
         └──────┼──────┘
                │
                ▼
        [ 💡 INSIGHTS ]
                │
                ▼
[ 📈 SOFTWARE QUALITY IMPROVEMENT ]

🎯 Objectives
| Objective | Purpose |
|---|---|
| Bug Monitoring | Track the overall defect lifecycle |
| Quality Analysis | Understand software defect patterns |
| Severity Analysis | Identify high-severity defects |
| Priority Analysis | Identify bugs requiring immediate attention |
| Module Analysis | Find high-defect modules |
| Root Cause Analysis | Identify recurring defect sources |
| Resolution Analysis | Monitor time required to resolve bugs |
| Trend Analysis | Monitor bug reporting over time |
| Memory Analysis | Analyze baseline and peak memory behavior |
| Insights | Support data-driven software-quality decisions |
🔑 Key Features
📊 KPI Intelligence
Six key metrics provide an instant overview of software quality:
| Metric | Description |
|---|---|
| TOTAL BUGS | Total number of logged defect records |
| RESOLVED BUGS | Total bugs successfully resolved |
| OPEN BUGS | Total active/unresolved bugs |
| AVG TIME | Average time taken to resolve issues |
| DENSITY | Calculated defect density ratio |
| TOP MODULE | Module with the highest filtered bug count |
🔍 Interactive Filtering
Filter the complete dashboard using:
 * Status
 * Severity
 * Priority
 * Product Module
 * Root Cause
📈 Interactive Analytics
The dashboard provides:
 * Bug status distribution
 * Severity distribution
 * Priority distribution
 * Module-wise bug analysis
 * Monthly bug trends
 * Average resolution-time trends
 * Root-cause treemap
 * Memory analysis
 * Detailed bug records
💡 Automated Insights
The system generates recommendations based on the selected data, including:
 * Prioritize high-severity bugs and focus testing efforts on high-defect modules.
🛠️ Technology Stack
| Technology | Role |
|---|---|
| Python | Core programming language |
| Streamlit | Dashboard framework |
| Pandas | Data processing and analysis |
| Plotly Express | Interactive visualization |
| Plotly Graph Objects | Advanced visualization |
| CSV | Dataset storage |
🧩 Dashboard Components
| Component | What It Shows | Business Value |
|---|---|---|
| Status | Open vs Resolved | Lifecycle monitoring |
| Severity | Severity distribution | Risk identification |
| Priority | Priority distribution | Work prioritization |
| Modules | Bugs by module | High-risk module detection |
| Bug Trend | Bugs reported monthly | Trend monitoring |
| Resolution | Average resolution time | Team efficiency |
| Root Cause | Recurring causes | Quality improvement |
| Memory | Baseline vs peak memory | Memory issue detection |
| Records | Filtered bug data | Detailed investigation |
📐 KPI Calculations
 * Total Bugs:
   
 * Resolved Bugs:
   
 * Open Bugs:
   
 * Average Resolution Time:
   
 * Defect Density:
   
   > Note: This is the project's current dashboard definition using users affected. A formal engineering definition could instead use defects per KLOC, function points, or module size.
   > 
 * Top Module: Module with the highest filtered bug count.
📁 Dataset Structure
The dashboard uses a CSV-based software defect dataset (data/memory_leak_bug_reports_dataset.csv).
| Field | Description |
|---|---|
| bug_id | Unique bug identifier |
| status | Current bug status |
| severity | Bug severity |
| priority | Bug priority |
| product_module | Product/module name |
| root_cause_category | Root cause category |
| created_at | Bug creation date |
| resolved_at | Bug resolution date |
| users_affected | Number of affected users |
| baseline_memory_mb | Baseline memory |
| peak_memory_mb | Peak memory |
📂 Project Structure
Bug-Lifecycle-Management/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── memory_leak_bug_reports_dataset.csv
└── images/
    ├── dashboard-overview.png
    ├── kpi-dashboard.png
    ├── bug-status.png
    ├── severity-analysis.png
    ├── priority-analysis.png
    ├── module-analysis.png
    ├── bug-trend.png
    ├── resolution-trend.png
    ├── root-cause-analysis.png
    ├── memory-analysis.png
    └── bug-records.png

🚀 How to Run Locally
Step 1 — Install Python
Download Python from python.org.
Verify:
python --version
# or
py --version

Step 2 — Install Git
Download Git from git-scm.com.
Verify:
git --version

Step 3 — Clone the Repository
git clone [https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git](https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git)
cd YOUR-REPOSITORY

Step 4 — Create Virtual Environment
python -m venv venv

Activate:
# Windows (cmd/PowerShell):
venv\Scripts\activate

If PowerShell blocks script execution, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

Step 5 — Install Dependencies
python -m pip install -r requirements.txt

Step 6 — Configure Dataset
Place the dataset inside the data/ folder:
data/memory_leak_bug_reports_dataset.csv
Ensure relative path loading in app.py:
df = pd.read_csv("data/memory_leak_bug_reports_dataset.csv")

Step 7 — Start Dashboard
python -m streamlit run app.py

Open in browser: http://localhost:8501
📝 Requirements
Create requirements.txt with:
streamlit
pandas
plotly

Install command:
python -m pip install -r requirements.txt

📤 GitHub Upload (First-Time Setup)
git init
git add .
git commit -m "Initial commit - Bug Lifecycle Dashboard"
git branch -M main
git remote add origin [https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git](https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git)
git push -u origin main

After Changing the Project:
git add .
git commit -m "Update dashboard"
git push

Check Status:
git status

Check Remote:
git remote -v

🔮 Future Enhancements (Machine Learning)
Future versions can include:
 * Bug severity prediction
 * Bug priority prediction
 * Duplicate bug detection
 * Bug classification
 * Root-cause prediction
 * Resolution-time prediction
Potential Algorithms:
 * Naive Bayes
 * Logistic Regression
 * Decision Tree
 * Random Forest
 * Support Vector Machine
Database Integration:
Replace CSV storage with MySQL, PostgreSQL, SQLite, or MongoDB.
👤 Author
Yalagandula Ganesh
Software Defect Tracking & Bug Lifecycle Management Dashboard
A software-quality analytics project focused on:
Analyze → Monitor → Identify → Improve
📜 License
This project is released under the MIT License.
To use the MIT License, create a file named LICENSE and add appropriate MIT License text.
