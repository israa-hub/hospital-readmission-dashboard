#  Hospital Readmission Prediction System

An **Intelligent Two-Stage Machine Learning System** for predicting hospital readmissions and identifying high-risk patients for early intervention.

 Final Year Bachelor's Project  
Student: Israa Atike  

---

##  Project Overview

Hospital readmissions are a major challenge in healthcare systems. This project develops an **intelligent end-to-end pipeline** that combines **patient segmentation and predictive modeling** to support early intervention.

- Uses **K-Medoids clustering** to group patients into clinically meaningful risk profiles based on health conditions, treatments, and hospital usage patterns
- Predicts whether a patient will be readmitted (Stage 1)
- Predicts whether the readmission is early (<30 days) or late (>30 days) (Stage 2)

This hybrid approach combines **unsupervised learning (clustering)** with a **Two-Stage Hurdle Model**, enabling both **risk stratification and accurate prediction**.

The system is deployed as an **interactive Streamlit dashboard**.

---

##  Dataset

- **Dataset:** UCI Machine Learning Repository - Diabetes 130-US Hospitals Dataset
- **Records:** 67,112
- **Features:** 59
- **Time Period:** 1999–2008

##   Methodology

### Data Cleaning

Cleaned the dataset and performed feature engineering before data reduction, clustering and modelling phases.

###  Data Reduction

High-dimensional healthcare data was transformed using:

- **PCA (Principal Component Analysis)**
  - Applied to numerical features
  - Reduces dimensionality while preserving variance
  - Improves model efficiency and reduces noise

- **MCA (Multiple Correspondence Analysis)**
  - Applied to categorical variables
  - Captures relationships between categorical features
  - Converts them into numerical components

These transformed features (PCA + MCA) and some other raw features are used as inputs for clustering.

---

###  Patient Clustering

Patients are grouped into distinct clinical profiles using:

- **K-Medoids Clustering**

####  Purpose:
- Identify **patient risk groups**
- Support **targeted interventions**
- Improve **interpretability of predictions**

Each cluster represents a group of patients with similar:
- Clinical conditions
- Medication patterns
- Hospital usage behavior

---


###  Stage 1: Readmission Prediction
- Binary classification (Readmitted vs Not Readmitted)
- Models used:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - XGBoost

###  Stage 2: Timing Prediction
- Applied only to readmitted patients
- Models used:
  - Logistic Regression 
  - Decision Tree
  - Random Forest
  - XGBoost


###  Key Techniques
- SMOTE & SMOTEENN for class imbalance
- Hyperparameter tuning with GridSearchCV
- Threshold optimization (F1-score based)
- Cross-validation (Stratified K-Fold)
- SHAP for explainability

---


##  Dashboard Guide

### Tab 1: Dataset Overview

**Purpose & Goals Section:**
- Understand the application's objectives
- See who benefits (administrators, providers, coordinators, analysts)

**Methodology:**
- Visual pipeline showing 8-stage process
- Model assessment approach

**Variable Analysis:**
- Interactive exploration of key variables
- Distribution comparisons between readmitted/not readmitted
- Correlation analysis

### Tab 2: Generate Test Data

**Features:**
- Customize patient demographics (age, gender)
- Set clinical parameters (diagnoses, medications, hospital stay)
- Generate 1-1000 synthetic patients
- Download cluster-ready or prediction-ready CSV files

**Use Cases:**
- Test model predictions
- Create demo datasets
- Validate model behavior

### Tab 3: Cluster Explorer

**Upload Requirements:**
- CSV file with 44 features
- Features include: demographics, diagnoses, medications, PCA/MCA components

**Outputs:**
- Patient assignment to 3 risk clusters
- Cluster distribution visualization
- Clinical characteristics of each cluster
- Evidence-based care recommendations


### Tab 4: Readmission Predictor

**Upload Requirements:**
- CSV file with 59 features
- Complete clinical and demographic data

**Prediction Process:**
1. **Stage 1:** Predicts readmission probability
2. **Stage 2:** For readmitted patients, predicts early vs. late timing
3. **SHAP Analysis:** Explains which features drove the prediction

**Outputs:**
- Overall readmission statistics
- Patient-by-patient predictions
- Probability scores for each outcome
- Feature importance visualizations
- Downloadable results CSV

**SHAP Explanations:**
- Stage 1 SHAP: Why readmission was predicted
- Stage 2 SHAP: Why early or late timing was predicted
- Top positive/negative contributing features

---
```



## Disclaimer

This tool is designed for **research and educational purposes only**. It should not be used as the sole basis for clinical decision-making without validation by healthcare professionals. Always consult with qualified medical personnel for patient care decisions.


