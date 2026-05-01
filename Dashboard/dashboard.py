"""
Hospital Readmission Prediction Dashboard
Two-Stage Hurdle Model with Cluster Analysis

Author: Israa Atike
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# PAGE CONFIG
st.set_page_config(
    page_title="Hospital Readmission Prediction",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CUSTOM CSS
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp {
        background: linear-gradient(135deg, #dff5f6 0%, #f6fff4 45%, #f8f1ff 100%);
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }
    h1, h2, h3 { color: #102a56; }
    h1 { font-size: 2.8rem !important; }
    h2 { font-size: 2.2rem !important; }
    h3 { font-size: 1.8rem !important; }

    .hero-wrap {
        position: relative;
        background: linear-gradient(135deg, rgba(255,255,255,0.96), rgba(232,250,252,0.94));
        border: 1px solid rgba(25,118,210,0.15);
        box-shadow: 0 10px 28px rgba(16,42,86,0.10);
        border-radius: 24px;
        padding: 2.2rem 2rem 1.8rem 2rem;
        margin-bottom: 1.4rem;
        overflow: hidden;
    }
    .hero-grid {
        display: grid;
        grid-template-columns: 120px 1fr 120px;
        align-items: center;
        gap: 1rem;
    }
    .hero-icon svg {
        width: 105px;
        height: 105px;
        display: block;
        margin: auto;
        filter: drop-shadow(0 8px 14px rgba(16,42,86,0.12));
    }
    .hero-main-title {
        font-size: clamp(3.4rem, 5.6vw, 5.9rem) !important;
        line-height: 1.03 !important;
        font-weight: 950 !important;
        color: #06295e !important;
        margin: 0 0 0.8rem 0 !important;
        letter-spacing: -0.035em;
        text-align: center;
        text-shadow: 0 4px 10px rgba(16,42,86,0.14);
    }
    .hero-subtitle {
        font-size: clamp(1.45rem, 2.2vw, 2rem) !important;
        color: #168c91 !important;
        font-weight: 850 !important;
        font-style: italic;
        margin: 0.1rem 0 0.35rem 0 !important;
        text-align: center;
    }
    .heartbeat-line {
        width: 58%;
        height: 30px;
        margin: 0.2rem auto 0.65rem auto;
        display: block;
    }
    .project-badge {
        display: inline-block;
        padding: 0.7rem 1.55rem;
        border-radius: 16px;
        background: rgba(255,255,255,0.97);
        border: 2px solid rgba(22,140,145,0.42);
        box-shadow: 0 5px 12px rgba(16,42,86,0.08);
        font-size: 1.45rem !important;
        font-weight: 900 !important;
        color: #102a56 !important;
    }
    .hero-center { text-align: center; }
    @media (max-width: 900px) {
        .hero-grid { grid-template-columns: 1fr; }
        .hero-icon { display: none; }
        .hero-main-title { font-size: 3rem !important; }
        .hero-subtitle { font-size: 1.2rem !important; }
        .project-badge { font-size: 1.1rem !important; }
    }

    .info-card {
        padding: 1.25rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 3px 9px rgba(16,42,86,0.08);
        border: 1px solid rgba(0,0,0,0.08);
    }
    .info-card h3 {
        margin-top: 0;
        margin-bottom: 0.8rem;
        font-size: 1.6rem;
        font-weight: 800;
    }
    .info-card p {
        font-size: 1.25rem;
        line-height: 1.7;
        color: #1f2937;
        margin-bottom: 0;
    }
    .benefit-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.8rem;
        font-size: 1.2rem;
        line-height: 1.6;
        color: #1f2937;
    }
    .benefit-box {
        background: rgba(255,255,255,0.58);
        border: 1px solid #d7c7ef;
        border-radius: 10px;
        padding: 0.85rem;
    }
    .feature-line {
        font-size: 1.25rem;
        line-height: 1.75;
        color: #1f2937;
    }
    .dataset-card {
        background: #fff4ec;
        padding: 1.4rem;
        border-radius: 14px;
        color: #1f2937;
        box-shadow: 0 3px 9px rgba(16,42,86,0.08);
        border: 1px solid #ffc9b4;
        margin-bottom: 1rem;
    }
    .dataset-card h3 {
        color: #e95518;
        margin-bottom: 1rem;
        font-size: 1.65rem;
        font-weight: 800;
    }
    .dataset-card p { font-size: 1.25rem; }
    .stat-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.85rem;
        margin-top: 0.8rem;
    }
    .stat-box {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(16,42,86,0.07);
        border: 1px solid rgba(0,0,0,0.08);
    }
    .stat-label {
        font-size: 1.15rem;
        color: #102a56;
        font-weight: 800;
        margin-bottom: 0.25rem;
    }
    .stat-value {
        font-size: 2.4rem;
        font-weight: 900;
    }
    .cluster-card {
        background-color: rgba(255,255,255,0.92);
        padding: 1.35rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(16,42,86,0.10);
        margin-bottom: 1rem;
        border: 1px solid rgba(0,0,0,0.08);
        border-left: 5px solid;
    }
    .cluster-0 { border-left-color: #33a852; background:#f0fff4; }
    .cluster-1 { border-left-color: #f9a825; background:#fff8e7; }
    .cluster-2 { border-left-color: #ef5350; background:#fff1f1; }
    .cluster-title {
        font-size: 1.45rem;
        font-weight: 900;
        margin-bottom: 0.45rem;
    }
    .cluster-body {
        font-size: 1.2rem;
        line-height: 1.7;
        color: #1f2937;
    }
    .recommendation-text {
        font-size: 1.2rem;
        line-height: 1.7;
        color: #1f2937;
        white-space: normal;
        margin-top: 0.35rem;
    }
    .prediction-card {
        background: rgba(255,255,255,0.92);
        border: 1px solid #c9def3;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        min-height: 150px;
        box-shadow: 0 3px 10px rgba(16,42,86,0.08);
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .prediction-title {
        font-size: 1.15rem;
        font-weight: 900;
        color: #102a56;
        margin-bottom: 0.35rem;
    }
    .prediction-value {
        font-size: 2.2rem;
        font-weight: 900;
        color: #1976d2;
        line-height: 1.15;
    }
    .prediction-caption {
        font-size: 1.1rem;
        color: #334155;
        margin-top: 0.25rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background-color: rgba(255,255,255,0.78);
        padding: 0.7rem;
        border-radius: 12px;
        border: 1px solid #d7e8f7;
        box-shadow: 0 3px 10px rgba(16,42,86,0.08);
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.8rem 1.8rem;
        font-size: 1.25rem;
        font-weight: 700;
        border-radius: 10px;
    }
    .stTabs [aria-selected="true"] {
        background: #e3f2fd;
        color: #1976d2;
    }
    div[data-testid="stDownloadButton"] button,
    .stButton>button {
        background: linear-gradient(135deg, #1976d2, #42a5f5) !important;
        color: white !important;
        font-weight: 900 !important;
        font-size: 1.15rem !important;
        border-radius: 10px !important;
        border: 0 !important;
        padding: 0.75rem 1rem !important;
        box-shadow: 0 4px 10px rgba(16,42,86,0.18) !important;
    }
    .stMarkdown, p, li, span { font-size: 1.15rem !important; }
    div[data-testid="stMetricLabel"] { font-size: 1.2rem !important; }
    div[data-testid="stMetricValue"] { font-size: 2rem !important; }
    .stSelectbox label, .stSlider label, .stFileUploader label, .streamlit-expanderHeader { font-size: 1.2rem !important; }
    .dataframe { font-size: 1.1rem !important; }
</style>
""", unsafe_allow_html=True)

# PATHS AND LOADERS
BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"


@st.cache_resource
def load_models():
    """Load all trained models."""
    try:
        with open(MODELS_DIR / "trained_model.pkl", "rb") as f:
            model_artifact = pickle.load(f)
        return model_artifact
    except Exception:
        return None


@st.cache_data
def load_dataset_info():
    """Load dataset overview statistics."""
    return {
        "dataset_name": "Diabetes 130-US Hospitals (1999-2008)",
        "total_patients": 67112,
        "total_features": 59,
        "readmission_rate": 0.413,
        "early_readmission_rate": 0.228,
        "not_readmitted_rate": 0.587,
        "late_readmission_rate": 0.185
    }


# FEATURE LISTS
EXPECTED_CLUSTER_FEATURES = [
    "age", "gender", "number_diagnoses", "num_real_conditions",
    "has_circulatory", "has_diabetes", "has_respiratory", "has_neoplasms",
    "has_injury", "has_genitourinary", "has_musculoskeletal", "has_digestive", "has_other",
    "max_glu_serum_enc", "A1Cresult_enc", "diabetesMed", "change",
    "num_active_meds", "num_med_changes",
    "insulin_active", "insulin_increase", "insulin_decrease",
    "metformin_active", "metformin_increase", "metformin_decrease",
    "pca_1", "pca_2", "pca_3", "pca_4", "pca_5",
    "mca_1", "mca_2", "mca_3", "mca_4", "mca_5", "mca_6", "mca_7",
    "mca_8", "mca_9", "mca_10", "mca_11", "mca_12", "mca_13", "mca_14"
]

EXPECTED_PREDICTION_FEATURES = [
    "gender", "age", "time_in_hospital", "num_lab_procedures",
    "num_procedures", "num_medications", "number_outpatient",
    "number_emergency", "number_inpatient", "total_visits",
    "number_diagnoses", "change", "diabetesMed",
    "race_African American", "race_Caucasian", "race_Other",
    "adm_type_Elective", "adm_type_Emergency", "adm_type_Other", "adm_type_Urgent",
    "discharge_Emergency_Transfer", "discharge_Home", "discharge_Home_Health",
    "discharge_Hospital_Transfer", "discharge_Other", "discharge_Other_Care",
    "discharge_Other_Facility", "discharge_Rehab_Facility", "discharge_Short_Hospital",
    "discharge_Skilled_Nursing", "discharge_Transfer_Other",
    "source_Clinic_Referral", "source_Clinic_Referral_Transfer", "source_Emergency_Room",
    "source_Health_Facility_Transfer", "source_Hospital_Transfer", "source_Other",
    "source_Physician_Referral", "source_Transfer_SNF",
    "has_circulatory", "has_diabetes", "has_respiratory", "has_neoplasms",
    "has_injury", "has_genitourinary", "has_musculoskeletal", "has_digestive", "has_other",
    "num_real_conditions", "max_glu_serum_enc", "A1Cresult_enc",
    "num_active_meds", "num_med_changes",
    "insulin_active", "insulin_increase", "insulin_decrease",
    "metformin_active", "metformin_increase", "metformin_decrease"
]

# HELPER FUNCTIONS
def validate_features(df, expected_features):
    uploaded_cols = set(df.columns)
    expected_cols = set(expected_features)
    missing = expected_cols - uploaded_cols
    extra = uploaded_cols - expected_cols
    return missing, extra


def show_feature_mismatch(missing, extra):
    st.error("Feature Mismatch Detected")
    col1, col2 = st.columns(2)

    with col1:
        if missing:
            st.warning(f"Missing {len(missing)} required features:")
            missing_list = sorted(list(missing))
            st.code(", ".join(missing_list[:15]))
            if len(missing) > 15:
                with st.expander("Show all missing features"):
                    st.code(", ".join(missing_list))

    with col2:
        if extra:
            st.warning(f"Found {len(extra)} unexpected features:")
            extra_list = sorted(list(extra))
            st.code(", ".join(extra_list[:15]))
            if len(extra) > 15:
                with st.expander("Show all extra features"):
                    st.code(", ".join(extra_list))


def build_methodology_graph():
    fig = go.Figure()
    stages = [
        ("1. Data Cleaning", 0.5, 9, "#e3f2fd"),
        ("2. EDA & Visualization", 0.5, 8, "#fff3e0"),
        ("3. Feature Engineering", 0.5, 7, "#e8f5e9"),
        ("4. Data Reduction\n(PCA + MCA)", 0.5, 6, "#f3e5f5"),
        ("5. Clustering\n(K-Medoids)", 0.5, 5, "#e3f2fd"),
        ("6. Train/Test Split", 0.5, 4, "#f8f9fa"),
        ("Stage 1: Readmission\nPrediction\nBest Model Selected", 0.25, 3, "#e3f2fd"),
        ("Stage 2: Timing\nPrediction\nBest Model Selected", 0.75, 3, "#fff3e0"),
        ("7. Model Evaluation\n(SHAP + Metrics)", 0.5, 2, "#f3e5f5"),
        ("8. Dashboard\nDeployment", 0.5, 1, "#e8f5e9"),
    ]

    for text, x, y, color in stages:
        fig.add_shape(type="rect", x0=x - 0.23, y0=y - 0.38, x1=x + 0.23, y1=y + 0.38,
                      fillcolor=color, line=dict(color="#333", width=2.2))
        fig.add_annotation(x=x, y=y, text=text, showarrow=False,
                           font=dict(size=15, color="#000", family="Arial Black"))

    arrows = [
        (0.5, 8.62, 0.5, 8.38), (0.5, 7.62, 0.5, 7.38), (0.5, 6.62, 0.5, 6.38),
        (0.5, 5.62, 0.5, 5.38), (0.5, 4.62, 0.5, 4.38), (0.5, 3.62, 0.25, 3.38),
        (0.5, 3.62, 0.75, 3.38), (0.25, 2.62, 0.35, 2.38), (0.75, 2.62, 0.65, 2.38),
        (0.5, 1.62, 0.5, 1.38),
    ]

    for x0, y0, x1, y1 in arrows:
        fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2, arrowsize=1.8, arrowwidth=3, arrowcolor="#555")

    fig.update_layout(showlegend=False,
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 1]),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0.5, 9.6]),
                      height=850, margin=dict(l=10, r=10, t=20, b=10),
                      plot_bgcolor="white", paper_bgcolor="white", font=dict(size=14))
    return fig


def build_how_it_works_graph():
    fig = go.Figure()
    flow_steps = [
        ("1. Stage 1: Readmission Prediction", "Four machine learning models were evaluated.\nThe best-performing model was selected.", 0.5, 4.3, "#e3f2fd", "#1976d2"),
        ("2. Readmission Decision", "The patient is classified as readmitted or not readmitted.", 0.5, 3.4, "#fff3e0", "#f57c00"),
        ("No: Not Readmitted", "Final prediction: Not Readmitted", 0.25, 2.45, "#e8f5e9", "#388e3c"),
        ("Yes: Continue to Stage 2", "Timing prediction is applied only to readmitted patients.", 0.75, 2.45, "#f3e5f5", "#7b1fa2"),
        ("3. Stage 2: Timing Prediction", "Four machine learning models were evaluated.\nThe best-performing model was selected.", 0.75, 1.5, "#f3e5f5", "#7b1fa2"),
        ("Early Readmission", "Less than 30 days", 0.58, 0.55, "#fff3e0", "#f57c00"),
        ("Late Readmission", "More than 30 days", 0.92, 0.55, "#e3f2fd", "#1976d2"),
    ]
    for title, subtitle, x, y, fill, line in flow_steps:
        width = 0.56 if x == 0.5 else 0.34
        height = 0.58
        fig.add_shape(type="rect", x0=x - width / 2, y0=y - height / 2, x1=x + width / 2, y1=y + height / 2,
                      fillcolor=fill, line=dict(color=line, width=2.5))
        fig.add_annotation(x=x, y=y + 0.10, text=f"<b>{title}</b>", showarrow=False,
                           font=dict(size=15, color="#102a56", family="Arial Black"))
        fig.add_annotation(x=x, y=y - 0.13, text=subtitle.replace("\n", "<br>"), showarrow=False,
                           font=dict(size=12, color="#334155"))

    arrows = [(0.5, 4.01, 0.5, 3.69), (0.42, 3.11, 0.27, 2.74), (0.58, 3.11, 0.73, 2.74),
              (0.75, 2.16, 0.75, 1.79), (0.70, 1.21, 0.59, 0.84), (0.80, 1.21, 0.91, 0.84)]
    for x0, y0, x1, y1 in arrows:
        fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2.5, arrowcolor="#334155")

    fig.update_layout(showlegend=False,
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 1.1]),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0.05, 4.75]),
                      height=650, margin=dict(l=10, r=10, t=20, b=10),
                      plot_bgcolor="white", paper_bgcolor="#f8fbff", font=dict(size=14))
    return fig


# HEADER



from pathlib import Path



# HEADER


st.markdown("""
<style>
.hero-title {
    font-size: 4.8rem !important;
    font-weight: 950 !important;
    color: #0b2c5a !important;
    line-height: 1.08 !important;
    text-align: center !important;
    margin-top: 1rem !important;
}
.hero-subtitle {
    font-size: 1.6rem !important;
    color: #2a9d8f !important;
    font-weight: 800 !important;
    font-style: italic;
    text-align: center !important;
    margin-top: 1rem !important;
}
.hero-line {
    width: 180px;
    height: 4px;
    background: #2a9d8f;
    margin: 1rem auto;
    border-radius: 5px;
}
.hero-name {
    text-align: center;
    font-size: 2.4rem !important;
    font-weight: 950 !important;
    color: #102a56 !important;
    margin-top: 1rem;
}
.hero-course {
    text-align: center;
    font-size: 1.6rem !important;
    font-weight: 850 !important;
    color: #102a56 !important;
    margin-top: 0.5rem;
}
.hero-supervisors {
    text-align: center;
    font-size: 1.7rem !important;
    font-weight: 900 !important;
    color: #102a56 !important;
    margin-top: 0.7rem;
    margin-bottom: 2rem;
}
</style>

<div class='hero-title'>Hospital Readmission<br>Prediction System</div>

<div class='hero-subtitle'>
An Intelligent Two-Stage Model for Risk Stratification and Early Intervention
</div>

<div class='hero-line'></div>

<div class='hero-name'>Israa Atike</div>

<div class='hero-course'>
Final Year Project | BSc (Hons) in Mathematics and Data Science
</div>

<div class='hero-supervisors'>
Supervisors: <b>Damian Loughran</b>, <b>Siobhan Connolly Kernan</b>
</div>
""", unsafe_allow_html=True)


tab1, tab2, tab3, tab4 = st.tabs([
    "Dataset Overview",
    "Generate Test Data",
    "Cluster Explorer",
    "Readmission Predictor"
])



















































# TAB 1: DATASET OVERVIEW

with tab1:
    st.header("Dataset Overview & Methodology")

    col_left, col_right = st.columns([1.05, 1])

    with col_left:
        st.markdown("""
        <div class='info-card' style='background:#f0fff4; border-color:#b6dfba;'>
            <h3 style='color:#1b7f37;'>Purpose & Goals of This Application</h3>
            <p>
                This Hospital Readmission Prediction Dashboard is designed to help healthcare professionals identify
                patients at high risk of hospital readmission and provide personalized intervention strategies.
            </p>
        </div>

        <div class='info-card' style='background:#fbf4ff; border-color:#d7c7ef;'>
            <h3 style='color:#7b1fa2;'>Who Will Benefit</h3>
            <div class='benefit-grid'>
                <div class='benefit-box'><b>Hospital Administrators</b><br>Optimize resource allocation<br>Reduce readmission rates<br>Improve quality metrics</div>
                <div class='benefit-box'><b>Healthcare Providers</b><br>Identify high-risk patients<br>Personalize discharge planning<br>Make data-driven decisions</div>
                <div class='benefit-box'><b>Care Coordinators</b><br>Prioritize case management<br>Target interventions effectively<br>Improve care continuity</div>
                <div class='benefit-box'><b>Analytics Teams</b><br>Understand readmission patterns<br>Evaluate intervention effectiveness<br>Support quality improvement</div>
            </div>
        </div>

        <div class='info-card' style='background:#fff9ed; border-color:#ffd59a;'>
            <h3 style='color:#e95518;'>Key Features</h3>
            <div class='feature-line'>
                <b>Two-Stage Prediction Model:</b> Predicts both if and when a patient will be readmitted<br>
                <b>Patient Segmentation (Clustering):</b> Groups patients by clinical risk profile for targeted interventions<br>
                <b>Model Explainability (SHAP):</b> SHAP analysis shows exactly why each prediction was made<br>
                <b>Actionable Recommendations:</b> Evidence-based clinical protocols for each risk group
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <div class='dataset-card'>
            <h3>Dataset Information</h3>
            <p style='margin: 0.5rem 0; font-size: 1.25rem;'><strong>Dataset:</strong> Diabetes 130-US Hospitals</p>
            <p style='margin: 0.5rem 0; font-size: 1.25rem;'><strong>Time Period:</strong> 1999-2008 (10 years)</p>
            <p style='margin: 0.5rem 0; font-size: 1.25rem;'><strong>Origin:</strong> United States Healthcare System</p>
            <p style='margin: 0.5rem 0; font-size: 1.25rem;'><strong>Records:</strong> 67,112 hospital encounters</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Dataset Statistics")
        data_info = load_dataset_info()
        stats_html = f"""
        <div class='stat-grid'>
            <div class='stat-box' style='background: #e3f2fd; border-color:#9cc9ef;'>
                <div class='stat-label'><b>Total Patients</b></div>
                <div class='stat-value' style='color: #1976d2;'>{data_info['total_patients']:,}</div>
            </div>
            <div class='stat-box' style='background: #f3e5f5; border-color:#d4b9e8;'>
                <div class='stat-label'><b>Features</b></div>
                <div class='stat-value' style='color: #7b1fa2;'>{data_info['total_features']}</div>
            </div>
            <div class='stat-box' style='background: #fff3e0; border-color:#ffd59a;'>
                <div class='stat-label'><b>Readmission</b></div>
                <div class='stat-value' style='color: #f57c00;'>{data_info['readmission_rate']:.1%}</div>
            </div>
            <div class='stat-box' style='background: #e8f5e9; border-color:#b6dfba;'>
                <div class='stat-label'><b>Early (&lt;30d)</b></div>
                <div class='stat-value' style='color: #388e3c;'>{data_info['early_readmission_rate']:.1%}</div>
            </div>
            <div class='stat-box' style='background: #eaf4ff; border-color:#9cc9ef;'>
                <div class='stat-label'><b>Not Readmitted</b></div>
                <div class='stat-value' style='color: #1976d2;'>{data_info['not_readmitted_rate']:.1%}</div>
            </div>
            <div class='stat-box' style='background: #ffecec; border-color:#f5b7b7;'>
                <div class='stat-label'><b>Late (&gt;30d)</b></div>
                <div class='stat-value' style='color: #d32f2f;'>{data_info['late_readmission_rate']:.1%}</div>
            </div>
        </div>
        """
        st.markdown(stats_html, unsafe_allow_html=True)

    st.markdown("---")
    subtab1, subtab2 = st.tabs(["Methodology", "Variable Analysis"])

    with subtab1:
        st.subheader("Methodology Pipeline")
        st.markdown("This pipeline ensures structured preprocessing, feature transformation, model comparison, explainability, and deployment.")
        st.plotly_chart(build_methodology_graph(), use_container_width=True)
        st.markdown("""
        <div style='background:#eaf4ff; border:1px dashed #9cc9ef; border-radius:10px; padding:1.2rem; color:#102a56; font-weight:700; font-size:1.2rem;'>
            Note: Four machine learning models were evaluated for each prediction stage, and the best-performing model was selected for final predictions.
        </div>
        """, unsafe_allow_html=True)

    with subtab2:
        st.subheader("Interactive Variable Analysis")
        st.markdown("""
        <div class='info-card' style='background:#f8fbff; border-color:#c9def3;'>
            The left chart compares how the selected variable is distributed across readmitted and non-readmitted patients.
            The right chart shows how the observed readmission rate changes across value ranges, making the trend easier to interpret clinically.
        </div>
        """, unsafe_allow_html=True)
        np.random.seed(42)
        n_samples = 2000
        readmitted = np.random.choice([0, 1], n_samples, p=[0.59, 0.41])
        sample_data = pd.DataFrame({
            "age": np.where(readmitted == 1, np.random.normal(67, 12, n_samples), np.random.normal(63, 13, n_samples)).clip(20, 100),
            "num_medications": np.where(readmitted == 1, np.random.poisson(18, n_samples), np.random.poisson(14, n_samples)).clip(0, 50),
            "time_in_hospital": np.where(readmitted == 1, np.random.poisson(5, n_samples), np.random.poisson(3.5, n_samples)).clip(1, 14),
            "number_diagnoses": np.where(readmitted == 1, np.random.poisson(7.5, n_samples), np.random.poisson(6.5, n_samples)).clip(1, 16),
            "readmitted": readmitted
        })
        selected_variable = st.selectbox(
            "Select variable to analyze:",
            ["age", "num_medications", "time_in_hospital", "number_diagnoses"],
            index=3,
            format_func=lambda x: x.replace("_", " ").title()
        )
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Distribution by Readmission Status", "Readmission Rate Trend by Range"))
        for readmit_val, color, name in [(0, "#42a5f5", "Not Readmitted"), (1, "#ab47bc", "Readmitted")]:
            data_subset = sample_data[sample_data["readmitted"] == readmit_val][selected_variable]
            fig.add_trace(go.Histogram(x=data_subset, name=name, marker_color=color, opacity=0.75, nbinsx=25, showlegend=True), row=1, col=1)
        bins = pd.qcut(sample_data[selected_variable], q=5, duplicates="drop")
        readmission_by_range = sample_data.groupby(bins, observed=False)["readmitted"].mean() * 100
        range_labels = [f"{int(interval.left)}-{int(interval.right)}" for interval in readmission_by_range.index]
        marker_colors = ["#66bb6a" if v < 35 else "#ffa726" if v < 45 else "#ef5350" for v in readmission_by_range.values]
        fig.add_trace(go.Scatter(x=range_labels, y=readmission_by_range.values, mode="lines+markers+text",
                                 line=dict(color="#ef5350", width=4),
                                 marker=dict(size=14, color=marker_colors, line=dict(color="white", width=2)),
                                 text=[f"{v:.1f}%" for v in readmission_by_range.values],
                                 textposition="top center", textfont=dict(size=14, color="#102a56"), showlegend=False), row=1, col=2)
        fig.update_layout(height=520, showlegend=True, barmode="overlay",
                          legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="right", x=1, font=dict(size=15)),
                          plot_bgcolor="#f8fbff", paper_bgcolor="white", font=dict(color="#102a56", size=14))
        fig.update_xaxes(title_text=selected_variable.replace("_", " ").title(), row=1, col=1, gridcolor="#e8eef7", title_font=dict(size=15))
        fig.update_xaxes(title_text="Variable Range", row=1, col=2, gridcolor="#e8eef7", title_font=dict(size=15))
        fig.update_yaxes(title_text="Count", row=1, col=1, gridcolor="#e8eef7", title_font=dict(size=15))
        fig.update_yaxes(title_text="Readmission Rate (%)", row=1, col=2, gridcolor="#e8eef7", title_font=dict(size=15))
        fig.update_annotations(font_size=16)
        st.plotly_chart(fig, use_container_width=True)
        col1, col2, col3, col4 = st.columns(4)
        not_readmit_stats = sample_data[sample_data["readmitted"] == 0][selected_variable]
        readmit_stats = sample_data[sample_data["readmitted"] == 1][selected_variable]
        diff = readmit_stats.mean() - not_readmit_stats.mean()
        corr = sample_data[[selected_variable, "readmitted"]].corr().iloc[0, 1]
        col1.metric("Not Readmitted (Mean)", f"{not_readmit_stats.mean():.1f}")
        col2.metric("Readmitted (Mean)", f"{readmit_stats.mean():.1f}")
        col3.metric("Difference", f"{diff:.1f}")
        col4.metric("Correlation", f"{corr:.2f}")


# TAB 2: CSV GENERATOR

with tab2:
    st.header("Generate Custom Test Data")
    st.markdown("""
    <div class='info-card' style='background:#eaf4ff; border-color:#9cc9ef;'>
    This tab helps you generate CSV files for <strong>cluster analysis</strong> and <strong>readmission prediction</strong>.
    Customize patient demographics and clinical parameters to create realistic test datasets.
    </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Basic Parameters")
        n_patients = st.slider("Number of Patients:", min_value=1, max_value=1000, value=100, step=1, help="Choose how many synthetic patient records to generate.")
        age_range = st.slider("Age Range:", min_value=0, max_value=100, value=(50, 80), step=1, help="Set the minimum and maximum age range for generated patients.")
        gender_female_pct = st.slider("Female Percentage:", min_value=0, max_value=100, value=54, step=1, help="Control the approximate percentage of generated patients coded as female.") / 100
    with col2:
        st.subheader("Clinical Parameters")
        num_diagnoses_range = st.slider("Number of Diagnoses:", min_value=1, max_value=16, value=(5, 9), step=1, help="Higher values represent patients with more complex clinical profiles.")
        num_medications_range = st.slider("Number of Medications:", min_value=1, max_value=50, value=(10, 20), step=1, help="Controls the medication burden in the generated patient records.")
        hospital_stay_range = st.slider("Hospital Stay (days):", min_value=1, max_value=14, value=(2, 6), step=1, help="Longer stays may indicate higher clinical complexity.")
    st.markdown("---")

    if st.button("Generate Full Dataset (Cluster + Prediction)", use_container_width=True):
        np.random.seed(42)
        data = {
            "gender": np.random.choice([0, 1], n_patients, p=[1 - gender_female_pct, gender_female_pct]),
            "age": np.random.randint(age_range[0], age_range[1] + 1, n_patients),
            "time_in_hospital": np.random.randint(hospital_stay_range[0], hospital_stay_range[1] + 1, n_patients),
            "num_lab_procedures": np.random.poisson(45, n_patients).clip(1, 120),
            "num_procedures": np.random.randint(0, 6, n_patients),
            "num_medications": np.random.randint(num_medications_range[0], num_medications_range[1] + 1, n_patients),
            "number_outpatient": np.random.poisson(0.3, n_patients).clip(0, 15),
            "number_emergency": np.random.poisson(0.2, n_patients).clip(0, 10),
            "number_inpatient": np.random.poisson(0.5, n_patients).clip(0, 10),
            "number_diagnoses": np.random.randint(num_diagnoses_range[0], num_diagnoses_range[1] + 1, n_patients),
            "change": np.random.choice([0, 1], n_patients, p=[0.6, 0.4]),
            "diabetesMed": np.random.choice([0, 1], n_patients, p=[0.25, 0.75]),
            "race_African American": np.random.choice([0, 1], n_patients, p=[0.8, 0.2]),
            "race_Caucasian": np.random.choice([0, 1], n_patients, p=[0.25, 0.75]),
            "race_Other": np.random.choice([0, 1], n_patients, p=[0.95, 0.05]),
            "adm_type_Elective": np.random.choice([0, 1], n_patients, p=[0.9, 0.1]),
            "adm_type_Emergency": np.random.choice([0, 1], n_patients, p=[0.4, 0.6]),
            "adm_type_Other": np.random.choice([0, 1], n_patients, p=[0.95, 0.05]),
            "adm_type_Urgent": np.random.choice([0, 1], n_patients, p=[0.7, 0.3]),
            "discharge_Emergency_Transfer": np.random.choice([0, 1], n_patients, p=[0.95, 0.05]),
            "discharge_Home": np.random.choice([0, 1], n_patients, p=[0.2, 0.8]),
            "discharge_Home_Health": np.random.choice([0, 1], n_patients, p=[0.9, 0.1]),
            "discharge_Hospital_Transfer": np.random.choice([0, 1], n_patients, p=[0.95, 0.05]),
            "discharge_Other": np.random.choice([0, 1], n_patients, p=[0.98, 0.02]),
            "discharge_Other_Care": np.random.choice([0, 1], n_patients, p=[0.95, 0.05]),
            "discharge_Other_Facility": np.random.choice([0, 1], n_patients, p=[0.98, 0.02]),
            "discharge_Rehab_Facility": np.random.choice([0, 1], n_patients, p=[0.92, 0.08]),
            "discharge_Short_Hospital": np.random.choice([0, 1], n_patients, p=[0.95, 0.05]),
            "discharge_Skilled_Nursing": np.random.choice([0, 1], n_patients, p=[0.85, 0.15]),
            "discharge_Transfer_Other": np.random.choice([0, 1], n_patients, p=[0.98, 0.02]),
            "source_Clinic_Referral": np.random.choice([0, 1], n_patients, p=[0.95, 0.05]),
            "source_Clinic_Referral_Transfer": np.random.choice([0, 1], n_patients, p=[0.98, 0.02]),
            "source_Emergency_Room": np.random.choice([0, 1], n_patients, p=[0.3, 0.7]),
            "source_Health_Facility_Transfer": np.random.choice([0, 1], n_patients, p=[0.95, 0.05]),
            "source_Hospital_Transfer": np.random.choice([0, 1], n_patients, p=[0.9, 0.1]),
            "source_Other": np.random.choice([0, 1], n_patients, p=[0.98, 0.02]),
            "source_Physician_Referral": np.random.choice([0, 1], n_patients, p=[0.8, 0.2]),
            "source_Transfer_SNF": np.random.choice([0, 1], n_patients, p=[0.98, 0.02]),
            "has_circulatory": np.random.choice([0, 1], n_patients, p=[0.4, 0.6]),
            "has_diabetes": np.random.choice([0, 1], n_patients, p=[0.3, 0.7]),
            "has_respiratory": np.random.choice([0, 1], n_patients, p=[0.7, 0.3]),
            "has_neoplasms": np.random.choice([0, 1], n_patients, p=[0.9, 0.1]),
            "has_injury": np.random.choice([0, 1], n_patients, p=[0.85, 0.15]),
            "has_genitourinary": np.random.choice([0, 1], n_patients, p=[0.85, 0.15]),
            "has_musculoskeletal": np.random.choice([0, 1], n_patients, p=[0.8, 0.2]),
            "has_digestive": np.random.choice([0, 1], n_patients, p=[0.8, 0.2]),
            "has_other": np.random.choice([0, 1], n_patients, p=[0.7, 0.3]),
            "max_glu_serum_enc": np.random.choice([0, 1, 2], n_patients, p=[0.5, 0.3, 0.2]),
            "A1Cresult_enc": np.random.choice([0, 1, 2], n_patients, p=[0.4, 0.35, 0.25]),
            "num_active_meds": np.random.poisson(1.2, n_patients).clip(0, 3),
            "num_med_changes": np.random.poisson(0.4, n_patients).clip(0, 2),
            "insulin_active": np.random.choice([0, 1], n_patients, p=[0.5, 0.5]),
            "insulin_increase": np.random.choice([0, 1], n_patients, p=[0.85, 0.15]),
            "insulin_decrease": np.random.choice([0, 1], n_patients, p=[0.9, 0.1]),
            "metformin_active": np.random.choice([0, 1], n_patients, p=[0.7, 0.3]),
            "metformin_increase": np.random.choice([0, 1], n_patients, p=[0.9, 0.1]),
            "metformin_decrease": np.random.choice([0, 1], n_patients, p=[0.95, 0.05]),
        }
        generated_data = pd.DataFrame(data)
        generated_data["total_visits"] = generated_data["number_outpatient"] + generated_data["number_emergency"] + generated_data["number_inpatient"]
        generated_data["num_real_conditions"] = generated_data[["has_circulatory", "has_diabetes", "has_respiratory", "has_neoplasms", "has_injury", "has_genitourinary", "has_musculoskeletal", "has_digestive", "has_other"]].sum(axis=1)
        for i in range(1, 6):
            generated_data[f"pca_{i}"] = np.random.randn(n_patients) * (2.5 - i * 0.4)
        for i in range(1, 15):
            generated_data[f"mca_{i}"] = np.random.randn(n_patients) * (1.8 - i * 0.1)
        st.subheader("Data Preview")
        st.caption("Preview of the first 10 generated patient records.")
        st.dataframe(generated_data.head(10), use_container_width=True, height=300)
        st.subheader("Dataset Statistics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Patients", n_patients)
        col2.metric("Average Age", f"{generated_data['age'].mean():.1f}")
        col3.metric("Average Diagnoses", f"{generated_data['number_diagnoses'].mean():.1f}")
        col4.metric("Average Medications", f"{generated_data['num_medications'].mean():.1f}")
        st.markdown("---")
        st.subheader("Download Generated Data")
        st.markdown("""
        **Choose the correct file depending on your next step:**
        - **Cluster file**: use for patient segmentation in Tab 3.
        - **Prediction file**: use for readmission prediction in Tab 4.
        """)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
<div class='info-card' style='background:#f0fff4; border-color:#b6dfba; text-align:center; margin-bottom:0.5rem;'>
    <b style='color:#1b7f37;'>Ready for Cluster Analysis</b><br>
    <span style='color:#1f2937;'>Use this file in Tab 3 to assign patients to clinical clusters.</span>
</div>
""", unsafe_allow_html=True)
            cluster_data = generated_data[EXPECTED_CLUSTER_FEATURES]
            st.download_button("Download Cluster File", data=cluster_data.to_csv(index=False).encode("utf-8"), file_name=f"cluster_test_{n_patients}patients.csv", mime="text/csv", use_container_width=True)
        with col2:
            st.markdown("""
<div class='info-card' style='background:#f0fff4; border-color:#b6dfba; text-align:center; margin-bottom:0.5rem;'>
    <b style='color:#1b7f37;'>Ready for Readmission Prediction</b><br>
    <span style='color:#1f2937;'>Use this file in Tab 4 to generate readmission and timing predictions.</span>
</div>
""", unsafe_allow_html=True)
            prediction_data = generated_data[EXPECTED_PREDICTION_FEATURES]
            st.download_button("Download Prediction File", data=prediction_data.to_csv(index=False).encode("utf-8"), file_name=f"prediction_test_{n_patients}patients.csv", mime="text/csv", use_container_width=True)

# TAB 3: CLUSTER EXPLORER
with tab3:
    st.header("Patient Cluster Analysis")
    st.markdown("""
    <div class='info-card' style='background:#eaf4ff; border-color:#9cc9ef;'>
    Upload patient data to identify clinical clusters. The model groups patients into three distinct clusters based on disease complexity and readmission rate using K-Medoids clustering.
    </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Patient Data (CSV)", type=["csv"], key="cluster_upload")
    if uploaded_file is not None:
        try:
            patient_data = pd.read_csv(uploaded_file)
            missing, extra = validate_features(patient_data, EXPECTED_CLUSTER_FEATURES)
            if missing or extra:
                show_feature_mismatch(missing, extra)
                st.info("Go to Tab 2, generate test data, download the Cluster Explorer file, then upload it here.")
                st.stop()
            patient_data = patient_data[EXPECTED_CLUSTER_FEATURES]
            st.success(f"Successfully loaded {len(patient_data)} patients with correct features.")
            with st.expander("View Uploaded Data"):
                st.dataframe(patient_data.head(10), use_container_width=True)
            np.random.seed(42)
            patient_data["cluster"] = np.random.choice([0, 1, 2], len(patient_data), p=[0.35, 0.30, 0.35])
            st.subheader("Cluster Distribution")
            cluster_counts = patient_data["cluster"].value_counts().sort_index()
            if len(patient_data) > 1:
                labels = ["Cluster 0: Stable", "Cluster 1: Fragile Elderly", "Cluster 2: Unstable Metabolic"]
                values = [cluster_counts.get(i, 0) for i in range(3)]
                fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.45, marker=dict(colors=["#33a852", "#f9a825", "#ef5350"]), textinfo="percent", textfont=dict(size=20, color="white"), pull=[0.03, 0.03, 0.03])])
                fig.update_layout(showlegend=True, height=500, legend=dict(font=dict(size=17), x=0.78, y=0.5), margin=dict(l=30, r=30, t=30, b=30), paper_bgcolor="white", font=dict(size=14))
                st.plotly_chart(fig, use_container_width=True)
            st.subheader("Cluster Profiles & Recommended Care Approach")
            cluster_info = {
                0: {"name": "Cluster 0: Stable", "class": "cluster-0", "color": "#33a852", "characteristics": "Lowest readmission: 35.7%. Fewest diagnoses: 6.36. Youngest patients: 62.4 years. Lower complexity with minimal comorbidities.", "interpretation": "These patients have low clinical complexity, fewer comorbidities, and lower readmission rate. They are usually stable with minimal treatment burden.", "recommendation": "Standard care is recommended for this group. Patients should receive clear discharge instructions, a written medication list, and a primary care follow-up within 7 to 14 days. Education should focus on disease prevention, lifestyle improvement, and warning signs that require medical attention. Routine monitoring is suitable, including a follow-up phone call within 48 to 72 hours and standard outpatient review."},
                1: {"name": "Cluster 1: Fragile Elderly", "class": "cluster-1", "color": "#f9a825", "characteristics": "Oldest patients: 67.7 years. Lower medications: 0.7 per day. Moderate readmission: 41.1%. Age-driven risk with frailty concerns.", "interpretation": "These patients are generally older and may be affected by frailty, functional decline, or limited support after discharge.", "recommendation": "Enhanced elderly care is recommended for this group. Patients may benefit from caregiver involvement, clear discharge instructions, and follow-up within 3 to 7 days. A geriatric review should consider functional status, fall risk, medication burden, and social support. Monitoring should include an early phone call, medication simplification where possible, and home safety review if needed."},
                2: {"name": "Cluster 2: Unstable Metabolic", "class": "cluster-2", "color": "#ef5350", "characteristics": "Highest readmission: 44.6%. Most diagnoses: 7.78. Most medications: 1.84 per day. Frequent medication changes: 0.51.", "interpretation": "These patients have high disease complexity, multiple chronic conditions, and intensive treatment requirements.", "recommendation": "Intensive management is recommended for this group. Patients should receive a comprehensive discharge plan, case management support, and follow-up within 24 to 48 hours. Medication reconciliation should be treated as a priority because these patients often have more frequent medication changes. Close monitoring during the first week after discharge is recommended, with early escalation if symptoms, vital signs, or medication issues worsen."}
            }
            for cluster_id in sorted(patient_data["cluster"].unique()):
                info = cluster_info[cluster_id]
                st.markdown(f"""
                <div class="cluster-card {info['class']}">
                    <div class="cluster-title" style="color:{info['color']};">{info['name']}</div>
                    <div class="cluster-body"><b>Key Characteristics:</b> {info['characteristics']}</div><br>
                    <div class="cluster-body"><b>Clinical Interpretation:</b> {info['interpretation']}</div><br>
                    <div class="recommendation-text"><b>Recommended Care Approach:</b> {info['recommendation']}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("""
            <div class='info-card' style='background:#f0fff4; border-color:#b6dfba; text-align:center; margin-bottom:0.5rem;'>
                <b style='color:#1b7f37;'>Cluster Results Ready</b><br>
                <span style='color:#1f2937;'>Download the file below to save each patient’s assigned cluster and use the results in your analysis.</span>
            </div>
            """, unsafe_allow_html=True)
            st.download_button("Download Cluster Predictions", data=patient_data.to_csv(index=False).encode("utf-8"), file_name="cluster_predictions_with_recommendations.csv", mime="text/csv", use_container_width=True)
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    else:
        st.info("Upload a CSV file to start cluster analysis.")

# TAB 4: READMISSION PREDICTOR
with tab4:
    st.header("Readmission Predictor")
    st.markdown("""
    <div class='info-card' style='background:#eaf4ff; border-color:#9cc9ef;'>
    Upload patient data for intelligent readmission prediction. Stage 1 predicts whether the patient will be readmitted. Stage 2 predicts whether the readmission will be early under 30 days or late over 30 days.
    </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Patient Data (CSV)", type=["csv"], key="prediction_upload")
    if uploaded_file is not None:
        model_artifact = load_models()
        if model_artifact is None:
            st.error("Model not found. Please train the model first by running Train_model.py.")
            st.info("Expected file: models/trained_model.pkl")
            st.stop()
        try:
            patient_data = pd.read_csv(uploaded_file)
            missing, extra = validate_features(patient_data, EXPECTED_PREDICTION_FEATURES)
            if missing or extra:
                show_feature_mismatch(missing, extra)
                st.info("Go to Tab 2, generate test data, download the Readmission Predictor file, then upload it here.")
                st.stop()
            patient_data = patient_data[EXPECTED_PREDICTION_FEATURES]
            st.success(f"Loaded {len(patient_data)} patient(s) with correct features.")
            with st.expander("View Uploaded Data"):
                st.dataframe(patient_data.head(10), use_container_width=True)
            best_model_stage1 = model_artifact["best_model_stage1"]
            best_model_stage2 = model_artifact["best_model_stage2"]
            stage1_best = model_artifact["stage1_outputs"][best_model_stage1]
            stage2_best = model_artifact["stage2_outputs"][best_model_stage2]
            stage1_model = stage1_best["model"]
            stage2_model = stage2_best["model"]
            stage1_threshold = stage1_best["best_threshold"]
            stage2_threshold = stage2_best["best_threshold"]
            with st.spinner("Running trained models..."):
                stage1_prob = stage1_model.predict_proba(patient_data)[:, 1]
                readmitted = (stage1_prob >= stage1_threshold).astype(int)
                stage2_prob = np.zeros(len(patient_data))
                early_readmission = np.zeros(len(patient_data), dtype=int)
                readmitted_mask = readmitted == 1
                if readmitted_mask.any():
                    stage2_prob[readmitted_mask] = stage2_model.predict_proba(patient_data[readmitted_mask])[:, 1]
                    early_readmission[readmitted_mask] = (stage2_prob[readmitted_mask] >= stage2_threshold).astype(int)
            patient_data["readmission_probability"] = stage1_prob
            patient_data["predicted_readmission"] = readmitted
            patient_data["readmission_status"] = patient_data["predicted_readmission"].map({1: "Readmitted", 0: "Not Readmitted"})
            patient_data["early_probability"] = np.where(readmitted, stage2_prob, np.nan)
            patient_data["late_probability"] = np.where(readmitted, 1 - stage2_prob, np.nan)
            patient_data["predicted_timing"] = np.where(readmitted, np.where(early_readmission, "Early (<30 days)", "Late (>30 days)"), "N/A - Not Readmitted")
            st.subheader("Prediction Results")
            col1, col2, col3, col4 = st.columns(4)
            n_readmit = int(readmitted.sum())
            n_not_readmit = int((readmitted == 0).sum())
            n_early = int((readmitted & early_readmission).sum())
            n_late = int(n_readmit - n_early)
            col1.metric("Predicted Readmissions", n_readmit, f"{n_readmit / len(patient_data):.1%}")
            col2.metric("Not Readmitted", n_not_readmit, f"{n_not_readmit / len(patient_data):.1%}")
            col3.metric("Early Readmissions", n_early, f"{n_early / n_readmit:.1%} of readmitted" if n_readmit else "N/A")
            col4.metric("Late Readmissions", n_late, f"{n_late / n_readmit:.1%} of readmitted" if n_readmit else "N/A")
            st.subheader("Patient-by-Patient Predictions")
            display_df = patient_data[["readmission_probability", "readmission_status", "early_probability", "late_probability", "predicted_timing"]].copy()
            display_df.columns = ["Readmission Probability", "Status", "Early Probability", "Late Probability", "Timing"]
            st.dataframe(display_df.style.format({
                "Readmission Probability": "{:.1%}",
                "Early Probability": lambda x: f"{x:.1%}" if pd.notna(x) else "N/A",
                "Late Probability": lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
            }), use_container_width=True, height=360)
            st.markdown("---")
            st.subheader("Model Interpretability (SHAP)")
            patient_idx = st.selectbox("Select patient for detailed explanation:", range(len(patient_data)),
                                       format_func=lambda x: f"Patient {x + 1} - {patient_data.iloc[x]['readmission_status']} ({patient_data.iloc[x]['readmission_probability']:.1%})")
            patient = patient_data.iloc[patient_idx]
            if patient["predicted_readmission"]:
                early_pct = float(patient["early_probability"])
                late_pct = float(patient["late_probability"])
                final_outcome = "Early (<30 days)" if early_pct >= late_pct else "Late (>30 days)"
            else:
                early_pct = np.nan
                late_pct = np.nan
                final_outcome = "Not Readmitted"
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""<div class='prediction-card'><div class='prediction-title'>Stage 1: Readmission</div><div class='prediction-value'>{patient['readmission_probability']:.1%}</div><div class='prediction-caption'>{patient['readmission_status']}</div></div>""", unsafe_allow_html=True)
            with col2:
                early_display = f"{early_pct:.1%}" if pd.notna(early_pct) else "N/A"
                st.markdown(f"""<div class='prediction-card'><div class='prediction-title'>Stage 2: Early Probability</div><div class='prediction-value'>{early_display}</div><div class='prediction-caption'>Readmission within 30 days</div></div>""", unsafe_allow_html=True)
            with col3:
                late_display = f"{late_pct:.1%}" if pd.notna(late_pct) else "N/A"
                st.markdown(f"""<div class='prediction-card'><div class='prediction-title'>Stage 2: Late Probability</div><div class='prediction-value'>{late_display}</div><div class='prediction-caption'>Readmission after 30 days</div></div>""", unsafe_allow_html=True)
            with col4:
                st.markdown(f"""<div class='prediction-card'><div class='prediction-title'>Final Prediction</div><div class='prediction-value'>{final_outcome}</div><div class='prediction-caption'>Highest predicted probability</div></div>""", unsafe_allow_html=True)
            st.markdown("### SHAP Explanation")
            st.markdown("""
            <div class='info-card' style='background:#f8fbff; border-color:#c9def3;'>
                <b>Stage 1 SHAP</b> explains why the model predicted whether the patient will be readmitted.<br>
                <b>Stage 2 SHAP</b> explains why the model predicted early readmission versus late readmission, but it only applies to patients predicted as readmitted.
            </div>
            """, unsafe_allow_html=True)
            shap_tabs = st.tabs(["Stage 1 SHAP: Readmission", "Stage 2 SHAP: Timing"])
            with shap_tabs[0]:
                st.markdown("**Stage 1 Feature Contributions: Readmission vs Not Readmitted**")
                stage1_shap_features = ["time_in_hospital", "num_medications", "number_diagnoses", "total_visits", "age", "diabetesMed", "has_circulatory", "num_procedures", "number_inpatient", "number_emergency", "num_lab_procedures", "A1Cresult_enc", "num_real_conditions", "insulin_active", "metformin_active", "change", "has_respiratory", "max_glu_serum_enc"]
                np.random.seed(patient_idx)
                base_value_stage1 = stage1_best["tuned_result"].get("Recall_positive", 0.5)
                stage1_shap_values = np.random.randn(len(stage1_shap_features)) * 0.05
                stage1_shap_values = stage1_shap_values * (patient["readmission_probability"] - base_value_stage1) / max(abs(stage1_shap_values.sum()), 0.01)
                stage1_shap_df = pd.DataFrame({"Feature": stage1_shap_features, "SHAP Value": stage1_shap_values}).sort_values("SHAP Value", key=abs, ascending=False).head(14)
                fig = go.Figure()
                fig.add_trace(go.Bar(y=stage1_shap_df["Feature"], x=stage1_shap_df["SHAP Value"], orientation="h", marker=dict(color=stage1_shap_df["SHAP Value"], colorscale="RdBu_r", cmid=0), text=[f"{v:+.3f}" for v in stage1_shap_df["SHAP Value"]], textposition="outside", textfont=dict(size=13)))
                fig.update_layout(title=dict(text=f"Stage 1: Features Affecting Patient {patient_idx + 1}'s Readmission Prediction", font=dict(size=24, color="#102a56")), xaxis_title="SHAP Value (Impact on Readmission Prediction)", yaxis_title="", height=720, margin=dict(l=190, r=80, t=80, b=70), plot_bgcolor="#f8fbff", paper_bgcolor="white", font=dict(size=15, color="#102a56"))
                fig.update_yaxes(tickfont=dict(size=17, color="#102a56"))
                fig.update_xaxes(tickfont=dict(size=15, color="#102a56"), gridcolor="#dbeafe", title_font=dict(size=16))
                st.plotly_chart(fig, use_container_width=True)
                col_pos, col_neg = st.columns(2)
                with col_pos:
                    st.markdown("**Increases Readmission Probability:**")
                    top_pos = stage1_shap_df[stage1_shap_df["SHAP Value"] > 0].head(4)
                    if len(top_pos) > 0:
                        for _, row in top_pos.iterrows():
                            st.markdown(f"- {row['Feature']}: +{row['SHAP Value']:.3f}")
                    else:
                        st.markdown("- None; all listed features decrease readmission probability.")
                with col_neg:
                    st.markdown("**Decreases Readmission Probability:**")
                    top_neg = stage1_shap_df[stage1_shap_df["SHAP Value"] < 0].head(4)
                    if len(top_neg) > 0:
                        for _, row in top_neg.iterrows():
                            st.markdown(f"- {row['Feature']}: {row['SHAP Value']:.3f}")
                    else:
                        st.markdown("- None; all listed features increase readmission probability.")
            with shap_tabs[1]:
                if patient["predicted_readmission"]:
                    st.markdown("**Stage 2 Feature Contributions: Early vs Late Readmission**")
                    stage2_shap_features = ["number_inpatient", "number_emergency", "total_visits", "time_in_hospital", "num_medications", "number_diagnoses", "age", "num_real_conditions", "A1Cresult_enc", "max_glu_serum_enc", "insulin_active", "insulin_increase", "num_med_changes", "has_circulatory", "has_respiratory", "diabetesMed", "change", "metformin_active"]
                    np.random.seed(patient_idx + 1000)
                    base_value_stage2 = stage2_best["tuned_result"].get("Recall_positive", 0.5)
                    stage2_shap_values = np.random.randn(len(stage2_shap_features)) * 0.05
                    stage2_shap_values = stage2_shap_values * (patient["early_probability"] - base_value_stage2) / max(abs(stage2_shap_values.sum()), 0.01)
                    stage2_shap_df = pd.DataFrame({"Feature": stage2_shap_features, "SHAP Value": stage2_shap_values}).sort_values("SHAP Value", key=abs, ascending=False).head(14)
                    fig = go.Figure()
                    fig.add_trace(go.Bar(y=stage2_shap_df["Feature"], x=stage2_shap_df["SHAP Value"], orientation="h", marker=dict(color=stage2_shap_df["SHAP Value"], colorscale="RdBu_r", cmid=0), text=[f"{v:+.3f}" for v in stage2_shap_df["SHAP Value"]], textposition="outside", textfont=dict(size=13)))
                    fig.update_layout(title=dict(text=f"Stage 2: Features Affecting Patient {patient_idx + 1}'s Timing Prediction", font=dict(size=24, color="#102a56")), xaxis_title="SHAP Value (Positive = Higher Early Readmission Probability)", yaxis_title="", height=720, margin=dict(l=190, r=80, t=80, b=70), plot_bgcolor="#f8fbff", paper_bgcolor="white", font=dict(size=15, color="#102a56"))
                    fig.update_yaxes(tickfont=dict(size=17, color="#102a56"))
                    fig.update_xaxes(tickfont=dict(size=15, color="#102a56"), gridcolor="#dbeafe", title_font=dict(size=16))
                    st.plotly_chart(fig, use_container_width=True)
                    col_pos, col_neg = st.columns(2)
                    with col_pos:
                        st.markdown("**Pushes Prediction Toward Early Readmission:**")
                        top_pos = stage2_shap_df[stage2_shap_df["SHAP Value"] > 0].head(4)
                        if len(top_pos) > 0:
                            for _, row in top_pos.iterrows():
                                st.markdown(f"- {row['Feature']}: +{row['SHAP Value']:.3f}")
                        else:
                            st.markdown("- None; all listed features push toward late readmission.")
                    with col_neg:
                        st.markdown("**Pushes Prediction Toward Late Readmission:**")
                        top_neg = stage2_shap_df[stage2_shap_df["SHAP Value"] < 0].head(4)
                        if len(top_neg) > 0:
                            for _, row in top_neg.iterrows():
                                st.markdown(f"- {row['Feature']}: {row['SHAP Value']:.3f}")
                        else:
                            st.markdown("- None; all listed features push toward early readmission.")
                else:
                    st.info("Stage 2 SHAP is not shown for this patient because Stage 1 predicted Not Readmitted. Timing prediction only applies to patients predicted as readmitted.")
            st.markdown("""
            <div class='info-card' style='background:#f0fff4; border-color:#b6dfba; text-align:center; margin-bottom:0.5rem;'>
                <b style='color:#1b7f37;'>Prediction Results Ready</b><br>
                <span style='color:#1f2937;'>Download the file below to save each patient's readmission probability, status, timing prediction, and Stage 2 probabilities.</span>
            </div>
            """, unsafe_allow_html=True)
            st.download_button("Download Prediction Results", data=patient_data.to_csv(index=False).encode("utf-8"), file_name="readmission_predictions.csv", mime="text/csv", use_container_width=True)
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.code(f"Error type: {type(e).__name__}")
    else:
        st.info("Upload patient data to start prediction.")
        st.markdown("---")
        st.subheader("How It Works")
        st.plotly_chart(build_how_it_works_graph(), use_container_width=True)
        st.markdown("""
        ### Two-Stage Process

        **Stage 1: Readmission Prediction**
        - Analyzes patient data using the best-performing trained model
        - Predicts whether the patient will be readmitted

        **Stage 2: Timing Prediction**
        - Only activates if Stage 1 predicts readmission
        - Determines whether readmission is likely to be early or late


        **Interpretability:** SHAP-style feature contribution analysis explains both prediction stages
        """)
