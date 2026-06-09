"""
South German Credit Risk — Streamlit Deployment App (CRISP-ML Phase 6)

Real-time credit-risk prediction against the trained pipeline
(`model/best_model.joblib`). The full CRISP-ML analysis (Phases 1–5) lives in
the notebook; this app is the deployment layer only.
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Credit Risk Predictor", page_icon="💳", layout="centered")

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model" / "best_model.joblib"
META_PATH = ROOT / "model" / "model_metadata.pkl"

COLORS = {
    "ice_silver": "#E6E8EB", "graphite": "#2A3038", "espresso_gold": "#C9A86A",
    "slate": "#424A53", "pebble": "#5E757D", "mist": "#B0B4B8",
    "silver": "#D5D6DB", "platinum": "#EBECEF",
    "info": "#4A67B0", "success": "#43936C", "warning": "#F2AE4A", "danger": "#D96B5F",
}

st.markdown(f"""
<style>
    .stApp {{ background-color: {COLORS['ice_silver']}; }}
    .main .block-container {{
        background-color: {COLORS['platinum']}; border-radius: 12px;
        padding: 2rem; box-shadow: 0 4px 20px rgba(36,3,56,0.1); }}
    h1, h2, h3 {{ color: {COLORS['graphite']} !important; }}
    h1 {{ border-bottom: 3px solid {COLORS['espresso_gold']}; padding-bottom: 0.5rem; }}
    p, span, label, .stMarkdown {{ color: {COLORS['graphite']}; }}
    .stCaption, small {{ color: {COLORS['pebble']} !important; }}
    hr {{ border-color: {COLORS['mist']} !important; }}
    .stButton > button[kind="primary"] {{
        background-color: {COLORS['espresso_gold']} !important; color: {COLORS['graphite']} !important;
        border: none !important; font-weight: 600; }}
    .stButton > button[kind="primary"]:hover {{
        background-color: {COLORS['slate']} !important; color: {COLORS['ice_silver']} !important; }}
    .stMetric {{ background-color: white; padding: 1rem; border-radius: 8px;
        border: 1px solid {COLORS['mist']}; box-shadow: 0 2px 8px rgba(36,3,56,0.08); }}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_meta():
    try:
        return joblib.load(META_PATH)
    except Exception:
        return {}


model = load_model()
meta = load_meta()
ct_info = meta.get("cost_optimal_threshold", {})
opt_threshold = float(ct_info.get("threshold", 0.5))

# ----------------------------- header -----------------------------
st.markdown(f"""
<div style="text-align:center; margin-bottom:1rem;">
    <h1 style="border-bottom:none; padding-bottom:0; font-size:2.2rem;">💳 Credit Risk Predictor</h1>
    <div style="width:80px; height:3px; background:{COLORS['espresso_gold']};
                margin:0.25rem auto 1rem auto; border-radius:2px;"></div>
    <p style="color:{COLORS['pebble']};">
        Predice si un solicitante es <strong>buen (0)</strong> o
        <strong>mal (1)</strong> riesgo crediticio · XGBoost + SMOTETomek
    </p>
</div>
""", unsafe_allow_html=True)

# ----------------------------- decision threshold -----------------------------
options = ["Por defecto (0.50)"]
if ct_info:
    options.append(f"Cost-óptimo ({opt_threshold:.2f}) — FN 5× más costoso que FP, prioriza recall")
mode = st.radio("Umbral de decisión", options, index=0)
threshold = 0.50 if mode.startswith("Por defecto") else opt_threshold

st.divider()
st.subheader("Información del solicitante")

col1, col2 = st.columns(2)
with col1:
    status = st.selectbox("Account Status", [1, 2, 3, 4],
        format_func=lambda x: {1: "No account", 2: "No balance", 3: "< 200 DM", 4: ">= 200 DM"}[x])
    duration = st.number_input("Loan Duration (months)", 1, 72, 24)
    amount = st.number_input("Credit Amount (DM)", 250, 20000, 2500)
    age = st.number_input("Age (years)", 18, 80, 35)
    savings = st.selectbox("Savings Account", [1, 2, 3, 4, 5],
        format_func=lambda x: {1: "Unknown/None", 2: "< 100 DM", 3: "100-500 DM", 4: "500-1000 DM", 5: ">= 1000 DM"}[x])
with col2:
    credit_history = st.selectbox("Credit History", [0, 1, 2, 3, 4],
        format_func=lambda x: {0: "Delay in past", 1: "Critical account", 2: "No credits", 3: "Existing paid", 4: "All paid"}[x])
    purpose = st.selectbox("Loan Purpose", list(range(11)),
        format_func=lambda x: {0: "New car", 1: "Used car", 2: "Furniture", 3: "Radio/TV", 4: "Appliances",
            5: "Repairs", 6: "Education", 7: "Vacation", 8: "Retraining", 9: "Business", 10: "Other"}[x])
    employment_duration = st.selectbox("Employment Duration", [1, 2, 3, 4, 5],
        format_func=lambda x: {1: "Unemployed", 2: "< 1 year", 3: "1-4 years", 4: "4-7 years", 5: ">= 7 years"}[x])
    housing = st.selectbox("Housing", [1, 2, 3],
        format_func=lambda x: {1: "Rent", 2: "Own", 3: "Free"}[x])
    job = st.selectbox("Job", [1, 2, 3, 4],
        format_func=lambda x: {1: "Unemployed", 2: "Unskilled", 3: "Skilled", 4: "Management"}[x])

with st.expander("Detalles adicionales"):
    col3, col4 = st.columns(2)
    with col3:
        installment_rate = st.slider("Installment Rate (%)", 1, 4, 3)
        present_residence = st.slider("Present Residence (years)", 1, 4, 3)
        number_credits = st.slider("Number of Credits", 1, 4, 1)
        people_liable = st.selectbox("People Liable", [1, 2])
    with col4:
        personal_status = st.selectbox("Personal Status", [1, 2, 3, 4],
            format_func=lambda x: {1: "Male divorced", 2: "Female", 3: "Male single", 4: "Male married"}[x])
        other_debtors = st.selectbox("Other Debtors", [1, 2, 3],
            format_func=lambda x: {1: "None", 2: "Co-applicant", 3: "Guarantor"}[x])
        property_type = st.selectbox("Property", [1, 2, 3, 4],
            format_func=lambda x: {1: "Real estate", 2: "Savings/Insurance", 3: "Car", 4: "None"}[x])
        other_installments = st.selectbox("Other Installments", [1, 2, 3],
            format_func=lambda x: {1: "Bank", 2: "Stores", 3: "None"}[x])
        telephone = st.selectbox("Telephone", [1, 2], format_func=lambda x: {1: "No", 2: "Yes"}[x])
        foreign_worker = st.selectbox("Foreign Worker", [1, 2], format_func=lambda x: {1: "Yes", 2: "No"}[x])

st.divider()

if st.button("Predecir riesgo", type="primary", width='stretch'):
    row = pd.DataFrame([{
        "status": status, "duration": duration, "credit_history": credit_history,
        "purpose": purpose, "amount": amount, "savings": savings,
        "employment_duration": employment_duration, "installment_rate": installment_rate,
        "personal_status_sex": personal_status, "other_debtors": other_debtors,
        "present_residence_since": present_residence, "property": property_type,
        "age": age, "other_installment_plans": other_installments, "housing": housing,
        "number_credits": number_credits, "job": job, "people_liable": people_liable,
        "telephone": telephone, "foreign_worker": foreign_worker,
    }])
    proba_bad = float(model.predict_proba(row)[0][1])
    pred = int(proba_bad >= threshold)

    st.subheader("Resultado")
    if pred == 0:
        st.success(f"✅ **Buen crédito** — P(malo) = {proba_bad*100:.1f}%  (< umbral {threshold:.2f})")
    else:
        st.error(f"⚠️ **Mal crédito** — P(malo) = {proba_bad*100:.1f}%  (≥ umbral {threshold:.2f})")

    m1, m2 = st.columns(2)
    m1.metric("Prob. buen crédito", f"{(1-proba_bad)*100:.1f}%")
    m2.metric("Prob. mal crédito", f"{proba_bad*100:.1f}%")
    st.progress(min(proba_bad, 1.0))

    # ---- per-applicant risk factors (native TreeSHAP) ----
    try:
        import xgboost as xgb
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pre = model.named_steps["ct"]; clf = model.named_steps["clf"]
        Xt = pre.transform(row)
        try:
            feat = list(pre.get_feature_names_out())
        except Exception:
            feat = [f"f{i}" for i in range(Xt.shape[1])]
        dm = xgb.DMatrix(Xt, feature_names=list(feat))
        contribs = clf.get_booster().predict(dm, pred_contribs=True)[0]
        cdf = (pd.DataFrame({"feature": feat, "contribution": contribs[:-1]})
               .assign(abs=lambda d: d["contribution"].abs())
               .sort_values("abs", ascending=False).head(10))
        st.subheader("Factores de riesgo (SHAP)")
        figc, ax = plt.subplots(figsize=(8, 4))
        bar_colors = [COLORS["danger"] if v > 0 else COLORS["success"]
                      for v in cdf["contribution"][::-1]]
        ax.barh(cdf["feature"][::-1], cdf["contribution"][::-1], color=bar_colors)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("← empuja a Bueno      |      empuja a Malo →")
        figc.tight_layout()
        st.pyplot(figc)
        plt.close(figc)
        st.caption("Rojo = empuja hacia mal crédito; verde = hacia buen crédito.")
    except Exception as e:
        st.caption(f"(Explicación SHAP no disponible: {e})")

st.divider()
st.caption("Fase 6 (Deployment) del proyecto CRISP-ML. El análisis completo "
           "(Fases 1–5) está en `notebooks/CRISP_ML_SouthGermanCredit.ipynb`.")
