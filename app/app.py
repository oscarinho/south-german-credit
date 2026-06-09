"""
South German Credit Risk — Streamlit Deployment App (CRISP-ML Phase 6)

Real-time credit-risk scoring against the trained pipeline
(`model/best_model.joblib`). Editorial-terminal UI. The full CRISP-ML
analysis (Phases 1–5) lives in the notebook; this app is the deployment layer.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Credit Risk Intelligence | CRISP-ML",
                   page_icon="💳", layout="wide", initial_sidebar_state="expanded")

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model" / "best_model.joblib"
META_PATH = ROOT / "model" / "model_metadata.pkl"

# ---- palette ----
INK, GOLD, GOLD_DK = "#1A1D23", "#C9A86A", "#8B7340"
SUCCESS, WARNING, DANGER, INFO = "#43936C", "#F2AE4A", "#D96B5F", "#4A67B0"
GRAPHITE_DEEP, SILVER, PEBBLE = "#240338", "#D5D6DB", "#5E757D"

# ---------------------------------------------------------------- styles
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=IBM+Plex+Mono:wght@300;400;500;600&family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;1,9..144,400;1,9..144,500&display=swap');

:root{
  --ink:#1A1D23; --ink-soft:rgba(26,29,35,0.55); --ink-faint:rgba(26,29,35,0.32);
  --rule:rgba(26,29,35,0.12); --gold-soft:rgba(201,168,106,0.09); --gold-mid:rgba(201,168,106,0.32);
}
#MainMenu, footer, header { visibility:hidden; }
html, body, [data-testid="stAppViewContainer"]{
  background:
    radial-gradient(ellipse 80% 50% at top left, rgba(201,168,106,0.07) 0%, transparent 55%),
    radial-gradient(ellipse 80% 50% at bottom right, rgba(36,3,56,0.05) 0%, transparent 55%),
    linear-gradient(180deg,#FAFAF7 0%,#F3F2EE 100%);
  min-height:100vh; font-family:'IBM Plex Mono',monospace; color:var(--ink);
}
[data-testid="stAppViewContainer"]::before{
  content:''; position:fixed; inset:0; pointer-events:none; z-index:0;
  background-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='180' height='180'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2' seed='3'/><feColorMatrix values='0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0.05 0'/></filter><rect width='100%' height='100%' filter='url(%23n)'/></svg>");
  mix-blend-mode:multiply;
}
.block-container{ padding-top:1.5rem !important; }

[data-testid="stSidebar"]{ background:linear-gradient(180deg,#20242C 0%,#161A21 100%) !important; border-right:none; position:relative; }
[data-testid="stSidebar"]::after{ content:''; position:absolute; top:0; right:0; bottom:0; width:2px;
  background:linear-gradient(180deg,transparent 0%,#C9A86A 22%,#C9A86A 78%,transparent 100%); }
[data-testid="stSidebar"] *{ color:#E6E8EB !important; }
[data-testid="stSidebar"] label{ font-family:'Fraunces',Georgia,serif !important; font-style:italic !important; }

hr{ border:none; height:1px; background:linear-gradient(90deg,transparent,var(--ink) 25%,var(--ink) 75%,transparent); opacity:0.25; margin:1.5rem 0; }

.issue-strip{ font-family:'IBM Plex Mono',monospace; font-size:0.62rem; color:var(--ink-soft);
  letter-spacing:0.3em; text-transform:uppercase; border-top:1px solid var(--ink); border-bottom:1px solid var(--rule);
  padding:0.55rem 0; margin:0 0 1.3rem; display:flex; justify-content:space-between; flex-wrap:wrap; gap:1rem; }
.issue-strip .dot{ display:inline-block; width:6px; height:6px; background:#43936C; border-radius:50%;
  margin-right:0.5rem; vertical-align:middle; box-shadow:0 0 8px rgba(67,147,108,0.65); animation:pulse 2.2s ease-in-out infinite; }
@keyframes pulse{ 0%,100%{opacity:1;transform:scale(1);} 50%{opacity:0.4;transform:scale(0.9);} }

.main-header{ font-family:'Fraunces',Georgia,serif; font-variation-settings:"opsz" 144; font-size:3.2rem;
  font-weight:600; color:var(--ink); letter-spacing:-0.025em; line-height:1; margin:0.1rem 0; }
.main-header em{ font-style:italic; font-weight:400; color:#8B7340; }
.sub-header{ font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:var(--ink-soft); letter-spacing:0.24em;
  text-transform:uppercase; margin:0.55rem 0 0; padding-bottom:1.3rem; border-bottom:2px solid var(--ink); }

.section-header{ font-family:'Fraunces',Georgia,serif; font-style:italic; font-size:1.3rem; font-weight:500;
  color:var(--ink); margin:1.8rem 0 1rem; display:flex; align-items:baseline; gap:0.75rem; line-height:1.2; }
.section-header::before{ content:''; flex:0 0 24px; height:1px; background:#C9A86A; align-self:center; }
.section-header::after{ content:''; flex:1 1 auto; height:1px; background:var(--rule); align-self:center; }

.metric-card{ background:#FFF; border:1px solid var(--rule); border-radius:0; padding:1.25rem 1.35rem 1rem;
  position:relative; transition:transform .25s ease, box-shadow .25s ease; box-shadow:0 1px 0 rgba(26,29,35,0.04); height:100%; }
.metric-card::before{ content:''; position:absolute; top:-1px; left:-1px; width:14px; height:14px; border-top:1.5px solid #C9A86A; border-left:1.5px solid #C9A86A; }
.metric-card::after{ content:''; position:absolute; bottom:-1px; right:-1px; width:14px; height:14px; border-bottom:1.5px solid #C9A86A; border-right:1.5px solid #C9A86A; }
.metric-card:hover{ transform:translateY(-2px); box-shadow:0 14px 30px rgba(26,29,35,0.08); }
.metric-label{ font-family:'Fraunces',Georgia,serif; font-style:italic; font-size:0.92rem; color:var(--ink-soft); line-height:1.2; }
.metric-value{ font-family:'Orbitron',monospace; font-size:1.9rem; font-weight:700; color:var(--ink); margin:0.4rem 0 0.15rem; letter-spacing:-0.02em; line-height:1; }
.metric-delta{ font-family:'IBM Plex Mono',monospace; font-size:0.62rem; color:var(--ink-faint); letter-spacing:0.16em; text-transform:uppercase; }

.input-card{ background:#FFF; border:1px solid var(--rule); border-radius:0; padding:1.1rem 1.4rem 0.6rem; margin-bottom:0.9rem;
  box-shadow:0 1px 0 rgba(26,29,35,0.04); position:relative; }
.input-card-header{ font-family:'Fraunces',Georgia,serif; font-style:italic; font-size:1.05rem; font-weight:500; color:var(--ink);
  border-bottom:1px solid var(--rule); padding-bottom:0.55rem; margin-bottom:0.4rem; display:flex; align-items:baseline; gap:0.5rem; }
.input-card-header::before{ content:'§'; color:#C9A86A; font-family:'IBM Plex Mono',monospace; font-style:normal; font-weight:600; }

.info-box{ background:var(--gold-soft); border-left:2px solid #C9A86A; border-top:1px solid var(--rule);
  border-right:1px solid var(--rule); border-bottom:1px solid var(--rule); padding:0.95rem 1.1rem;
  font-family:'IBM Plex Mono',monospace; font-size:0.78rem; color:var(--ink); margin:0.5rem 0; line-height:1.65; }

.prediction-box{ background:#FFF; border:1px solid var(--rule); border-radius:0; padding:1.7rem 1.9rem 1.4rem;
  margin:1rem 0; position:relative; box-shadow:0 18px 50px rgba(26,29,35,0.07); }
.prediction-box::before{ content:''; position:absolute; top:0; left:0; right:0; height:4px; }
.prediction-box.success::before{ background:#43936C; }
.prediction-box.warning::before{ background:#F2AE4A; }
.prediction-box.danger::before{ background:#D96B5F; }
.pred-stamp{ position:absolute; top:1rem; right:1.2rem; font-family:'IBM Plex Mono',monospace; font-size:0.55rem;
  letter-spacing:0.32em; color:var(--ink-soft); background:var(--gold-soft); padding:0.25rem 0.6rem; border:1px solid var(--gold-mid); text-transform:uppercase; }
.prediction-label{ font-family:'Fraunces',Georgia,serif; font-style:italic; font-size:1rem; color:var(--ink-soft); margin-bottom:0.2rem; }
.prediction-value{ font-family:'Orbitron',monospace; font-size:3.4rem; font-weight:900; line-height:1; margin:0.2rem 0; letter-spacing:-0.03em; }
.prediction-advisory{ font-family:'IBM Plex Mono',monospace; font-size:0.78rem; margin-top:0.85rem; font-weight:500;
  padding:0.55rem 0.85rem; background:rgba(26,29,35,0.035); border-left:2px solid #C9A86A; }

.sidebar-brand{ text-align:left; padding:0.6rem 0 0.9rem; border-bottom:1px solid rgba(255,255,255,0.08); margin-bottom:1rem; }
.sidebar-brand-mark{ font-family:'Fraunces',Georgia,serif; font-style:italic; font-size:1.5rem; color:#C9A86A; line-height:1.1; font-weight:500; }
.sidebar-brand-mark span{ font-style:normal; color:#FAFAF7; font-weight:600; }
.sidebar-brand-id{ font-family:'IBM Plex Mono',monospace; font-size:0.6rem; color:rgba(176,180,184,0.55) !important; letter-spacing:0.3em; text-transform:uppercase; margin-top:0.55rem; }
.sidebar-pill{ display:inline-flex; align-items:center; gap:0.4rem; font-family:'IBM Plex Mono',monospace; font-size:0.58rem;
  letter-spacing:0.22em; text-transform:uppercase; padding:0.18rem 0.55rem; border:1px solid rgba(201,168,106,0.4); background:rgba(201,168,106,0.08); color:#C9A86A !important; }
.sidebar-pill .pd{ display:inline-block; width:5px; height:5px; background:#43936C; border-radius:50%; box-shadow:0 0 6px rgba(67,147,108,0.7); animation:pulse 2.2s ease-in-out infinite; }

[data-testid="stButton"] > button{ background:var(--ink); color:#FAFAF7; font-family:'IBM Plex Mono',monospace;
  font-size:0.74rem; font-weight:500; letter-spacing:0.24em; text-transform:uppercase; border:1px solid var(--ink);
  border-radius:0; padding:0.78rem 2rem; width:100%; transition:all .18s ease; }
[data-testid="stButton"] > button:hover{ background:#C9A86A; color:var(--ink); border-color:#C9A86A; letter-spacing:0.26em; }
[data-baseweb="select"] > div{ border-radius:0 !important; border-color:var(--rule) !important; }
.stSlider [data-baseweb="slider"] [role="slider"]{ background:#C9A86A !important; border:2px solid var(--ink) !important; }
[data-testid="stPlotlyChart"]{ background:#FFF; border:1px solid var(--rule); padding:0.5rem 0.7rem; box-shadow:0 1px 0 rgba(26,29,35,0.04); }
.colophon{ font-family:'IBM Plex Mono',monospace; font-size:0.6rem; color:var(--ink-faint); letter-spacing:0.22em;
  text-transform:uppercase; text-align:center; margin-top:2rem; padding-top:1rem; border-top:1px solid var(--rule); }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- loaders
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
cv = meta.get("metrics_cv_15fold", {})
ct = meta.get("cost_optimal_threshold", {})
opt_threshold = float(ct.get("threshold", 0.5))


def cvv(metric):
    m = cv.get(metric, {})
    return m.get("mean"), m.get("std")


def metric_card(col, label, value, delta):
    col.markdown(f"""<div class='metric-card'><div class='metric-label'>{label}</div>
        <div class='metric-value'>{value}</div><div class='metric-delta'>{delta}</div></div>""",
        unsafe_allow_html=True)


def section(title):
    st.markdown(f"<div class='section-header'>{title}</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("""<div class='sidebar-brand'>
        <div class='sidebar-brand-mark'>Credit <span>Intel.</span></div>
        <div class='sidebar-brand-id'>v03 · MMXXVI · OP</div>
        <div style='margin-top:0.7rem;'><span class='sidebar-pill'><span class='pd'></span>Model online</span></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header' style='font-size:1rem;margin-top:0.4rem;'>Policy</div>",
                unsafe_allow_html=True)
    options = ["Default (0.50)"]
    if ct:
        options.append(f"Cost-optimal ({opt_threshold:.2f})")
    mode = st.radio("Decision threshold", options, index=1 if ct else 0)
    threshold = 0.50 if mode.startswith("Default") else opt_threshold
    if ct:
        st.markdown(f"""<div class='info-box' style='font-size:0.7rem;'>
            <b>Cost-sensitive policy.</b> A false negative (approving a bad credit) is treated as
            <b>5×</b> costlier than a false positive. The cost-optimal cut-off is
            <b>{opt_threshold:.2f}</b> → recall {ct.get('recall_at_opt',0):.2f},
            precision {ct.get('precision_at_opt',0):.2f}.</div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header' style='font-size:1rem;'>Model</div>", unsafe_allow_html=True)
    st.markdown(f"""<div style='font-family:"IBM Plex Mono";font-size:0.68rem;line-height:1.9;'>
        PIPELINE · {meta.get('model','XGBoost + SMOTETomek')}<br>
        VALIDATION · {meta.get('cv','RepeatedStratifiedKFold(5x3)')}<br>
        SPLIT · {meta.get('split','70/15/15 stratified')}
    </div>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- masthead
st.markdown("""
<div class='issue-strip'>
    <span><span class='dot'></span>LIVE · LAB 01 · CREDIT</span>
    <span>CRISP-ML(Q)</span>
    <span>OSCAR PONCE · MMXXVI</span>
</div>
<div class='main-header'>Credit Risk <em>Intelligence</em></div>
<div class='sub-header'>South German Credit · 1,000 applicants · XGBoost + SMOTETomek · scoring layer</div>
""", unsafe_allow_html=True)

# ---- KPI strip: model performance (stable CV) ----
section("Model performance · 15-fold cross-validation")
k = st.columns(5)
for col, mname, label in zip(
        k, ["recall", "f1", "roc_auc", "precision", "accuracy"],
        ["Recall (bad)", "F1", "ROC-AUC", "Precision", "Accuracy"]):
    mean, std = cvv(mname)
    val = f"{mean:.3f}" if mean is not None else "—"
    delta = f"± {std:.3f}" if std is not None else ""
    metric_card(col, label, val, delta)

st.markdown("""<div class='info-box'>Metrics are the <b>stable mean ± sd over 15 CV folds</b> of the
deployed pipeline — not a single small test draw. Recall on the <i>bad</i> class is the primary KPI:
a missed bad credit (false negative) is the costly error in lending.</div>""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)


# ---------------------------------------------------------------- input form
section("Applicant dossier")
form_col, result_col = st.columns([1.05, 0.95], gap="large")

with form_col:
    st.markdown("<div class='input-card'><div class='input-card-header'>Account & history</div>",
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        status = st.selectbox("Account Status", [1, 2, 3, 4],
            format_func=lambda x: {1: "No account", 2: "No balance", 3: "< 200 DM", 4: ">= 200 DM"}[x])
        savings = st.selectbox("Savings", [1, 2, 3, 4, 5],
            format_func=lambda x: {1: "Unknown/None", 2: "< 100 DM", 3: "100-500 DM", 4: "500-1000 DM", 5: ">= 1000 DM"}[x])
    with c2:
        credit_history = st.selectbox("Credit History", [0, 1, 2, 3, 4],
            format_func=lambda x: {0: "Delay in past", 1: "Critical account", 2: "No credits", 3: "Existing paid", 4: "All paid"}[x])
        number_credits = st.slider("Existing credits", 1, 4, 1)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='input-card'><div class='input-card-header'>Loan</div>", unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        duration = st.number_input("Duration (months)", 1, 72, 24)
        amount = st.number_input("Amount (DM)", 250, 20000, 2500)
        installment_rate = st.slider("Installment rate (%)", 1, 4, 3)
    with c4:
        purpose = st.selectbox("Purpose", list(range(11)),
            format_func=lambda x: {0: "New car", 1: "Used car", 2: "Furniture", 3: "Radio/TV", 4: "Appliances",
                5: "Repairs", 6: "Education", 7: "Vacation", 8: "Retraining", 9: "Business", 10: "Other"}[x])
        other_installments = st.selectbox("Other installment plans", [1, 2, 3],
            format_func=lambda x: {1: "Bank", 2: "Stores", 3: "None"}[x])
        other_debtors = st.selectbox("Other debtors", [1, 2, 3],
            format_func=lambda x: {1: "None", 2: "Co-applicant", 3: "Guarantor"}[x])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='input-card'><div class='input-card-header'>Applicant profile</div>",
                unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        age = st.number_input("Age (years)", 18, 80, 35)
        employment_duration = st.selectbox("Employment", [1, 2, 3, 4, 5],
            format_func=lambda x: {1: "Unemployed", 2: "< 1 year", 3: "1-4 years", 4: "4-7 years", 5: ">= 7 years"}[x])
        job = st.selectbox("Job", [1, 2, 3, 4],
            format_func=lambda x: {1: "Unemployed", 2: "Unskilled", 3: "Skilled", 4: "Management"}[x])
        housing = st.selectbox("Housing", [1, 2, 3],
            format_func=lambda x: {1: "Rent", 2: "Own", 3: "Free"}[x])
        property_type = st.selectbox("Property", [1, 2, 3, 4],
            format_func=lambda x: {1: "Real estate", 2: "Savings/Insurance", 3: "Car", 4: "None"}[x])
    with c6:
        personal_status = st.selectbox("Personal status", [1, 2, 3, 4],
            format_func=lambda x: {1: "Male divorced", 2: "Female", 3: "Male single", 4: "Male married"}[x])
        present_residence = st.slider("Residence (years)", 1, 4, 3)
        people_liable = st.selectbox("People liable", [1, 2])
        telephone = st.selectbox("Telephone", [1, 2], format_func=lambda x: {1: "No", 2: "Yes"}[x])
        foreign_worker = st.selectbox("Foreign worker", [1, 2], format_func=lambda x: {1: "Yes", 2: "No"}[x])
    st.markdown("</div>", unsafe_allow_html=True)

    run = st.button("▸ Score applicant", type="primary")

# ---------------------------------------------------------------- result
with result_col:
    if run:
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

        if pred == 1:
            box_class, color, verdict = "danger", DANGER, "BAD CREDIT"
        elif proba_bad >= threshold - 0.10:
            box_class, color, verdict = "warning", WARNING, "GOOD · BORDERLINE"
        else:
            box_class, color, verdict = "success", SUCCESS, "GOOD CREDIT"

        advisory = ("▸ <b>Decline / review</b> — probability of default is at or above the policy cut-off."
                    if pred == 1 else
                    "✓ <b>Approve</b> — probability of default is below the policy cut-off.")
        adv_border = DANGER if pred == 1 else SUCCESS

        st.markdown(f"""
        <div class='prediction-box {box_class}'>
            <div class='pred-stamp'>CRISP · SCORE v1</div>
            <div class='prediction-label'>Probability of default</div>
            <div class='prediction-value' style='color:{color};'>{proba_bad*100:.1f}%</div>
            <div style='font-family:"IBM Plex Mono";font-size:0.66rem;letter-spacing:0.18em;
                        text-transform:uppercase;color:{color};font-weight:600;'>{verdict}</div>
            <hr style='margin:0.9rem 0 0.7rem;'>
            <div style='font-family:"IBM Plex Mono";font-size:0.74rem;color:var(--ink-soft);line-height:1.7;'>
                Decision threshold · <b style='color:var(--ink);'>{threshold:.2f}</b>
                ({'cost-optimal' if abs(threshold-opt_threshold)<1e-9 else 'default'})<br>
                P(good) {(1-proba_bad)*100:.1f}%  ·  P(bad) {proba_bad*100:.1f}%
            </div>
            <div class='prediction-advisory' style='border-left-color:{adv_border};'>{advisory}</div>
        </div>
        """, unsafe_allow_html=True)

        # gauge
        g = go.Figure(go.Indicator(
            mode="gauge+number", value=proba_bad * 100,
            number={"suffix": "%", "font": {"family": "Orbitron", "size": 30, "color": color}},
            gauge={
                "axis": {"range": [0, 100], "tickfont": {"family": "IBM Plex Mono", "size": 9}},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "white",
                "steps": [{"range": [0, threshold * 100], "color": "rgba(67,147,108,0.14)"},
                          {"range": [threshold * 100, 100], "color": "rgba(217,107,95,0.14)"}],
                "threshold": {"line": {"color": INK, "width": 2}, "thickness": 0.85,
                              "value": threshold * 100},
            }))
        g.update_layout(height=190, margin=dict(l=20, r=20, t=10, b=10),
                        paper_bgcolor="white", font_family="IBM Plex Mono")
        st.plotly_chart(g, use_container_width=True, config={"displayModeBar": False})

        # per-applicant risk factors (native TreeSHAP)
        try:
            import xgboost as xgb
            pre = model.named_steps["ct"]; clf = model.named_steps["clf"]
            Xt = pre.transform(row)
            try:
                feat = list(pre.get_feature_names_out())
            except Exception:
                feat = [f"f{i}" for i in range(Xt.shape[1])]
            dm = xgb.DMatrix(Xt, feature_names=list(feat))
            contribs = clf.get_booster().predict(dm, pred_contribs=True)[0]
            cdf = (pd.DataFrame({"feature": feat, "c": contribs[:-1]})
                   .assign(a=lambda d: d["c"].abs())
                   .sort_values("a", ascending=False).head(8).iloc[::-1])
            bar = go.Figure(go.Bar(
                x=cdf["c"], y=cdf["feature"], orientation="h",
                marker_color=[DANGER if v > 0 else SUCCESS for v in cdf["c"]]))
            bar.update_layout(
                height=290, margin=dict(l=10, r=10, t=34, b=10),
                title=dict(text="WHY · top risk factors", font=dict(family="Orbitron", size=12, color=GRAPHITE_DEEP)),
                plot_bgcolor="white", paper_bgcolor="white", font_family="IBM Plex Mono",
                xaxis=dict(title="← pushes to Good   |   pushes to Bad →", gridcolor=SILVER, zerolinecolor=INK),
                yaxis=dict(gridcolor="white"))
            st.plotly_chart(bar, use_container_width=True, config={"displayModeBar": False})
        except Exception as e:
            st.caption(f"(SHAP explanation unavailable: {e})")
    else:
        st.markdown("""<div class='input-card' style='min-height:320px;display:flex;flex-direction:column;
            align-items:center;justify-content:center;text-align:center;'>
            <div style='font-family:"Fraunces";font-style:italic;font-size:1.2rem;color:var(--ink-soft);'>Awaiting input</div>
            <div style='font-family:"IBM Plex Mono";font-size:0.68rem;letter-spacing:0.2em;text-transform:uppercase;
                color:var(--ink-faint);margin-top:0.6rem;'>Fill the dossier · press Score</div>
        </div>""", unsafe_allow_html=True)

st.markdown("""<div class='colophon'>South German Credit · CRISP-ML(Q) · Phase 6 Deployment ·
Full analysis in notebooks/CRISP_ML_SouthGermanCredit.ipynb</div>""", unsafe_allow_html=True)
