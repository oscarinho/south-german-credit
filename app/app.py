"""
South German Credit — CRISP-ML Dashboard + Predictor (Streamlit)

A multi-page app that mirrors the CRISP-ML notebook:
Business → Data → Modeling → Evaluation → Fairness → live Predictor.

Heavy results are precomputed by ``app/precompute.py`` into ``app/artifacts/``
and loaded instantly here. The predictor runs live against the trained model.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="South German Credit — CRISP-ML",
                   page_icon="💳", layout="wide")

ROOT = Path(__file__).resolve().parent.parent
ART = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ROOT / "models" / "best_model.joblib"

COLORS = {
    "ice_silver": "#E6E8EB", "graphite": "#2A3038", "espresso_gold": "#C9A86A",
    "graphite_deep": "#240338", "slate": "#424A53", "pebble": "#5E757D",
    "mist": "#B0B4B8", "silver": "#D5D6DB", "platinum": "#EBECEF",
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
    [data-testid="stSidebar"] {{ background-color: {COLORS['graphite']} !important; }}
    [data-testid="stSidebar"] * {{ color: {COLORS['ice_silver']} !important; }}
    .stMetric {{ background-color: white; padding: 1rem; border-radius: 8px;
        border: 1px solid {COLORS['mist']}; box-shadow: 0 2px 8px rgba(36,3,56,0.08); }}
    .stMetric label {{ color: {COLORS['pebble']} !important; }}
    .stMetric [data-testid="stMetricValue"] {{ color: {COLORS['graphite']} !important; }}
</style>
""", unsafe_allow_html=True)


# ----------------------------- data loaders -----------------------------
@st.cache_data
def load_results():
    with open(ART / "results.json") as f:
        return json.load(f)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def fig(name, caption=None):
    p = ART / name
    if p.exists():
        st.image(str(p), width='stretch', caption=caption)
    else:
        st.warning(f"Missing artifact: {name}. Run `python -m app.precompute`.")


def df_from(records):
    return pd.DataFrame(records)


R = load_results() if (ART / "results.json").exists() else None


# ----------------------------- sidebar nav -----------------------------
st.sidebar.markdown("## 💳 South German Credit")
st.sidebar.caption("CRISP-ML dashboard")
PAGE = st.sidebar.radio("Navegación", [
    "🏠 Resumen",
    "📊 Datos (EDA)",
    "🤖 Modelado",
    "🎯 Evaluación",
    "⚖️ Fairness",
    "💳 Predictor",
])
st.sidebar.divider()
st.sidebar.caption("Modelo: XGBoost + SMOTETomek\nMétricas = CV estable (15 folds)")

if R is None:
    st.title("Artefactos no encontrados")
    st.error("No existe `app/artifacts/results.json`. Genera los artefactos con:\n\n"
             "`python -m app.precompute`")
    st.stop()


# ============================== PAGE: RESUMEN ==============================
if PAGE == "🏠 Resumen":
    st.title("Predicción de riesgo crediticio — South German Credit")
    st.markdown(
        "Proyecto de ML end-to-end siguiendo **CRISP-ML**. Objetivo: clasificar "
        "solicitantes como **buen (0)** o **mal (1)** crédito, en un contexto "
        "desbalanceado (70/30) donde detectar la clase minoritaria es lo crítico.")

    cv = R["cv_final"]
    st.subheader("Métrica principal — estimado CV estable (15 folds)")
    c = st.columns(5)
    for col, m, label in zip(c, ["accuracy", "precision", "recall", "f1", "roc_auc"],
                             ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]):
        col.metric(label, f"{cv[m]['mean']:.3f}", f"± {cv[m]['std']:.3f}",
                   delta_color="off")

    st.info(
        f"**Por qué CV y no un solo test:** el test es de solo "
        f"{R['splits']['test']} filas (alta varianza). Re-sorteando el split, el "
        f"recall se mueve entre {R['split_variance']['p5']:.2f} y "
        f"{R['split_variance']['p95']:.2f}. El valor del ejercicio académico "
        f"(0.556) cae en el percentil ~{R['split_variance']['master_percentile']:.0f} "
        f"de esa distribución — un sorteo favorable, no un modelo distinto.")

    st.subheader("Hallazgos clave")
    st.markdown(
        "- **El accuracy engaña** en datos desbalanceados: sin resampling, recall de "
        "la clase mala cae hasta ~0.08.\n"
        "- **SMOTETomek** da el mejor balance recall / estabilidad train-test.\n"
        "- **Drivers principales** (SHAP & permutación): `status` de cuenta, historial "
        "de crédito, `duration` y `amount`.\n"
        "- **El umbral 0.50 es incorrecto para crédito**: con FN 5× más costoso que FP, "
        f"el umbral óptimo baja a **{R['cost_threshold']['threshold']:.2f}** "
        f"(recall {R['cost_threshold']['recall_at_opt']:.2f}).\n"
        "- **Fairness**: existen brechas por subgrupo — auditoría obligatoria antes de desplegar.")


# ============================== PAGE: EDA ==============================
elif PAGE == "📊 Datos (EDA)":
    st.title("📊 Entendimiento de los datos")
    t = R["target"]
    c = st.columns(3)
    c[0].metric("Buen crédito (0)", t["good"])
    c[1].metric("Mal crédito (1)", t["bad"])
    c[2].metric("Ratio de desbalance", f"{R['imbalance_ratio']:.2f} : 1")
    st.caption(f"Partición 70/15/15 → train {R['splits']['train']} · "
               f"val {R['splits']['val']} · test {R['splits']['test']}")

    st.subheader("Distribución del target y numéricas")
    fig("fig_eda.png")
    st.markdown("`duration` y `amount` tienen sesgo a la derecha → se transforman con "
                "Yeo-Johnson + StandardScaler. `age` solo se escala.")

    st.subheader("Tasa de mal crédito por categoría")
    fig("fig_badrate.png", "Línea punteada = tasa global de mal crédito")
    st.markdown("El `status` de la cuenta y el `credit_history` separan muy bien el "
                "riesgo — anticipan su peso en el modelo.")


# ============================== PAGE: MODELADO ==============================
elif PAGE == "🤖 Modelado":
    st.title("🤖 Modelado")
    st.markdown("7 algoritmos × estrategias de resampling, con "
                "`RepeatedStratifiedKFold(5×3)`. El resampler va **dentro** del pipeline "
                "de imblearn → solo toca el fold de entrenamiento (sin fuga).")

    st.subheader("1) Baseline — sin resampling")
    st.dataframe(df_from(R["baseline_table"]), width='stretch', hide_index=True)
    fig("fig_baseline_box.png")
    st.markdown("Varios modelos logran accuracy alto con **recall pésimo** — la trampa "
                "clásica del desbalance.")

    st.subheader("2) Comparación de resamplers (XGBoost afinado)")
    st.dataframe(df_from(R["resampler_table"]), width='stretch', hide_index=True)
    fig("fig_resampler_box.png")

    st.subheader("3) Todos los modelos × SMOTETomek")
    st.dataframe(df_from(R["smotetomek_table"]), width='stretch', hide_index=True)
    fig("fig_smotetomek_box.png")
    st.markdown("Se elige **XGBoost + SMOTETomek** por la menor brecha train/test "
                "(estabilidad en producción), no solo por el recall puntual.")


# ============================== PAGE: EVALUACIÓN ==============================
elif PAGE == "🎯 Evaluación":
    st.title("🎯 Evaluación del modelo final")
    cv = R["cv_final"]; tm = R["test_metrics"]; sv = R["split_variance"]

    st.subheader("Estimado estable (CV) vs un solo test sembrado")
    comp = pd.DataFrame({
        "Métrica": ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
        "CV (media ± sd)": [f"{cv[m]['mean']:.3f} ± {cv[m]['std']:.3f}"
                            for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]],
        "Test (seed=42)": [f"{tm[m]:.3f}" for m in
                           ["accuracy", "precision", "recall", "f1", "roc_auc"]],
    })
    st.dataframe(comp, width='stretch', hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Matriz de confusión (test)**")
        fig("fig_confusion.png")
    with col2:
        st.markdown("**Curva ROC (test)**")
        fig("fig_roc.png")

    st.subheader("Varianza del split — reconciliación con el ejercicio académico")
    fig("fig_split_variance.png")
    st.info(
        f"Recall sobre 100 re-sorteos: media **{sv['mean']:.3f}** ± {sv['std']:.3f} "
        f"(rango {sv['p5']:.2f}–{sv['p95']:.2f}). El 0.556 del académico está en el "
        f"percentil ~{sv['master_percentile']:.0f}: dentro de la varianza normal.")

    st.subheader("Importancia de variables")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Permutación (test, caída de F1)**")
        fig("fig_perm.png")
    with col4:
        st.markdown("**SHAP (mean |SHAP|)**")
        fig("fig_shap_bar.png")
    st.caption("Nota: `telephone_2` aparece alto pero es probablemente un proxy "
               "socioeconómico, no un driver causal — bandera para uso en producción.")

    st.subheader("Umbral de decisión sensible al costo (FN : FP = 5 : 1)")
    ct = R["cost_threshold"]
    c = st.columns(4)
    c[0].metric("Umbral óptimo", f"{ct['threshold']:.2f}", f"def. {ct['default_threshold']}")
    c[1].metric("Recall @ óptimo", f"{ct['recall_at_opt']:.3f}")
    c[2].metric("Precision @ óptimo", f"{ct['precision_at_opt']:.3f}")
    c[3].metric("F1 @ óptimo", f"{ct['f1_at_opt']:.3f}")
    fig("fig_threshold.png")


# ============================== PAGE: FAIRNESS ==============================
elif PAGE == "⚖️ Fairness":
    st.title("⚖️ Auditoría de equidad")
    st.markdown("Métricas por subgrupo y **disparate impact** (regla del 80% del EEOC: "
                "un ratio < 0.80 es bandera roja). Atributos sensibles en este dataset: "
                "`personal_status_sex` y `foreign_worker`.")
    for attr, block in R["fairness"].items():
        st.subheader(f"`{attr}`")
        st.dataframe(df_from(block["table"]), width='stretch', hide_index=True)
        di = block["min_disparate_impact"]
        flag = "🔴 bandera" if di < 0.80 else "🟢 ok"
        c = st.columns(3)
        c[0].metric("Disparate impact (mín.)", f"{di:.2f}", flag, delta_color="off")
        c[1].metric("Brecha TPR", f"{block['tpr_gap']:.3f}")
        c[2].metric("Brecha FPR", f"{block['fpr_gap']:.3f}")
    st.warning("El test es pequeño, así que esto es ilustrativo. En producción la "
               "auditoría debe correrse sobre un holdout mayor y de forma continua.")


# ============================== PAGE: PREDICTOR ==============================
elif PAGE == "💳 Predictor":
    st.title("💳 Predictor de riesgo crediticio")
    model = load_model()
    ct = R["cost_threshold"]

    mode = st.radio("Umbral de decisión", [
        "Por defecto (0.50)",
        f"Cost-óptimo ({ct['threshold']:.2f}) — FN 5× más costoso, mayor recall",
    ], index=1)
    threshold = 0.50 if mode.startswith("Por defecto") else ct["threshold"]

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
            st.success(f"✅ **Buen crédito** — P(malo) = {proba_bad*100:.1f}% "
                       f"(< umbral {threshold:.2f})")
        else:
            st.error(f"⚠️ **Mal crédito** — P(malo) = {proba_bad*100:.1f}% "
                     f"(≥ umbral {threshold:.2f})")
        st.progress(min(proba_bad, 1.0))

        # ---- live per-applicant SHAP contributions (native TreeSHAP) ----
        try:
            import xgboost as xgb
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from app.preprocessing import get_feature_names

            pre = model.named_steps["ct"]; clf = model.named_steps["clf"]
            Xt = pre.transform(row)
            feat = get_feature_names(pre) or [f"f{i}" for i in range(Xt.shape[1])]
            dm = xgb.DMatrix(Xt, feature_names=list(feat))
            contribs = clf.get_booster().predict(dm, pred_contribs=True)[0]
            cdf = (pd.DataFrame({"feature": feat, "contribution": contribs[:-1]})
                   .assign(abs=lambda d: d["contribution"].abs())
                   .sort_values("abs", ascending=False).head(10))
            st.subheader("¿Por qué? — top contribuciones (SHAP)")
            figc, ax = plt.subplots(figsize=(8, 4))
            bar_colors = [COLORS["danger"] if v > 0 else COLORS["success"]
                          for v in cdf["contribution"][::-1]]
            ax.barh(cdf["feature"][::-1], cdf["contribution"][::-1], color=bar_colors)
            ax.axvline(0, color="black", lw=0.8)
            ax.set_xlabel("← empuja a Bueno      |      empuja a Malo →")
            ax.set_title("Contribución por variable (espacio margin)")
            figc.tight_layout()
            st.pyplot(figc)
            plt.close(figc)
            st.caption("Rojo = empuja hacia mal crédito; verde = hacia buen crédito.")
        except Exception as e:
            st.caption(f"(Explicación SHAP no disponible: {e})")
