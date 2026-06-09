# Post 1 — El threshold default está costando dinero

**Hook visual sugerido:** screenshot del cost-curve plot (celda 51 del notebook) o un side-by-side de la matriz de confusión @ 0.5 vs @ 0.235.

---

Entrené un modelo de credit risk con buenas métricas:

- Accuracy: 0.77
- ROC AUC: 0.78
- F1: 0.59

Sonaba listo para producción. Hasta que miré la matriz de confusión.

A threshold default 0.5, el modelo detectaba solo el 29% de los créditos malos. 32 falsos negativos sobre 45 casos reales de impago.

En crédito, un FN no cuesta lo mismo que un FP. La documentación del South German Credit dataset usa una ratio de costo 5:1 (un mal cliente aprobado cuesta 5× más que un buen cliente rechazado).

Apliqué cost-aware threshold tuning. Recorrí thresholds de 0.01 a 0.99 y minimicé:

`cost = 5 × FN + 1 × FP`

Resultado en threshold = 0.235:

- Recall: 0.29 → **0.84**
- FN: 32 → 7
- Costo total: 168 → **79** (-53%)
- Precision baja de 0.62 → 0.46 (es el trade-off correcto)

El mismo modelo. Sin reentrenar. Solo cambiando el umbral de decisión.

Lecciones:

1) `model.predict()` con threshold implícito 0.5 es una decisión de negocio camuflada de default técnico.

2) La métrica que importa es la que refleja el costo real, no la que se ve bien en el README.

3) Antes de tunear hiperparámetros, tunea el threshold. Es la mejora más barata que vas a hacer.

Repo completo (notebook CRISP-ML, modular code en `app/`, Streamlit demo): [link]

#MachineLearning #DataScience #CreditRisk #MLOps

---

**Variaciones de hook para A/B:**

A) "Mi modelo tenía 77% de accuracy. Y aprobaba el 71% de los créditos malos."
B) "Un cambio de 1 línea me ahorró 53% del costo del modelo."
C) "Llevo 6 años en data. Sigo viendo modelos en producción con threshold 0.5 sin justificación. Esto es lo que pasa cuando lo cuestionas."
