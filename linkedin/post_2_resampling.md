# Post 2 — Probé 4 técnicas de resampling. Una falló silenciosamente.

**Hook visual sugerido:** el boxplot de comparación de resamplers (celda 16 del notebook), o una tabla con los Recall side-by-side.

---

Dataset desbalanceado: 70% buenos créditos, 30% malos.

Sin resampling, mi Random Forest dio accuracy de 0.71. Sonaba bien hasta que vi el recall: **0.08**.

El modelo detectaba el 8% de los créditos malos. El otro 92% pasaba como bueno. Pero como la clase mayoritaria es "bueno", la accuracy se veía decente.

Probé 4 estrategias para corregirlo:

→ RandomUnderSampler (descartar mayoritarios)
→ SMOTE (sintetizar minoritarios)
→ KMeansSMOTE (SMOTE dentro de clusters)
→ SMOTETomek (SMOTE + limpieza de borde)

Resultados con XGBoost tuneado (15 folds, CV repetido):

| Resampler | Accuracy | Recall | F1 |
|---|---|---|---|
| None | 0.74 | 0.50 | 0.54 |
| RandomUnder | 0.67 | **0.70** | 0.56 |
| SMOTE | 0.73 | 0.49 | 0.52 |
| KMeansSMOTE | NaN ❌ | NaN | NaN |
| SMOTETomek | 0.74 | 0.48 | 0.53 |

Tres hallazgos no obvios:

**1. KMeansSMOTE falló silenciosamente.** Con n=850 y k_neighbors=5, los clusters de la minoría quedaban demasiado pequeños y KMeansSMOTE devolvía NaN sin lanzar excepción. Ajusté `k_neighbors=3` y `cluster_balance_threshold=0.1` para hacerlo funcionar. Si solo hubiera leído la celda final, habría asumido que "no funciona en este dataset" — pero el problema era de configuración, no del método.

**2. RandomUnderSampler ganó en Recall**, pero a costa de descartar el 60% del training. En un dataset chico (1000 filas) eso es brutal.

**3. SMOTETomek no fue el ganador por métrica cruda**, pero sí por estabilidad: el gap entre train y test era el más pequeño. Eso es lo que importa para producción.

Para la versión final del modelo combiné SMOTETomek + XGBoost tuneado por GridSearch (256 combinaciones × 15 folds). El gap train/test quedó por debajo de 3 puntos.

Lección: la "mejor" técnica de resampling no es la que da más recall en CV — es la que generaliza mejor cuando el modelo vea data nueva.

Notebook completo con el GridSearch, las 4 técnicas y las matrices de confusión: [link]

#MachineLearning #ImbalancedData #SMOTE #DataScience

---

**Variaciones de hook:**

A) "Mi modelo tenía 71% accuracy. Detectaba el 8% de los malos."
B) "4 técnicas de resampling. 1 ganó por una razón que no es la que crees."
C) "El método más popular falló sin avisar. Tres horas debugging después..."
