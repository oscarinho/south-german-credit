# Post 3 — Mi modelo era 77% accurate. También era injusto.

**Hook visual sugerido:** el gráfico de fairness por subgrupo (celda 44 del notebook) o una tabla con los TPR por grupo.

---

Mi modelo de credit risk tenía métricas decentes:

- Accuracy: 0.77
- Recall: 0.84 (después del threshold tuning)
- ROC AUC: 0.78

Antes de declararlo "listo para producción", corrí una auditoría de fairness por subgrupo. Los resultados me hicieron parar.

El dataset tiene una variable sensible: `personal_status_sex`. Calculé True Positive Rate (capacidad de detectar malos créditos) por subgrupo:

→ Mujeres: TPR 0.71
→ Hombres casados: TPR 0.56
→ Hombres solteros: TPR **0.38**

El modelo detecta el doble de malos créditos en mujeres que en hombres solteros. **Equalized odds gap: 0.62**.

Disparate impact ratio (selection rate del grupo más afectado / grupo de referencia): **0.41**.

La regla del 80% del EEOC marca 0.80 como umbral. Estamos en la mitad de eso.

Tres lecturas posibles:

**a)** El dataset histórico refleja decisiones de aprobación sesgadas, y el modelo está aprendiendo ese sesgo. (Muy probable.)

**b)** Los subgrupos tienen tamaños muy distintos en test (n=84 hombres solteros vs n=42 mujeres), y la métrica es ruidosa.

**c)** Hay una feature en el modelo que actúa como proxy. Por ejemplo: `telephone` aparece como #2 en permutation importance. En 1990s Germany, tener teléfono correlacionaba con ingresos, empleo estable y residencia. Una feature como esa probablemente está empujando el resultado en una dirección que no es causal.

Las 3 lecturas pueden ser ciertas a la vez. Y ninguna se resuelve con `pip install more_data`.

Lo que no haría:

- Reportar el accuracy promedio sin desglose por subgrupo.
- Lanzar a producción sin la auditoría documentada.
- Asumir que "los datos no mienten" cuando los datos son el espejo de decisiones humanas anteriores.

Lo que sí haría:

- Removería `telephone` y otras features que puedan ser proxies socioeconómicos.
- Aplicaría post-processing (`fairlearn.postprocessing.ThresholdOptimizer`) para igualar TPR entre subgrupos.
- Re-correría el audit en cada retraining como gate obligatorio.

El modelo más interesante no es el que tiene mejor AUC. Es el que documenta sus límites.

Repo completo con la auditoría, SHAP, calibración y el Streamlit demo: [link]

#MachineLearning #ResponsibleAI #Fairness #CreditRisk #DataScience

---

**Variaciones de hook:**

A) "El modelo más peligroso es el que parece bueno en métricas agregadas."
B) "Mi modelo aprobó al 71% de los créditos malos entre hombres solteros. Y nadie lo habría visto en el dashboard."
C) "Tres lecturas posibles. Ninguna se arregla con más datos."

---

**Notas para Oscar:**

- Este es el post más diferenciador de los tres. Los otros dos hablan de técnica; este habla de juicio. Es lo que separa un perfil de "data scientist más" de un perfil senior.
- El dato del 0.41 disparate impact es crudo y memorable. Úsalo en el primer scroll.
- Si quieres bajar la intensidad: cambia "injusto" por "no equitativo entre subgrupos" en el título.
- Si quieres subirla: pega el screenshot del gráfico con los TPR side-by-side. Es difícil de ignorar.
