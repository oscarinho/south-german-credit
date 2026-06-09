# Serie LinkedIn — South German Credit

3 posts encadenados sobre el proyecto. Cada uno se sostiene solo pero ganan publicados en orden a lo largo de 1-2 semanas.

| Post | Ángulo | Hallazgo clave | Audiencia |
|---|---|---|---|
| 1 | Threshold tuning | Recall 29% → 84%, costo -53% solo cambiando umbral | Practitioners + hiring managers técnicos |
| 2 | Resampling deep-dive | 4 técnicas comparadas, KMeansSMOTE falla silenciosa, SMOTETomek gana por estabilidad no por métrica | Audiencia técnica (data + ML) |
| 3 | Fairness audit | TPR gap 0.62 entre subgrupos, disparate impact 0.41 | Audiencia amplia (líderes, recruiters, responsible AI) |

## Orden y cadencia recomendados

- **Lunes** — Post 1 (gancho cuantitativo, accesible)
- **Jueves** — Post 2 (técnico, para tu audiencia de pares)
- **Lunes siguiente** — Post 3 (reflexión, mayor alcance)

## Asets visuales por post

| Post | Asset | Origen |
|---|---|---|
| 1 | Cost-curve plot | Celda 51 del notebook |
| 1 | Matriz de confusión @ 0.5 vs @ 0.235 | Reconstruir desde valores: 32→7 FN, 8→44 FP |
| 2 | Boxplot resamplers | Celda 16 del notebook |
| 2 | Tabla CV results con NaN destacado | Celda 15 output |
| 3 | Bar plot TPR por subgrupo | Celda 44 del notebook |
| 3 | Tabla disparate impact | Celda 42 output |

## Tono y estilo

- **No** emojis (puedes agregar uno o dos en el cierre si tu marca personal lo lleva)
- **Sí** números crudos arriba — los algoritmos de LinkedIn premian especificidad
- Bullets cortos, líneas en blanco generosas (LinkedIn corta el feed agresivamente)
- Cierre siempre con link al repo

## CTA secundario

Después del post 3, ofrece en comentarios el notebook completo. Eso convierte engagement en visitas reales al repo y, eventualmente, en mensajes directos.
