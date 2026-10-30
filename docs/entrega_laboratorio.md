# Laboratorio 3: organización de la entrega

## Diseño experimental

El **modelo base o de referencia es Full**, la adaptación completa del método del artículo al problema de reidentificación de personas. La **ablación es Sin MIM**, que elimina el objetivo de reconstrucción enmascarada y se entrena por separado. Se mantienen el agrupamiento global y de parches, ME-MAX y el mecanismo maestro–estudiante.

Full conserva los componentes centrales del método original, con las diferencias de entradas, número de vistas, prototipos y decodificador explicadas en el README. No se presenta como una reproducción literal del artículo. El nombre interno `baseline` de otra variante disponible en el código no identifica el modelo de referencia de esta entrega.

## Correspondencia con los requisitos

| Requisito | Código o evidencia |
|---|---|
| Carga de datos y selección de particiones | `src/maskcluster_ssl/data/hdf5.py`, `scripts/train.py` y `scripts/evaluate.py`. El entrenamiento lee `training.h5`; la validación selecciona y la prueba evalúa. |
| Preprocesamiento y transformaciones | `src/maskcluster_ssl/data/transforms.py`: vistas globales/locales, normalización, fotometría y máscaras. El evaluador distingue consultas originales y reducidas con la misma galería. |
| Implementación del método | `src/maskcluster_ssl/models/vit.py`, `models/maskcluster.py` y `losses/maskcluster.py`: ViT, maestro EMA, agrupamiento y reconstrucción. |
| Entrenamiento y validación | `scripts/train.py` y `training/trainer.py`. Configuraciones, resúmenes y metadatos efectivos en `docs/experimentos/full/` y `docs/experimentos/no_mim/`. |
| Evaluación del modelo base | Full: evaluación de maestro y estudiante en las dos resoluciones. Métricas en `docs/experimentos/full/metrics_teacher.json` y `metrics_student.json`. |
| Experimento de ablación | Sin MIM frente a Full, con configuración común y entrenamiento independiente. Métricas en `docs/experimentos/no_mim/`. |
| Evaluación en ambas condiciones y cálculo de métricas | `scripts/evaluate.py`, `evaluation/extract.py` y `evaluation/reid.py`. mAP, Rank-1/5/10 y caída de mAP. |
| Gráficos y resultados en Markdown | [Informe experimental](resultados_experimento.md), tablas del README y figuras en `assets/figures/`, con ejes etiquetados. |
| Instrucciones breves de ejecución | Sección «Uso del proyecto» del README. Las configuraciones efectivas registran los valores de las ejecuciones descritas. |
| Diferencias respecto del artículo | README e informe: entrada rectangular, dimensiones de vistas, prototipos, presupuesto, decoder MLP, L1 media y atención visible densa. |
| Cinco aciertos y cinco fallos, con explicación | [Full](resultados_full.md) y [Sin MIM](resultados_no_mim.md): cinco aciertos y cinco fallos por resolución y por modelo, cuarenta paneles en total. |

## Selección y evaluación

Se completaron 30 épocas por modelo sobre un calendario previsto de 40. La validación seleccionó al estudiante de cada modelo, con los mejores pesos de la época 25. La evaluación final utilizó el mismo checkpoint por codificador para ambas resoluciones, representaciones normalizadas L2 y similitud coseno. La prueba no actualizó los pesos.

También se reportaron los maestros para documentar el mecanismo de transferencia. Los cuarenta paneles explicados se concentran en los estudiantes elegidos por validación. Los resultados cuantitativos incluyen los cuatro codificadores.

## Lectura de los resultados

Sin MIM obtuvo resultados ligeramente mejores que Full en las cuatro métricas y las dos resoluciones de consulta para los estudiantes. La resolución reducida perjudicó las métricas agregadas de ambos modelos. Las pérdidas descendieron, pero la precisión de recuperación siguió siendo baja; no se observó una mejora atribuible al objetivo de reconstrucción en estas ejecuciones.

Las observaciones de errores distinguen los datos verificables —corrección en primera posición y puesto de la primera coincidencia— de las interpretaciones visuales de vestimenta, detalle y punto de vista. No se generaliza desde los ejemplos seleccionados ni se afirma significancia estadística con una sola semilla.

## Material de referencia

`Maskcluster_architecture.html` explica el método del artículo original y sus tareas de transferencia. No es un registro de las corridas de laboratorio ni sustituye el informe experimental.
