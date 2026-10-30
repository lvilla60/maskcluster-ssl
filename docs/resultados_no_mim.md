# Recuperaciones de prueba: Ablación — Sin MIM

## Procedencia y lectura de los paneles

Estos veinte ejemplos corresponden al **estudiante de Ablación — Sin MIM**, seleccionado mediante validación y evaluado en `testing.h5`. Los pesos seleccionados son de la época 25 de una ejecución de 30 épocas. Se muestran cinco aciertos y cinco fallos en cada resolución.

Cada panel contiene una consulta y cuatro resultados de galería, de izquierda a derecha y en orden de similitud. No es una secuencia de video. `+` indica la misma identidad según las etiquetas de evaluación y `−` una identidad diferente. La similitud coseno no es una probabilidad de acierto.

Un acierto significa que el primer candidato es correcto (Rank-1); no exige que los cuatro sean correctos. Un fallo significa que el primero es incorrecto; la persona buscada puede aparecer más abajo. La posición de la primera coincidencia procede del registro de evaluación y puede superar los cuatro resultados visibles.

Las consultas originales tienen 128 × 64 píxeles. Las consultas reducidas parten de 32 × 16 y se redimensionan al tamaño de entrada del codificador; la galería permanece en resolución original. Ambos grupos utilizan los mismos pesos.

Los casos son los primeros disponibles de cada categoría según el orden de consultas: no son una muestra aleatoria ni un cálculo adicional de las métricas. Los grupos de aciertos originales y reducidos pueden contener consultas distintas. Las observaciones visuales describen patrones compatibles con los errores, sin atribuir causalidad a un color, objeto o fondo.

## Resolución original: cinco aciertos

### Acierto 1 · consulta 9

![Ablación — Sin MIM: resolución original, acierto 1](../assets/results/no_mim/original_success_01.png)

**Registro:** primera coincidencia correcta en el puesto **1**; similitud del primer resultado **0.9580**.

La primera imagen recuperada corresponde a la misma identidad y a una cámara diferente. Es una recuperación correcta en Rank-1, Rank-5 y Rank-10; la presencia de otros candidatos incorrectos no invalida este acierto.

### Acierto 2 · consulta 16

![Ablación — Sin MIM: resolución original, acierto 2](../assets/results/no_mim/original_success_02.png)

**Registro:** primera coincidencia correcta en el puesto **1**; similitud del primer resultado **0.9753**.

La primera imagen recuperada corresponde a la misma identidad y a una cámara diferente. Es una recuperación correcta en Rank-1, Rank-5 y Rank-10; la presencia de otros candidatos incorrectos no invalida este acierto.

### Acierto 3 · consulta 25

![Ablación — Sin MIM: resolución original, acierto 3](../assets/results/no_mim/original_success_03.png)

**Registro:** primera coincidencia correcta en el puesto **1**; similitud del primer resultado **0.9543**.

La primera imagen recuperada corresponde a la misma identidad y a una cámara diferente. Es una recuperación correcta en Rank-1, Rank-5 y Rank-10; la presencia de otros candidatos incorrectos no invalida este acierto.

### Acierto 4 · consulta 59

![Ablación — Sin MIM: resolución original, acierto 4](../assets/results/no_mim/original_success_04.png)

**Registro:** primera coincidencia correcta en el puesto **1**; similitud del primer resultado **0.9749**.

La primera imagen recuperada corresponde a la misma identidad y a una cámara diferente. Es una recuperación correcta en Rank-1, Rank-5 y Rank-10; la presencia de otros candidatos incorrectos no invalida este acierto.

### Acierto 5 · consulta 65

![Ablación — Sin MIM: resolución original, acierto 5](../assets/results/no_mim/original_success_05.png)

**Registro:** primera coincidencia correcta en el puesto **1**; similitud del primer resultado **0.9650**.

La primera imagen recuperada corresponde a la misma identidad y a una cámara diferente. Es una recuperación correcta en Rank-1, Rank-5 y Rank-10; la presencia de otros candidatos incorrectos no invalida este acierto.

## Resolución original: cinco fallos

### Fallo 1 · consulta 0

![Ablación — Sin MIM: resolución original, fallo 1](../assets/results/no_mim/original_failure_01.png)

**Registro:** primera coincidencia correcta en el puesto **78**; similitud del primer resultado **0.9291**.

La consulta muestra una prenda clara sobre ropa verde; los candidatos incluyen combinaciones de colores parecidas, aunque pertenecen a otras identidades. El parecido de vestimenta no basta para resolver la consulta. La primera coincidencia ocupa el puesto 78, por lo que esta consulta falla también en Rank-5 y Rank-10. No se muestra una coincidencia correcta entre los cuatro candidatos del panel.

### Fallo 2 · consulta 1

![Ablación — Sin MIM: resolución original, fallo 2](../assets/results/no_mim/original_failure_02.png)

**Registro:** primera coincidencia correcta en el puesto **4**; similitud del primer resultado **0.9610**.

Los candidatos comparten prendas azules y pantalones oscuros; algunos también muestran una mochila. El modelo coloca identidades distintas por delante de la correcta pese al parecido visual. La identidad correcta aparece en el puesto 4: falla Rank-1, pero esta consulta sí es un acierto en Rank-5 y Rank-10.

### Fallo 3 · consulta 2

![Ablación — Sin MIM: resolución original, fallo 3](../assets/results/no_mim/original_failure_03.png)

**Registro:** primera coincidencia correcta en el puesto **212**; similitud del primer resultado **0.9785**.

Predominan camisetas amarillas, prendas oscuras y mochilas, con bicicletas en varias imágenes. La similitud de apariencia y entorno coincide con una confusión entre identidades; el panel no demuestra por sí solo qué señal causó el error. La primera coincidencia ocupa el puesto 212, por lo que esta consulta falla también en Rank-5 y Rank-10. No se muestra una coincidencia correcta entre los cuatro candidatos del panel.

### Fallo 4 · consulta 3

![Ablación — Sin MIM: resolución original, fallo 4](../assets/results/no_mim/original_failure_04.png)

**Registro:** primera coincidencia correcta en el puesto **1703**; similitud del primer resultado **0.9540**.

La consulta combina camiseta clara, pantalón corto y un bolso cruzado. Se recuperan otras personas con prendas claras y pantalón corto; las diferencias de identidad no quedan suficientemente separadas en la clasificación. La primera coincidencia ocupa el puesto 1703, por lo que esta consulta falla también en Rank-5 y Rank-10. No se muestra una coincidencia correcta entre los cuatro candidatos del panel.

### Fallo 5 · consulta 4

![Ablación — Sin MIM: resolución original, fallo 5](../assets/results/no_mim/original_failure_05.png)

**Registro:** primera coincidencia correcta en el puesto **98**; similitud del primer resultado **0.9606**.

La consulta contiene una mochila y ropa oscura, con postura lateral o de espaldas. Entre los candidatos aparecen mochilas y prendas de tonos próximos. El resultado evidencia confusión ante apariencias parecidas y cambios de punto de vista. La primera coincidencia ocupa el puesto 98, por lo que esta consulta falla también en Rank-5 y Rank-10. No se muestra una coincidencia correcta entre los cuatro candidatos del panel.

## Baja resolución: cinco aciertos

### Acierto 1 · consulta 7

![Ablación — Sin MIM: baja resolución, acierto 1](../assets/results/no_mim/low_resolution_success_01.png)

**Registro:** primera coincidencia correcta en el puesto **1**; similitud del primer resultado **0.9555**.

La primera imagen recuperada corresponde a la misma identidad y a una cámara diferente. Es una recuperación correcta en Rank-1, Rank-5 y Rank-10; la presencia de otros candidatos incorrectos no invalida este acierto.

### Acierto 2 · consulta 59

![Ablación — Sin MIM: baja resolución, acierto 2](../assets/results/no_mim/low_resolution_success_02.png)

**Registro:** primera coincidencia correcta en el puesto **1**; similitud del primer resultado **0.9627**.

La primera imagen recuperada corresponde a la misma identidad y a una cámara diferente. Es una recuperación correcta en Rank-1, Rank-5 y Rank-10; la presencia de otros candidatos incorrectos no invalida este acierto.

### Acierto 3 · consulta 65

![Ablación — Sin MIM: baja resolución, acierto 3](../assets/results/no_mim/low_resolution_success_03.png)

**Registro:** primera coincidencia correcta en el puesto **1**; similitud del primer resultado **0.9772**.

La primera imagen recuperada corresponde a la misma identidad y a una cámara diferente. Es una recuperación correcta en Rank-1, Rank-5 y Rank-10; la presencia de otros candidatos incorrectos no invalida este acierto.

### Acierto 4 · consulta 96

![Ablación — Sin MIM: baja resolución, acierto 4](../assets/results/no_mim/low_resolution_success_04.png)

**Registro:** primera coincidencia correcta en el puesto **1**; similitud del primer resultado **0.9762**.

La primera imagen recuperada corresponde a la misma identidad y a una cámara diferente. Es una recuperación correcta en Rank-1, Rank-5 y Rank-10; la presencia de otros candidatos incorrectos no invalida este acierto.

### Acierto 5 · consulta 103

![Ablación — Sin MIM: baja resolución, acierto 5](../assets/results/no_mim/low_resolution_success_05.png)

**Registro:** primera coincidencia correcta en el puesto **1**; similitud del primer resultado **0.9484**.

La primera imagen recuperada corresponde a la misma identidad y a una cámara diferente. Es una recuperación correcta en Rank-1, Rank-5 y Rank-10; la presencia de otros candidatos incorrectos no invalida este acierto.

## Baja resolución: cinco fallos

### Fallo 1 · consulta 0

![Ablación — Sin MIM: baja resolución, fallo 1](../assets/results/no_mim/low_resolution_failure_01.png)

**Registro:** primera coincidencia correcta en el puesto **95**; similitud del primer resultado **0.9188**.

La consulta muestra una prenda clara sobre ropa verde; los candidatos incluyen combinaciones de colores parecidas, aunque pertenecen a otras identidades. El parecido de vestimenta no basta para resolver la consulta. La consulta reducida pierde detalles finos de ropa y accesorios, aunque eso no prueba que la reducción sea la única causa de este fallo. La primera coincidencia ocupa el puesto 95, por lo que esta consulta falla también en Rank-5 y Rank-10. No se muestra una coincidencia correcta entre los cuatro candidatos del panel.

### Fallo 2 · consulta 1

![Ablación — Sin MIM: baja resolución, fallo 2](../assets/results/no_mim/low_resolution_failure_02.png)

**Registro:** primera coincidencia correcta en el puesto **4**; similitud del primer resultado **0.9543**.

Los candidatos comparten prendas azules y pantalones oscuros; algunos también muestran una mochila. El modelo coloca identidades distintas por delante de la correcta pese al parecido visual. La consulta reducida pierde detalles finos de ropa y accesorios, aunque eso no prueba que la reducción sea la única causa de este fallo. La identidad correcta aparece en el puesto 4: falla Rank-1, pero esta consulta sí es un acierto en Rank-5 y Rank-10.

### Fallo 3 · consulta 2

![Ablación — Sin MIM: baja resolución, fallo 3](../assets/results/no_mim/low_resolution_failure_03.png)

**Registro:** primera coincidencia correcta en el puesto **219**; similitud del primer resultado **0.9714**.

Predominan camisetas amarillas, prendas oscuras y mochilas, con bicicletas en varias imágenes. La similitud de apariencia y entorno coincide con una confusión entre identidades; el panel no demuestra por sí solo qué señal causó el error. La consulta reducida pierde detalles finos de ropa y accesorios, aunque eso no prueba que la reducción sea la única causa de este fallo. La primera coincidencia ocupa el puesto 219, por lo que esta consulta falla también en Rank-5 y Rank-10. No se muestra una coincidencia correcta entre los cuatro candidatos del panel.

### Fallo 4 · consulta 3

![Ablación — Sin MIM: baja resolución, fallo 4](../assets/results/no_mim/low_resolution_failure_04.png)

**Registro:** primera coincidencia correcta en el puesto **1803**; similitud del primer resultado **0.9481**.

La consulta combina camiseta clara, pantalón corto y un bolso cruzado. Se recuperan otras personas con prendas claras y pantalón corto; las diferencias de identidad no quedan suficientemente separadas en la clasificación. La consulta reducida pierde detalles finos de ropa y accesorios, aunque eso no prueba que la reducción sea la única causa de este fallo. La primera coincidencia ocupa el puesto 1803, por lo que esta consulta falla también en Rank-5 y Rank-10. No se muestra una coincidencia correcta entre los cuatro candidatos del panel.

### Fallo 5 · consulta 4

![Ablación — Sin MIM: baja resolución, fallo 5](../assets/results/no_mim/low_resolution_failure_05.png)

**Registro:** primera coincidencia correcta en el puesto **297**; similitud del primer resultado **0.9623**.

La consulta contiene una mochila y ropa oscura, con postura lateral o de espaldas. Entre los candidatos aparecen mochilas y prendas de tonos próximos. El resultado evidencia confusión ante apariencias parecidas y cambios de punto de vista. La consulta reducida pierde detalles finos de ropa y accesorios, aunque eso no prueba que la reducción sea la única causa de este fallo. La primera coincidencia ocupa el puesto 297, por lo que esta consulta falla también en Rank-5 y Rank-10. No se muestra una coincidencia correcta entre los cuatro candidatos del panel.

## Interpretación conjunta

Los aciertos confirman que el modelo resuelve algunas consultas entre cámaras. Los fallos muestran que una similitud alta puede asignarse a otra identidad, especialmente cuando coinciden la vestimenta y la apariencia general. Las pérdidas de entrenamiento descendieron, pero las representaciones no separan suficientemente a todas las personas.

Los ejemplos individuales no demuestran superioridad entre modelos. Esa comparación corresponde a las métricas del conjunto completo: en estas ejecuciones, el estudiante Sin MIM supera al estudiante Full en mAP y Rank-1/5/10, tanto con resolución original como reducida. La reducción de resolución disminuye las métricas agregadas de ambos.

Fuente: evaluación final `testing30_59085`; configuraciones y registros de la campaña `staged_20261029_052556`. No se han vuelto a entrenar modelos ni recalculado predicciones para preparar esta documentación.
