# Análisis y Reporte sobre el desempeño del modelo
_Juan Carlos Corona Vega A01660135_

_Septiembre 2023_

## Introducción
En este informe, se presenta un análisis detallado del modelo de Árbol de Decisión aplicado a un conjunto de datos utilizando Python. El objetivo es demostrar cómo se ha implementado y evaluado el modelo, así como explorar conceptos clave como el sesgo, la varianza y el ajuste del modelo. A lo largo del informe, se utilizarán gráficos y métricas para respaldar el análisis.

### Acerca de la elección del conjunto de datos
El conjunto de datos utilizado en este análisis es el conjunto de datos "Iris", disponible en la librería scikit-learn. La elección de este conjunto de datos se justifica por las siguientes razones:

- **Relevancia:** 
    El conjunto de datos "Iris" es ampliamente utilizado en la comunidad de aprendizaje automático como un conjunto de datos de referencia para la clasificación. Contiene información sobre tres especies de flores iris, lo que lo hace adecuado para un problema de clasificación multiclase.

- **Facilidad de Entendimiento:** 
    El conjunto de datos "Iris" es simple y fácil de entender, lo que lo convierte en un excelente punto de partida para la implementación y evaluación de modelos.

- **Generalización:** 
    Dado que es un conjunto de datos bien conocido, permite evaluar la capacidad de un modelo de Árbol de Decisión para generalizar y clasificar nuevas muestras de flores iris.

## Separación de datos

El conjunto de datos se divide en tres conjuntos: entrenamiento, validación y prueba. La división es de 60% para entrenamiento, 20% para validación y 20% para prueba.

[insertar grafico de pastel]

## Modelos de Árbol de Decisión
### Primer Modelo (Criterio GINI)
El primer modelo se entrena utilizando el criterio GINI para la ganancia de información. A continuación, se muestran los resultados de este modelo:

- Exactitud (Accuracy): [Valor]
- Precisión: [Valor]
- Recall: [Valor]
- Score F1: [Valor]
- Matriz de Confusión: [Matriz]

### Segundo Modelo (Criterio Entropía)
El segundo modelo se entrena utilizando el criterio de entropía para la ganancia de información. A continuación, se muestran los resultados de este modelo:

- Exactitud (Accuracy): [Valor]
- Precisión: [Valor]
- Recall: [Valor]
- Score F1: [Valor]
- Matriz de Confusión: [Matriz]

Se observa que al cambiar el criterio de ganancia de información, no se ve afectado el desempeño del modelo.

### Tercer Modelo (Mejores Parámetros)
El tercer modelo se entrena con los mejores parámetros seleccionados. A continuación, se presentan los resultados y se visualiza el árbol de decisión:

- Exactitud (Accuracy): [Valor]
- Precisión: [Valor]
- Recall: [Valor]
- Score F1: [Valor]

#### Análisis de Sesgo, Varianza y Ajuste del Modelo
El diagnóstico de sesgo, varianza y ajuste del modelo se realiza en el tercer modelo con los mejores parámetros. Los resultados son los siguientes:

- Diagnóstico de Bias: [Resultado]
- Diagnóstico de Varianza: [Resultado]
- Diagnóstico de Ajuste de Modelo: [Resultado]
- Gráficos y Visualizaciones

A continuación, se presentan gráficos y visualizaciones que respaldan el análisis:

Separación de Datos

[Gráfico de separación de datos]

Aprendizaje de Conjuntos de Datos (Entrenamiento, Validación, Prueba)

[Gráfico comparativo del aprendizaje de conjuntos de datos]

Diagnóstico de Sesgo

[Gráfico comparativo del diagnóstico de sesgo]

Diagnóstico de Varianza

[Gráfico comparativo del diagnóstico de varianza]

Ajuste del Modelo

[Gráfico comparativo del ajuste del modelo]

## Mejoras en el Desempeño
Se aplicaron tres técnicas para mejorar el desempeño del modelo:

[Explicación de la Técnica 1 y comparativo antes y después]

[Explicación de la Técnica 2 y comparativo antes y después]

[Explicación de la Técnica 3 y comparativo antes y después]

## Conclusiones
En este informe, se ha analizado en detalle la implementación y evaluación de un modelo de Árbol de Decisión en el conjunto de datos "Iris". Se han explorado conceptos clave como sesgo, varianza y ajuste del modelo. Además, se han aplicado técnicas de mejora de desempeño para optimizar el modelo. El conjunto de datos "Iris" resultó ser apropiado para el algoritmo de Árbol de Decisión, demostrando su capacidad de generalización. El diagnóstico de sesgo y varianza reveló que el modelo tiene un buen ajuste y una varianza baja. Las técnicas de mejora proporcionaron resultados positivos en el desempeño del modelo.

