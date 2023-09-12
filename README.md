# Uso de framework o biblioteca de aprendizaje máquina para la implementación de una solución.

En este trabajo seguiremos utilizando un modelo de aprendizaje máquina de un árbol de decisión. 
Como se menciona en el titulo, este modelo se implementa con la ayuda de la librería (framework) Sci-Kit Learn de Python.

Se ha utilizado el conjuto de datos Iris por su versatilidad y otras razones prácticas como:

- Tamaño Adecuado
- Balance entre clases
- Reduccion de Overfitting

## Contenido

- [Requisitos](#requisitos)
- [Instrucciones de Uso](#instrucciones-de-uso)
- [Funcionamiento](#funcionamiento)

## Requisitos

- Python 3.x
- Bibliotecas: numpy, pandas, scikit-learn, matplotlib (para generar datos y visualización de resultados)

## Instrucciones de Uso

1. Clona este repositorio en tu máquina local:

    https://github.com/juancarloscorona14/mod2-frame.git

2. Navega al directorio del repositorio (En este caso no es necesario modificar la ruta del dataset ya que este está incluido dentro de las librerías)

3. Ejecuta

4. Si deseas cambiar los datos de prueba, por favor modifica el .py

## Funcionamiento
Esta seccion contiene una descripción general de la estructura del programa.

Se definen dos funciones personalizadas:

- **evaluate_model(y_true, y_pred)**: 

    Esta función calcula y muestra métricas de evaluación del modelo, como exactitud, precisión, recall, puntuación F1 y la matriz de confusión.

- **diagnose_model_performance(y_train_true, y_train_pred, y_val_true, y_val_pred)**: 

    Esta función calcula y muestra diagnósticos de bias, varianza y ajuste del modelo en conjuntos de entrenamiento y validación.

### Construcción de Modelos

Con el fin de visualizar claramente la generalización del modelo de un arbol de decision, se han realizado 3 modelos de árbol de decision con diferentes parámetros:

1. Primer Modelo GINI

    Se entrena un modelo de árbol de decisión utilizando el criterio GINI para la ganancia de información. Se imprime la importancia de las variables y se muestra un resumen de las predicciones realizadas por el modelo.

2. Segundo Modelo Entropía

    Se entrena un segundo modelo de árbol de decisión utilizando el criterio de entropía para la ganancia de información. Al igual que en el primer modelo, se imprime la importancia de las variables y se muestra un resumen de las predicciones

3. Tercer Modelo (Mejores Parámetros - Default Parámetros)

    En esta sección, se entrena un modelo de árbol de decisión con los mejores parámetros y parámetros predeterminados. Se divide el conjunto de datos en conjuntos de entrenamiento, validación y prueba. 
    
    Se muestra el árbol de decisión visualmente y se evalúa el rendimiento del modelo en los conjuntos de validación y prueba. También se realizan predicciones en nuevos datos.