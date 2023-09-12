# importar librerias utilizadas
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree # Importamos el modelo de arboles de decision
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score, classification_report # metricas
from sklearn.preprocessing import LabelEncoder # Para relizar encoding de la variable objetivo
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt

""" Definicion de funciones """

# Funcion propia para obtener un reporte de clasificacion en conjunto
def evaluate_model(y_true, y_pred):
    # se utiliza el parametro weighted para que se calcule la media ponderada por la naturaleza de clasificacion multiclase
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"Exactitud (Accuracy): {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Score F1: {f1}")
    print("Matriz de confusion:\n")
    print(cm)

# Funcion propia para obtener un reporte de clasificacion en conjunto para realizar analisis de bias,varianza,fitting
def diagnose_model_performance(y_train_true, y_train_pred, y_val_true, y_val_pred):
    # Calcular las métricas de rendimiento de los conjuntos de entrenamiento y validación
    train_accuracy = accuracy_score(y_train_true, y_train_pred)
    val_accuracy = accuracy_score(y_val_true, y_val_pred)
    
    train_f1 = f1_score(y_train_true, y_train_pred, average='weighted')
    val_f1 = f1_score(y_val_true, y_val_pred, average='weighted')
    
    # Las siguientes condiciones para determinar el rendimiento del modelo estan basadas en las definiciones de los terminos vistos en clase 
    # Diagnostico de Bias (sesgo) (underfitting vs. fitting vs. overfitting)
    if train_accuracy >= val_accuracy:
        bias_diagnosis = "Sesgo bajo (Fitting)"
    else:
        bias_diagnosis = "Sesgo elevado (Underfitting)"
    
    # Diagnostico de Varianza (overfitting)
    if train_f1 >= val_f1:
        variance_diagnosis = "Varianza baja"
    else:
        variance_diagnosis = "Varianza media (Overfitting)"
    
    # Diagnóstico del nivel de ajuste
    if bias_diagnosis == "Sesgo bajo (Fitting)" and variance_diagnosis == "Varianza baja":
        model_fit = "Buen ajuste (Reasonably Fitting)"
    else:
        model_fit = "Overfitting"  # Si no cumple los criterios de un buen ajuste
    
    return bias_diagnosis, variance_diagnosis, model_fit

""" Construccion de los modelos """

print('Inicializando...\n')
print('Cargando Datos...\n')

# Se utiliza la libreria load_iris de sklearn para obtener los datos de la muestra iris por motivos practicos
data = load_iris()

print('Separacion de datos...\n\n')
# Separacion de los datos en entrenamiento y prueba utilizando train_test_split de sklearn
X = data.data
Y = data.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=0)

""" Primer modelo GINI """
print('Entrenamiento de primer modelo:')
print('Utilizando Gini como criterio para la ganancia de informacion:\n')
classifier = DecisionTreeClassifier(criterion='gini', random_state=0, splitter = 'best', max_depth = 3)
classifier.fit(X_train,Y_train)

for feature_idx, importance in enumerate(classifier.feature_importances_):
    print(f"Variable {feature_idx}: Importancia: {importance:.4f}")

y_pred = classifier.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(Y_test, y_pred))
print(confusion_matrix(Y_test, y_pred))
# Accuracy score
print('Excatitud del modelo:\n', accuracy_score(y_pred,Y_test))

""" Segundo modelo Entropía """
print('Entrenamiento del segundo modelo:')
print('Utilizando Entropía como criterio para la ganancia de informacion:\n')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=0)
classifier_ent = DecisionTreeClassifier(criterion='entropy', random_state=0, splitter = 'best', max_depth = 3)
classifier_ent.fit(X_train,Y_train)

for feature_idx, importance in enumerate(classifier_ent.feature_importances_):
    print(f"Variable {feature_idx}: Importancia: {importance:.4f}")

y_pred_ent = classifier_ent.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(Y_test, y_pred_ent))
print(confusion_matrix(Y_test, y_pred_ent))
# Accuracy score
print('Excatitud del modelo:\n',accuracy_score(y_pred_ent,Y_test))

print('Se observa que al cambiar el criterio de ganancia de informacion\nno se ve afectado el desempeño del modelo')
print('NOTA: A partir de este momento utilizaremos el modelo a continuación (mejores parámetros)')



""" Tercer modelo (Best Params --Default Params--) """
print('\nEntrenamiento de modelo con los mejores parámetros:\n')
# Utilizando los mejores parámetros utilizados para realizar predicciones y observar la generalización del modelo
X = data.data
y = data.target

# Dividir el conjunto de datos en conjuntos de entrenamiento, validación y prueba 60-20-20
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Modelo de clasificación con parametros default (best params)
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

for feature_idx, importance in enumerate(clf.feature_importances_):
    print(f"Variable {feature_idx}: Importancia: {importance:.4f}")

# Realizar predicciones sobre el conjunto de validación
y_pred = clf.predict(X_validation)

# Visualizar arbol de decision
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=data.feature_names, class_names=data.target_names)
plt.show()
plt.savefig("arbol_decision_best_params.png")

# Evaluar el modelo
print("Validation Set Metrics:")
evaluate_model(y_validation, y_pred)

# Predicciones para el conjunto de prueba
y_test_pred = clf.predict(X_test)

# Evaluar el modelo con los datos de prueba
print("\nTest Set Metrics:")
evaluate_model(y_test, y_test_pred)

print('Realizar predicciones sobre nuevos datos:')
# Define the new data as a numpy array
new_data = np.array([[4.5, 3.2, 1.3, 0.2],
                     [6.7, 3.1, 4.4, 1.4],
                     [5.1, 2.9, 3.3, 1.0],
                     [5.0, 2.0, 3.5, 1.0],
                     [5.9, 3.0, 5.1, 1.8]])

# Datos a usar
print(f'Nuevos datos a usar:\n{new_data}')

# Use the trained model to make predictions on the new data
new_data_predictions = clf.predict(new_data)

# Display the predictions
print('Predicciones para nuevos datos:')
for i, prediction in enumerate(new_data_predictions):
    print(f"Dato {i+1}: Prediccion clase - {prediction}")


""" Ejecucion de Diagnostico de Bias, Varianza y ajuste """
print('\nEjecucion de Diagnostico de Bias, Varianza y ajuste\n')
# Obtener reporte de clasificacion en conjunto
bias_diagnosis, variance_diagnosis, model_fit = diagnose_model_performance(y_train, clf.predict(X_train), y_validation, y_pred)
print("Diagnostico de Bias:", bias_diagnosis)
print("Diagnostico de Varianza:", variance_diagnosis)
print("Diagnostico de Ajuste de Modelo (fit):", model_fit)


print('\nNOTA: Esta ejecucion de Diagnostico de Bias, Varianza y ajuste fue realizada para el modelo con los mejores parámetros.\nPor favor referirse a la documentación de las funciones para obtener un mejor entendimiento de las funciones utilizadas en este programa')
