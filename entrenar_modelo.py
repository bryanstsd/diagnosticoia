import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os


def crear_directorios():
    """Crea los directorios necesarios si no existen"""
    directorios = ['models', 'metrics', 'plots']
    for dir in directorios:
        os.makedirs(dir, exist_ok=True)


def generar_graficas_evaluacion(model, X_test, y_test, feature_names, label_encoders, y_pred, y_pred_proba):
    """Genera y guarda las gráficas de evaluación del modelo"""

    # Obtener nombres de las clases
    class_names = label_encoders['Disease'].classes_.tolist()
    n_classes = len(class_names)

    # 1. Matriz de Confusión
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Matriz de Confusión')
    plt.ylabel('Real')
    plt.xlabel('Predicción')
    plt.tight_layout()
    plt.savefig('plots/confusion_matrix.png')
    plt.close()

    # 2. Importancia de características
    plt.figure(figsize=(10, 6))
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    sns.barplot(data=importances, x='importance', y='feature')
    plt.title('Importancia de Características')
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png')
    plt.close()

    # 3. Curvas ROC
    plt.figure(figsize=(10, 8))

    # Calcular curva ROC para cada clase
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Convertir y_test a formato one-hot
    y_test_bin = pd.get_dummies(y_test).values

    # Asegurarse de que y_pred_proba tenga el mismo número de columnas que clases
    if y_pred_proba.shape[1] != n_classes:
        print(
            f"Advertencia: El número de columnas en y_pred_proba ({y_pred_proba.shape[1]}) no coincide con el número de clases ({n_classes})")
        return None

    try:
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

            plt.plot(
                fpr[i],
                tpr[i],
                label=f'ROC {class_names[i]} (AUC = {roc_auc[i]:.2f})'
            )
    except Exception as e:
        print(f"Error al generar curvas ROC: {str(e)}")
        print(f"Forma de y_test_bin: {y_test_bin.shape}")
        print(f"Forma de y_pred_proba: {y_pred_proba.shape}")
        print(f"Número de clases: {n_classes}")
        return None

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curvas ROC')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('plots/roc_curves.png')
    plt.close()

    # Preparar datos para el retorno
    metrics_dict = {
        'confusion_matrix': cm.tolist(),
        'class_names': class_names,
        'feature_importance': importances.to_dict('records'),
        'roc_curves': {}
    }

    # Agregar datos de curvas ROC
    for i in range(n_classes):
        metrics_dict['roc_curves'][class_names[i]] = {
            'fpr': fpr[i].tolist(),
            'tpr': tpr[i].tolist(),
            'auc': float(roc_auc[i])
        }

    return metrics_dict


def guardar_metricas(metrics, filename='metrics/model_metrics.json'):
    """Guarda las métricas en un archivo JSON"""
    # Asegurar que el directorio existe
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Convertir arrays numpy a listas para JSON
    metrics_json = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            metrics_json[key] = value.tolist()
        elif isinstance(value, np.float32):
            metrics_json[key] = float(value)
        else:
            metrics_json[key] = value

    with open(filename, 'w') as f:
        json.dump(metrics_json, f, indent=4)


def entrenar_modelo():
    """Entrena el modelo y genera todas las métricas y visualizaciones"""
    try:
        # Crear directorios necesarios
        crear_directorios()

        # Cargar datos
        print("Cargando datos...")
        data = pd.read_csv('data/dataset.csv')

        # Verificar datos
        print("\nInformación del dataset:")
        print(data.info())
        print("\nPrimeras filas:")
        print(data.head())
        print("\nValores únicos en Disease:")
        print(data['Disease'].unique())

        # Definir columnas
        feature_columns = [
            'Age',
            'Gender',
            'Blood Pressure',
            'Cholesterol Level',
            'Fever',
            'Cough',
            'Fatigue',
            'Difficulty Breathing'
        ]
        target_column = 'Disease'

        # Verificar columnas
        missing_columns = [col for col in feature_columns + [target_column] if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Columnas faltantes en el dataset: {missing_columns}")

        # Preparar datos
        print("Preparando datos...")
        X = data[feature_columns].copy()
        y = data[target_column].copy()

        # Asegurar que Age sea numérica
        X['Age'] = pd.to_numeric(X['Age'], errors='coerce')

        # Label encoders
        label_encoders = {}
        categorical_columns = [col for col in feature_columns if col != 'Age']

        for column in categorical_columns:
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column].astype(str))
            label_encoders[column] = le

        # Label encoder para Disease
        le_disease = LabelEncoder()
        y = le_disease.fit_transform(y)
        label_encoders['Disease'] = le_disease

        # División de datos
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Entrenamiento
        print("Entrenando modelo...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Predicciones
        print("Generando predicciones y métricas...")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        print(f"\nForma de y_pred_proba: {y_pred_proba.shape}")
        print(f"Número de clases únicas: {len(np.unique(y))}")

        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5)

        # Calcular métricas
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision_weighted': float(precision_score(y_test, y_pred, average='weighted')),
            'recall_weighted': float(recall_score(y_test, y_pred, average='weighted')),
            'f1_weighted': float(f1_score(y_test, y_pred, average='weighted')),
            'cross_val_scores': {
                'mean': float(cv_scores.mean()),
                'std': float(cv_scores.std()),
                'scores': cv_scores.tolist()
            },
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Generar gráficas y obtener métricas visuales
        print("Generando visualizaciones...")
        visual_metrics = generar_graficas_evaluacion(
            model,
            X_test,
            y_test,
            feature_names=feature_columns,
            label_encoders=label_encoders,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba
        )

        if visual_metrics is not None:
            metrics.update(visual_metrics)

        # Guardar modelo, encoders y métricas
        print("Guardando archivos...")
        joblib.dump(model, 'models/modelo_random_forest.pkl')
        joblib.dump(label_encoders, 'models/label_encoders.pkl')
        guardar_metricas(metrics)

        print("\nEntrenamiento completado exitosamente!")

    except Exception as e:
        print(f"\nError durante el entrenamiento: {str(e)}")
        import traceback
        print("\nTraceback completo:")
        print(traceback.format_exc())
        return


if __name__ == "__main__":
    entrenar_modelo()