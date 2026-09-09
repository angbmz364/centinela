"""
utils.py

Funciones auxiliares para el cálculo, visualización y guardado de
métricas detalladas de validación y prueba del modelo.

El informe se imprime en consola (en español) y se guarda en
logs/latest.txt. Si ya existe un logs/latest.txt, se renombra
automáticamente con su fecha de creación (latest_YYYYMMDD_HHMMSS.txt).
"""

import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    cohen_kappa_score,
)

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


class Tee:
    """Envía la salida hacia varios destinos a la vez (consola y archivo)."""

    def __init__(self, *files):
        self.files = files

    def write(self, text):
        for f in self.files:
            f.write(text)

    def flush(self):
        for f in self.files:
            f.flush()


def setup_log_file(filename="latest.txt"):
    """
    Prepara logs/<filename> para escritura.

    Si ya existe, se renombra a logs/latest_<YYYYMMDD_HHMMSS>.txt
    utilizando la fecha de creación del archivo anterior.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    latest_path = LOG_DIR / filename

    if latest_path.exists():
        timestamp = datetime.fromtimestamp(
            latest_path.stat().st_mtime
        ).strftime("%Y%m%d_%H%M%S")
        archived = LOG_DIR / f"latest_{timestamp}.txt"
        latest_path.rename(archived)

    return open(latest_path, "w", encoding="utf-8")


@contextmanager
def capture_output_to_log(filename="latest.txt"):
    """
    Contexto que redirige la salida estándar hacia logs/<filename>
    manteniendo también la impresión en consola.
    """
    log_file = setup_log_file(filename)
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_file)
    try:
        yield
    finally:
        sys.stdout.flush()
        sys.stdout = original_stdout
        log_file.flush()
        log_file.close()


def log_detailed_metrics(
    labels,
    predictions,
    probabilities,
    confidences,
    class_names,
    dataset_name="validación",
):
    """
    Calcula e imprime (en español) un informe completo de métricas.

    Parámetros
    ----------
    labels : list/array
        Etiquetas reales.
    predictions : list/array
        Clases predichas.
    probabilities : list/array
        Probabilidades Softmax de cada imagen (forma: n x num_clases).
    confidences : list/array
        Confianza (probabilidad máxima) de cada predicción.
    class_names : list[str]
        Nombres de las clases (en el orden de ImageFolder).
    dataset_name : str
        Nombre del conjunto evaluado (validación / prueba).
    """

    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    probabilities = np.asarray(probabilities, dtype=float)
    confidences = np.asarray(confidences, dtype=float)

    n_classes = len(class_names)

    # ----------------------------------------------------------------
    # Cálculo de métricas base
    # ----------------------------------------------------------------
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )

    accuracy = accuracy_score(labels, predictions) * 100
    balanced_acc = balanced_accuracy_score(labels, predictions) * 100
    mcc = matthews_corrcoef(labels, predictions)
    kappa = cohen_kappa_score(labels, predictions)
    cm = confusion_matrix(labels, predictions)

    # ----------------------------------------------------------------
    # Cabecera del informe
    # ----------------------------------------------------------------
    print("\n" + "=" * 74)
    print(f"   INFORME DETALLADO DE MÉTRICAS — CONJUNTO DE {dataset_name.upper()}")
    print("=" * 74)

    # ----------------------------------------------------------------
    # 1) Exactitud
    # ----------------------------------------------------------------
    print("\n[1] EXACTITUD GLOBAL")
    print(f"  - Exactitud (Accuracy):         {accuracy:.2f}%")
    print(f"  - Exactitud balanceada:         {balanced_acc:.2f}%")
    print("      (Promedio de la sensibilidad de ambas clases;")
    print("       robusta ante desbalance de clases).")

    # ----------------------------------------------------------------
    # 2) Precisión, Recall, F1 por clase
    # ----------------------------------------------------------------
    print("\n[2] MÉTRICAS POR CLASE")
    header = f"  {'Clase':<20}{'Precisión':>12}{'Recall':>12}{'F1-Score':>12}{'Soporte':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, name in enumerate(class_names):
        print(
            f"  {name:<20}"
            f"{precision[i] * 100:>10.2f}%"
            f"{recall[i] * 100:>11.2f}%"
            f"{f1[i] * 100:>11.2f}%"
            f"{support[i]:>10}"
        )

    print("\n  Significado para el proyecto:")
    for i, name in enumerate(class_names):
        print(
            f"    - Precisión '{name}': {precision[i] * 100:.2f}% "
            f"(de las predicciones '{name}', esta proporción fue correcta)."
        )
        print(
            f"    - Recall/Sensibilidad '{name}': {recall[i] * 100:.2f}% "
            f"(de los '{name}' reales, esta proporción fue detectada)."
        )
        print(
            f"    - F1-Score '{name}': {f1[i] * 100:.2f}% "
            f"(equilibrio armónico entre Precisión y Recall)."
        )

    # ----------------------------------------------------------------
    # 3) Promedios y medidas globales robustas
    # ----------------------------------------------------------------
    print("\n[3] PROMEDIOS Y MEDIDAS GLOBALES")
    print(
        "  - Macro (promedio simple por clase):\n"
        f"      Precisión {macro_p * 100:.2f}% | "
        f"Recall {macro_r * 100:.2f}% | "
        f"F1-Score {macro_f1 * 100:.2f}%"
    )
    print(
        "  - Ponderado (según el soporte de cada clase):\n"
        f"      Precisión {weighted_p * 100:.2f}% | "
        f"Recall {weighted_r * 100:.2f}% | "
        f"F1-Score {weighted_f1 * 100:.2f}%"
    )
    print(f"  - Correlación de Matthews (MCC): {mcc:.4f}")
    print("      (Calidad global de la clasificación; 1 = perfecta, 0 = aleatoria).")
    print(f"  - Kappa de Cohen:                {kappa:.4f}")
    print("      (Acuerdo entre predicción y realidad, corregido por azar).")

    # ----------------------------------------------------------------
    # 4) Análisis de confianza
    # ----------------------------------------------------------------
    print("\n[4] ANÁLISIS DE CONFIANZA (probabilidad Softmax)")

    avg_conf = float(np.mean(confidences) * 100)
    print(f"  - Confianza promedio general:      {avg_conf:.2f}%")

    correct = predictions == labels
    if correct.any():
        print(f"  - Confianza promedio en ACIERTOS:  {np.mean(confidences[correct]) * 100:.2f}%")
    if (~correct).any():
        print(f"  - Confianza promedio en ERRORES:   {np.mean(confidences[~correct]) * 100:.2f}%")
        print("      (En un modelo robusto, los errores deben tener baja confianza).")

    print("  - Confianza promedio por clase predicha:")
    for i, name in enumerate(class_names):
        mask = predictions == i
        if mask.any():
            avg = np.mean(confidences[mask]) * 100
            print(f"      '{name}': {avg:.2f}%")
        else:
            print(f"      '{name}': sin predicciones")

    # ----------------------------------------------------------------
    # 5) Métricas de discriminación (curvas ROC / PR)
    # ----------------------------------------------------------------
    print("\n[5] MÉTRICAS AVANZADAS DE DISCRIMINACIÓN")

    if n_classes == 2:
        try:
            roc_auc = roc_auc_score(labels, probabilities[:, 1])
            ap_neg = average_precision_score(1 - labels, probabilities[:, 0])
            ap_pos = average_precision_score(labels, probabilities[:, 1])
            print(f"  - ROC AUC: {roc_auc:.4f}")
            print("      (Capacidad de distinguir las clases sin depender del umbral;")
            print("       1.0 = separación perfecta).")
            print(f"  - PR AUC / Average Precision '{class_names[0]}': {ap_neg:.4f}")
            print(f"  - PR AUC / Average Precision '{class_names[1]}': {ap_pos:.4f}")
        except Exception as exc:
            print(f"  - No se pudieron calcular las curvas: {exc}")
    else:
        print("  - Disponible únicamente para clasificación binaria.")

    # ----------------------------------------------------------------
    # 6) Matriz de confusión
    # ----------------------------------------------------------------
    print("\n[6] MATRIZ DE CONFUSIÓN")
    print("  (Filas = etiqueta real, Columnas = predicción)")

    col_width = max(len(name) for name in class_names)
    cell = max(col_width + 6, 10)

    print(f"  {'':>{col_width}}", end="")
    for name in class_names:
        print(f"{name:>{cell}}", end="")
    print()

    for i, name in enumerate(class_names):
        print(f"  {'Real ' + name:>{col_width + 5}}", end="")
        for j in range(n_classes):
            print(f"{cm[i, j]:>{cell}}", end="")
        print()

    # ----------------------------------------------------------------
    # 7) Interpretación para sustento del proyecto
    # ----------------------------------------------------------------
    print("\n[7] INTERPRETACIÓN PARA SUSTENTO DEL PROYECTO")
    if n_classes == 2:
        target, other = class_names[0], class_names[1]
        tp = int(cm[0, 0])
        fn_pos = int(cm[0, 1])
        fp_pos = int(cm[1, 0])
        tn = int(cm[1, 1])
        print(f"  Tomando '{target}' como la clase objetivo:")
        print(f"    - Verdaderos Positivos (VP): {tp}  — '{target}' detectado correctamente.")
        print(f"    - Verdaderos Negativos (VN): {tn}  — '{other}' descartado correctamente.")
        print(f"    - Falsos Positivos (FP):     {fp_pos}  — se predijo '{target}' siendo '{other}'.")
        print(f"    - Falsos Negativos (FN):     {fn_pos}  — se predijo '{other}' siendo '{target}'.")
        sens = tp / (tp + fn_pos) if (tp + fn_pos) else 0.0
        spec = tn / (tn + fp_pos) if (tn + fp_pos) else 0.0
        print(f"\n  Sensibilidad del '{target}' ({sens * 100:.2f}%):")
        print("      De todos los casos reales de la plaga, cuántos detectó el modelo.")
        print("      Métrica más crítica: un valor alto evita que la plaga pase desapercibida.")
        print(f"  Especificidad del '{target}' ({spec * 100:.2f}%):")
        print("      De todos los casos que NO son la plaga, cuántos descartó correctamente.")
        print("      Evita falsas alarmas y la toma innecesaria de medidas de control.")
    else:
        for i, name in enumerate(class_names):
            print(f"    - Aciertos de '{name}': {cm[i, i]}/{int(support[i])}")

    print("\n" + "=" * 74 + "\n")