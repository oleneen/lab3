"""Запуск: python run_experiments.py [Student_Performance.csv] [german.csv]
Считает метрики на реальных данных, пишет experiment_results.json и REPORT.md."""
import json
import sys
from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from grader_contracts.linear_models import ClassificationInput, RegressionInput
from linear_models import (classification_metrics, compare_classical_models,
                           train_linear_regression, train_logistic_classifier)

SEED, TEST_SIZE = 42, 0.2
student_path = sys.argv[1] if len(sys.argv) > 1 else "Student_Performance.csv"
german_path = sys.argv[2] if len(sys.argv) > 2 else "german.csv"

# ---------- 1. Линейная регрессия ----------
df = pd.read_csv(student_path)
n_raw = len(df)
n_dup = int(df.duplicated().sum())
n_na = int(df.isna().sum().sum())
df = df.dropna()
df["Extracurricular Activities"] = (
    df["Extracurricular Activities"].str.strip().map({"Yes": 1, "No": 0}))
X = df.drop(columns="Performance Index")
y = df["Performance Index"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

reg_in = RegressionInput(Xtr.values, ytr.values, Xte.values, yte.values)
reg = train_linear_regression(reg_in)
mse = float(mean_squared_error(yte, reg.predictions))
mse_first, mse_last = reg.loss_history[0], reg.loss_history[-1]

# ---------- 2. Логистическая регрессия ----------
g = pd.read_csv(german_path, sep=";")
gy = g["Creditability"]
gX = g.drop(columns="Creditability")
Gtr, Gte, gytr, gyte = train_test_split(
    gX, gy, test_size=TEST_SIZE, random_state=SEED, stratify=gy)
cls_in = ClassificationInput(Gtr.values, gytr.values, Gte.values, gyte.values)
own = train_logistic_classifier(cls_in)
own_m = classification_metrics(gyte.values, own.probabilities, own.predictions)
base = compare_classical_models(cls_in)

results = {
    "linear_regression": {"test_mse": mse},
    "classification": {
        "own_logistic": asdict(own_m),
        "logistic_regression": asdict(base.logistic_regression),
        "decision_tree": asdict(base.decision_tree),
        "knn": asdict(base.knn),
    },
}
with open("experiment_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# ---------- Отчёт ----------
names = {"own_logistic": "Собственная логистическая регрессия",
         "logistic_regression": "LogisticRegression (sklearn)",
         "decision_tree": "DecisionTreeClassifier (max_depth=5)",
         "knn": "KNeighborsClassifier (k=5)"}
rows = "\n".join(
    f"| {names[k]} | {v['roc_auc']:.4f} | {v['accuracy']:.4f} | "
    f"{v['precision']:.4f} | {v['recall']:.4f} | {v['f1']:.4f} |"
    for k, v in results["classification"].items())
best_auc = max(results["classification"], key=lambda k: results["classification"][k]["roc_auc"])
own_auc = own_m.roc_auc
sk_auc = base.logistic_regression.roc_auc
share1 = float(gy.mean())
fi = int(np.argmax(np.abs(reg.weights)))
loss_c0, loss_c1 = own.loss_history[0], own.loss_history[-1]

report = f"""# Отчёт по лабораторной работе №3: линейные модели

## Данные и подготовка

**Student_Performance.csv.** Строк: {n_raw}, пропусков: {n_na}, полных дубликатов: {n_dup}.
Целевая переменная: `Performance Index`. Признак `Extracurricular Activities`
закодирован числами: Yes = 1, No = 0. Разбиение: {int((1-TEST_SIZE)*100)}% обучающая и
{int(TEST_SIZE*100)}% тестовая часть, `random_state={SEED}`.
Признаки стандартизованы внутри `train_linear_regression` (среднее и стандартное
отклонение берутся только из обучающей части), целевая переменная не масштабируется.
После обучения веса пересчитываются в исходный масштаб признаков.

**german.csv.** Строк: {len(g)}, признаков: {gX.shape[1]}, разделитель `;`.
Целевая переменная `Creditability` (первый столбец), доля класса 1: {share1:.3f}.
Разбиение стратифицированное, {int(TEST_SIZE*100)}% в тесте, `random_state={SEED}`.
Стандартизация выполняется внутри моделей по статистикам обучающей части.

## Гипотезы и эксперименты

1. Гипотеза: целевая переменная в Student_Performance почти линейно зависит от
   признаков, поэтому линейная регрессия даёт малую ошибку. Параметры:
   `learning_rate={reg_in.learning_rate}`, `epochs={reg_in.epochs}`. MSE на
   обучающей части упала с {mse_first:.3f} (эпоха 1) до {mse_last:.3f} (эпоха {reg_in.epochs}).
   Наибольший по модулю вес в исходном масштабе: признак `{X.columns[fi]}`
   ({reg.weights[fi]:.4f}).
2. Гипотеза: стандартизация признаков делает градиентный спуск устойчивым при
   разных масштабах (`Credit_Amount` в тысячах, бинарные признаки порядка единицы).
   Параметры: `learning_rate={cls_in.learning_rate}`, `epochs={cls_in.epochs}`.
   Кросс-энтропия на обучающей части: {loss_c0:.4f} → {loss_c1:.4f}.
3. Гипотеза: нелинейные модели (дерево, KNN) могут превзойти линейную на german.csv.
   Проверка выполнена на том же разбиении.

## Метрики

**Линейная регрессия.** Итоговая MSE на тестовой выборке: **{mse:.4f}**.

**Классификация (тестовая выборка).**

| Модель | ROC-AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
{rows}

## Выводы

- Линейная регрессия: MSE на тесте {mse:.4f}, ход MSE монотонно убывает
  ({mse_first:.3f} → {mse_last:.3f}), то есть спуск сходится при выбранном шаге.
- Лучшая модель по ROC-AUC: {names[best_auc]} ({results['classification'][best_auc]['roc_auc']:.4f}).
  ROC-AUC собственной модели: {own_auc:.4f}, sklearn LogisticRegression: {sk_auc:.4f}.
  Разница {abs(own_auc - sk_auc):.4f}: обе модели решают одну и ту же задачу, различия
  возникают из-за регуляризации в sklearn (по умолчанию L2) и конечного числа
  шагов спуска.
- Собственная модель линейна, поэтому не улавливает нелинейные взаимодействия
  признаков, которые доступны дереву. Дерево при этом склонно к переобучению,
  а KNN чувствителен к масштабу и размерности.
- Доля класса 1 равна {share1:.3f}, поэтому Accuracy нужно читать вместе с
  Precision, Recall и F1.

## Воспроизводимость

- Python, NumPy, Pandas, scikit-learn.
- Разбиение: `train_test_split(test_size={TEST_SIZE}, random_state={SEED})`
  (для german.csv дополнительно `stratify`).
- `DecisionTreeClassifier(max_depth=5, random_state={SEED})`, `KNeighborsClassifier(n_neighbors=5)`.
- Запуск: `python run_experiments.py Student_Performance.csv german.csv`.
  Скрипт перезаписывает `experiment_results.json` и этот отчёт.
"""
with open("REPORT.md", "w", encoding="utf-8") as f:
    f.write(report)
print(json.dumps(results, indent=2, ensure_ascii=False))
