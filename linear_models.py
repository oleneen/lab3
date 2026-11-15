import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)

RANDOM_STATE = 42


def _as_2d(x) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(len(x), -1)


def _standardize(train: np.ndarray, test: np.ndarray):
    """Масштабирование по статистикам обучающей выборки (без утечки из test)."""
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0] = 1.0  # константный признак не делится на ноль
    return (train - mean) / std, (test - mean) / std, mean, std


def _to_original_scale(w_std, b_std, mean, std):
    """Пересчёт весов из стандартизованного пространства в исходное."""
    w = w_std / std
    b = b_std - float(np.sum(w_std * mean / std))
    return w, b


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Линейная регрессия: пакетный градиентный спуск по MSE."""
    x_train = _as_2d(data.train_features)
    x_test = _as_2d(data.test_features)
    y = np.asarray(data.train_target, dtype=float).ravel()

    xs_train, xs_test, mean, std = _standardize(x_train, x_test)
    n, k = xs_train.shape
    w = np.zeros(k)
    b = 0.0
    history: list[float] = []

    for _ in range(data.epochs):
        error = xs_train @ w + b - y
        history.append(float(np.mean(error ** 2)))
        w -= data.learning_rate * (2.0 / n) * (xs_train.T @ error)
        b -= data.learning_rate * (2.0 / n) * float(np.sum(error))

    w_orig, b_orig = _to_original_scale(w, b, mean, std)
    predictions = x_test @ w_orig + b_orig
    return RegressionResult(weights=w_orig, bias=float(b_orig),
                            loss_history=history, predictions=predictions)


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Бинарный логистический классификатор: пакетный GD по кросс-энтропии."""
    x_train = _as_2d(data.train_features)
    x_test = _as_2d(data.test_features)
    y = np.asarray(data.train_target, dtype=float).ravel()

    xs_train, xs_test, mean, std = _standardize(x_train, x_test)
    n, k = xs_train.shape
    w = np.zeros(k)
    b = 0.0
    eps = 1e-12
    history: list[float] = []

    for _ in range(data.epochs):
        p = _sigmoid(xs_train @ w + b)
        loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        history.append(float(loss))
        error = p - y
        w -= data.learning_rate * (xs_train.T @ error) / n
        b -= data.learning_rate * float(np.sum(error)) / n

    w_orig, b_orig = _to_original_scale(w, b, mean, std)
    probabilities = _sigmoid(x_test @ w_orig + b_orig)
    predictions = (probabilities >= 0.5).astype(int)
    return ClassificationResult(weights=w_orig, bias=float(b_orig),
                                loss_history=history,
                                probabilities=probabilities,
                                predictions=predictions)


def classification_metrics(y_true, proba, pred) -> ClassificationMetrics:
    y_true = np.asarray(y_true).ravel()
    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(y_true, proba)),
        accuracy=float(accuracy_score(y_true, pred)),
        precision=float(precision_score(y_true, pred, zero_division=0)),
        recall=float(recall_score(y_true, pred, zero_division=0)),
        f1=float(f1_score(y_true, pred, zero_division=0)),
    )


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Готовые LogisticRegression, DecisionTree и KNN на том же разбиении."""
    x_train = _as_2d(data.train_features)
    x_test = _as_2d(data.test_features)
    y_train = np.asarray(data.train_target).ravel()
    y_test = np.asarray(data.test_target).ravel()

    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)),
        "decision_tree": DecisionTreeClassifier(
            max_depth=5, random_state=RANDOM_STATE),
        "knn": make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    }
    results = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        proba = model.predict_proba(x_test)[:, 1]
        pred = model.predict(x_test)
        results[name] = classification_metrics(y_test, proba, pred)
    return BaselineComparisonResult(**results)
