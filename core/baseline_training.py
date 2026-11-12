import time
from typing import Any, Optional

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score, \
    accuracy_score

import core.visualization as p
from core.training_results import ClassificationMetrics, RocCurveData, ClassificationReportRow, \
    MultipleModelsResults


def _calculate_classification_metrics(y_true: Any, y_pred: Any, y_probs: Optional[Any] = None) -> ClassificationMetrics:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    metrics = ClassificationMetrics(
        roc_auc=roc_auc_score(y_true, y_probs) if y_probs is not None else None,
        f1_score=f1_score(y_true, y_pred, average='macro', zero_division=0),
        precision=precision_score(y_true, y_pred, average='macro', zero_division=0),
        recall=recall_score(y_true, y_pred, average='macro', zero_division=0),
        accuracy=accuracy_score(y_true, y_pred),
        confusion_matrix=confusion_matrix(y_true, y_pred)
    )
    metrics.classification_report = [
        ClassificationReportRow(class_label='Positive',
                                precision=precision_score(y_true, y_pred, pos_label=1, zero_division=0),
                                recall=recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        ClassificationReportRow(class_label='Negative',
                                precision=precision_score(y_true, y_pred, pos_label=0, zero_division=0),
                                recall=recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
    ]
    if y_probs is not None:
        fpr, tpr, thresholds = roc_curve(y_true, y_probs)
        metrics.roc_curve = RocCurveData(fpr=fpr, tpr=tpr, thresholds=thresholds)
    return metrics


def evaluate_classification(y_true: Any, y_pred: Any, y_probs: Optional[Any] = None, model_name: str = "Model",
                            enable_plot: bool = True) -> ClassificationMetrics:
    metrics = _calculate_classification_metrics(y_true, y_pred, y_probs)
    metrics.name = model_name
    if enable_plot:
        p.plot_classification_results(metrics, model_name)
        p.print_classification_report(metrics.to_report_dict(), model_name)
    return metrics


def evaluate_probas(results: MultipleModelsResults, name: str, group: str, y_true: Any, proba: Any,
                    training_time: Optional[float] = None, threshold: float = 0.5,
                    enable_plot: bool = False) -> ClassificationMetrics:
    """Оценка по вероятностям положительного класса на тесте + запись в общий реестр результатов."""
    proba = np.asarray(proba)
    metrics = evaluate_classification(y_true, (proba >= threshold).astype(int), proba, name, enable_plot)
    metrics.group = group
    metrics.training_time = training_time
    results.add(metrics)
    print(f"{name:<55} acc={metrics.accuracy:.4f}  f1={metrics.f1_score:.4f}  roc_auc={metrics.roc_auc:.4f}")
    return metrics


def fit_evaluate_sklearn(results: MultipleModelsResults, name: str, group: str, model: BaseEstimator,
                         X_train: Any, y_train: Any, X_test: Any, y_test: Any,
                         enable_plot: bool = False) -> tuple[BaseEstimator, np.ndarray]:
    """Обучение sklearn-модели (обычно Pipeline векторизатор + классификатор) на train и оценка на test."""
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0
    proba = model.predict_proba(X_test)[:, 1]
    evaluate_probas(results, name, group, y_test, proba, fit_time, enable_plot=enable_plot)
    return model, proba
