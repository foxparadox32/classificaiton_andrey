from typing import Optional, Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator


class BaseMetrics(BaseModel):
    name: Optional[str] = None
    group: Optional[str] = Field(default=None, description="Группа модели по пунктам ТЗ")
    training_time: Optional[float] = Field(default=None, description="Training time in seconds")


class ClassificationReportRow(BaseModel):
    class_label: str
    precision: float
    recall: float


class RocCurveData(BaseModel):
    fpr: Any
    tpr: Any
    thresholds: Any


class ClassificationMetrics(BaseMetrics):
    roc_auc: Optional[float] = Field(default=None, description="Area under ROC curve")
    f1_score: float
    precision: float
    recall: float
    accuracy: float
    confusion_matrix: Any
    classification_report: Optional[list[ClassificationReportRow]] = None
    roc_curve: Optional[RocCurveData] = None

    @field_validator("accuracy", "f1_score", "precision", "recall", mode="before")
    @classmethod
    def _ensure_float(cls, value: Any) -> float:
        try:
            return float(value)
        except Exception:
            return float("nan")

    def get_numeric_metrics(self) -> dict[str, float]:
        metrics = {
            'ROC AUC': float(self.roc_auc) if self.roc_auc is not None else float('nan'),
            'F1 Score': float(self.f1_score),
            'Precision': float(self.precision),
            'Recall': float(self.recall),
            'Accuracy': float(self.accuracy)
        }
        if self.training_time is not None:
            metrics['Training Time (s)'] = float(self.training_time)
        return metrics

    def to_compact_dict(self) -> dict[str, Any]:
        return self.get_numeric_metrics()

    def to_report_dict(self) -> dict[str, Any]:
        report = {
            'ROC AUC': self.roc_auc,
            'F1 Score': self.f1_score,
            'Precision': self.precision,
            'Recall': self.recall,
            'Accuracy': self.accuracy,
        }
        if self.classification_report is not None:
            report['Classification Report'] = {
                'Class': [row.class_label for row in self.classification_report],
                'Precision': [row.precision for row in self.classification_report],
                'Recall': [row.recall for row in self.classification_report],
            }
        return report


class MultipleModelsResults(BaseModel):
    results: list[BaseMetrics] = Field(default_factory=list, description="List of model evaluation results")
    task_type: str = Field(default="classification", description="Type of task")

    def add(self, metrics: BaseMetrics) -> None:
        # повторный запуск ячейки перезаписывает результат модели с тем же именем
        self.results = [r for r in self.results if r.name != metrics.name] + [metrics]

    def to_dataframe(self) -> pd.DataFrame:
        rows = {}
        for result in self.results:
            row = {"Group": result.group}
            row.update(result.to_compact_dict())
            rows[result.name or "Unknown"] = row
        df = pd.DataFrame.from_dict(rows, orient='index')
        df.index.name = "Model"
        return df

    def save_csv(self, path: str) -> pd.DataFrame:
        df = self.to_dataframe()
        df.to_csv(path)
        return df

    def get_models_names(self) -> list[str]:
        return [result.name for result in self.results if result.name]

    def get_best_model(self, metric: str = 'F1 Score') -> Optional[BaseMetrics]:
        best_model, best_score = None, float('-inf')
        for result in self.results:
            score = result.to_compact_dict().get(metric, float('nan'))
            if not np.isnan(score) and score > best_score:
                best_score, best_model = score, result
        return best_model
