from typing import Optional, Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from core.training_results import ClassificationMetrics, BaseMetrics

DEFAULT_FIGSIZE = (8, 6)
DEFAULT_PALETTE = "viridis"
DEFAULT_GRID_ALPHA = 0.3
NEG_POS_COLORS = ["#C44E52", "#55A868"]


def _setup_plot_style(figsize: tuple[int, int] = DEFAULT_FIGSIZE) -> None:
    plt.figure(figsize=figsize)
    plt.grid(alpha=DEFAULT_GRID_ALPHA)


def _finalize_plot(title: str, x_label: str = "", y_label: str = "") -> None:
    if title:
        plt.title(title, fontsize=14, fontweight='bold')
    if x_label:
        plt.xlabel(x_label, fontsize=12)
    if y_label:
        plt.ylabel(y_label, fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_histogram_numeric(data: pd.DataFrame, feature: str, hue: Optional[str] = None,
                           figsize: tuple[int, int] = (10, 4), bins: int = 80,
                           x_min: Optional[float] = None, x_max: Optional[float] = None,
                           title: Optional[str] = None) -> None:
    filtered = data
    if x_min is not None:
        filtered = filtered[filtered[feature] >= x_min]
    if x_max is not None:
        filtered = filtered[filtered[feature] <= x_max]
    _setup_plot_style(figsize=figsize)
    sns.histplot(data=filtered, x=feature, hue=hue, bins=bins, element="step" if hue else "bars", kde=hue is None)
    _finalize_plot(title or f'Distribution of {feature}', feature, 'Frequency')


def barplot(category_counts: pd.Series, title: str, y_label: str, figsize: tuple[int, int] = (5, 6),
            top_n: Optional[int] = None, color_palette: str = DEFAULT_PALETTE) -> None:
    plot_data = category_counts.nlargest(top_n) if top_n is not None else category_counts
    plt.figure(figsize=figsize)
    plt.grid(axis='x', alpha=DEFAULT_GRID_ALPHA)
    sns.barplot(x=plot_data.values, y=plot_data.index.astype(str), hue=plot_data.index.astype(str),
                palette=color_palette, orient='h', legend=False, dodge=False)
    _finalize_plot(title, 'Frequency', y_label)


def plot_wordcloud(token_counts: pd.Series, title: str = "Word Cloud", figsize: tuple[int, int] = (12, 6),
                   max_words: int = 100, background_color: str = "white", colormap: str = "viridis",
                   width: int = 800, height: int = 400, ax=None) -> None:
    from wordcloud import WordCloud
    wordcloud = WordCloud(background_color=background_color, width=width, height=height, max_words=max_words,
                          colormap=colormap, relative_scaling=0.5, random_state=42
                          ).generate_from_frequencies(token_counts.to_dict())
    standalone = ax is None
    if standalone:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight='bold')
    if standalone:
        plt.tight_layout()
        plt.show()


def plot_pie_chart(data: pd.DataFrame, column: str, title: str) -> None:
    counts = data[column].value_counts(dropna=False)
    colors = sns.color_palette(DEFAULT_PALETTE, len(counts))
    plt.figure(figsize=(5, 5))
    _, texts, autotexts = plt.pie(counts, labels=counts.index.astype(str), autopct="%1.1f%%", colors=colors,
                                  startangle=140, wedgeprops={"edgecolor": "white", "linewidth": 1})
    plt.setp(autotexts, size=12, weight="bold", color="black")
    plt.setp(texts, size=12)
    plt.title(title, fontsize=14, weight="bold")
    plt.show()


def plot_signed_scores(names: Any, scores: Any, title: str, x_label: str, top_n: int = 15,
                       figsize: tuple[int, int] = (8, 7)) -> pd.DataFrame:
    """Top-n самых отрицательных и самых положительных значений (веса линейной модели, log ratio и т.п.)."""
    names, scores = np.asarray(names), np.asarray(scores)
    order = np.argsort(scores)
    idx = np.r_[order[:top_n], order[-top_n:]]
    df = pd.DataFrame({"feature": names[idx], "score": scores[idx]})
    plt.figure(figsize=figsize)
    sns.barplot(data=df, y="feature", x="score", hue=df["score"] > 0, palette=NEG_POS_COLORS, legend=False)
    _finalize_plot(title, x_label, "")
    return df


def plot_classification_results(metrics: ClassificationMetrics, model_name: str = "Model") -> None:
    plt.figure(figsize=(12, 4.5))
    if metrics.confusion_matrix is not None:
        plt.subplot(1, 2, 1)
        sns.heatmap(metrics.confusion_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted Negative', 'Predicted Positive'],
                    yticklabels=['Actual Negative', 'Actual Positive'])
        plt.title(f'{model_name} - Confusion Matrix', fontsize=12)
    if metrics.roc_curve is not None and metrics.roc_auc is not None:
        plt.subplot(1, 2, 2)
        plt.plot(metrics.roc_curve.fpr, metrics.roc_curve.tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {metrics.roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic', fontsize=12)
        plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def plot_metrics_heatmap(metrics: list[BaseMetrics] | pd.DataFrame,
                         title: str = 'Model Evaluation Metrics Comparison',
                         figsize: tuple[int, int] = (12, 6)) -> None:
    if isinstance(metrics, pd.DataFrame):
        metrics_df = metrics.select_dtypes("number")
    else:
        metrics_df = pd.DataFrame.from_dict({m.name or f'Model {i + 1}': m.to_compact_dict()
                                             for i, m in enumerate(metrics)}, orient='index')
    normalized_df = metrics_df.copy()
    for column in metrics_df.columns:
        lo, hi = metrics_df[column].min(), metrics_df[column].max()
        normalized_df[column] = (metrics_df[column] - lo) / (hi - lo) if hi != lo else 0.5
    plt.figure(figsize=figsize)
    sns.heatmap(normalized_df, annot=metrics_df, fmt='.3f', cmap='RdBu_r',
                cbar_kws={'label': 'Normalized Score (0-1 per metric)'})
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_models_bar(summary: pd.DataFrame, metric: str = "F1 Score", group_col: str = "Group",
                    title: Optional[str] = None, figsize: tuple[int, int] = (10, 8)) -> None:
    plot_df = summary.reset_index().sort_values(metric)
    model_col = plot_df.columns[0]
    plt.figure(figsize=figsize)
    sns.barplot(data=plot_df, y=model_col, x=metric, hue=group_col, dodge=False, palette="tab10")
    plt.xlim(plot_df[metric].min() - 0.02, 1.0)
    for i, v in enumerate(plot_df[metric]):
        plt.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=9)
    plt.legend(title=group_col, loc="upper left", bbox_to_anchor=(1.01, 1))
    _finalize_plot(title or f"{metric} на тестовой выборке", metric, "")


def plot_histories(histories: dict[str, pd.DataFrame], figsize: tuple[int, int] = (14, 4)) -> None:
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for name, h in histories.items():
        axes[0].plot(h["epoch"], h["train_loss"], marker="o", label=name)
        axes[1].plot(h["epoch"], h["val_acc"], marker="o", label=name)
    axes[0].set_title("Train loss")
    axes[1].set_title("Validation accuracy")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=DEFAULT_GRID_ALPHA)
        ax.legend()
    plt.tight_layout()
    plt.show()


def print_classification_report(metrics: Dict[str, Any], model_name: str = 'Model'):
    metrics_df = pd.DataFrame({
        'Metric': ['ROC AUC', 'F1 Score', 'Precision', 'Recall', 'Accuracy'],
        'Value': [f'{metrics["ROC AUC"]:.4f}' if metrics["ROC AUC"] is not None else 'N/A',
                  f'{metrics["F1 Score"]:.4f}', f'{metrics["Precision"]:.4f}',
                  f'{metrics["Recall"]:.4f}', f'{metrics["Accuracy"]:.4f}']
    })
    print("\n" + "=" * 60)
    print(f"{model_name.upper()} EVALUATION".center(60))
    print("=" * 60)
    print("\nMAIN METRICS:")
    print(metrics_df.to_string(index=False))
    if 'Classification Report' in metrics:
        print("\n\nCLASSIFICATION REPORT:")
        print(pd.DataFrame(metrics['Classification Report']).to_string(index=False))
    print("\n" + "=" * 60)
