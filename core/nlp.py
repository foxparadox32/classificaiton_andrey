import re
import string
from collections import Counter
from functools import lru_cache
from typing import Sequence, Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import f_classif, mutual_info_classif, chi2

# В шаблоне токенизация через tiktoken (BPE); для английских отзывов и словаря GloVe
# нужна токенизация по словам: нижний регистр, отрицание n't отдельным токеном, ! и ? сохраняем.
TOKEN_RE = re.compile(r"n't|'[a-z]+|[a-z0-9]+|[!?]")

# Отрицания и усилители/ограничители — важны для тональности, из списков стоп-слов их исключаем
SENTIMENT_KEEP_WORDS = {"no", "not", "nor", "never", "none", "nothing", "nobody", "neither", "nowhere", "cannot",
                        "against", "very", "too", "much", "least", "less", "few", "more", "most", "only", "but",
                        "off", "out", "don", "didn", "doesn", "isn", "wasn", "weren", "aren", "couldn", "wouldn",
                        "shouldn", "won", "hadn", "hasn", "haven", "n't"}


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower().replace("n't", " n't"))


@lru_cache(maxsize=None)
def get_stopwords_set(language: str = 'english', keep_sentiment_words: bool = False) -> frozenset[str]:
    import nltk
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    words = set(stopwords.words(language))
    if keep_sentiment_words:
        words -= SENTIMENT_KEEP_WORDS
    return frozenset(words)


def _is_punctuation_token(token: str) -> bool:
    stripped = token.strip()
    return stripped != '' and all(c in string.punctuation for c in stripped)


def filter_tokens(tokens: Sequence[str], *, remove_stopwords: bool = False,
                  remove_punctuation_tokens: bool = False, language: str = 'english') -> list[str]:
    stop = get_stopwords_set(language) if remove_stopwords else frozenset()
    return [t for t in tokens
            if t and not (remove_punctuation_tokens and _is_punctuation_token(t)) and t not in stop]


def _flatten_array_column(df: pd.DataFrame, column_name: str) -> list[Any]:
    return [item for sublist in df[column_name] for item in sublist]


def token_counts(df: pd.DataFrame, column: str, remove_stopwords: bool = False,
                 remove_punctuation: bool = False) -> pd.Series:
    tokens = filter_tokens(_flatten_array_column(df, column), remove_stopwords=remove_stopwords,
                           remove_punctuation_tokens=remove_punctuation)
    return pd.Series(Counter(tokens)).sort_values(ascending=False)


def stopwords_counts(df: pd.DataFrame, column: str, language: str = 'english') -> pd.Series:
    stop = get_stopwords_set(language)
    return pd.Series(Counter(t for t in _flatten_array_column(df, column) if t in stop)).sort_values(ascending=False)


def _ngram_matrix(texts_tokenized: list[list[str]], n: int, min_count: int):
    """Разреженная матрица документ × n-грамма (1..n) с отсечением редких n-грамм.
    N-граммы генерируются «на лету» (без промежуточных списков строк на весь корпус), матрица — uint32,
    поэтому на 50k отзывов хватает ~1–2 ГБ оперативки (в шаблоне матрица плотная и не помещается в память)."""
    def analyzer(tokens):
        for k in range(1, n + 1):
            for i in range(len(tokens) - k + 1):
                yield ' '.join(tokens[i:i + k]) if k > 1 else tokens[i]

    vec = CountVectorizer(analyzer=analyzer, dtype=np.uint32)
    X = vec.fit_transform(texts_tokenized).tocsc()
    keep = np.flatnonzero(np.asarray(X.sum(axis=0)).ravel() >= min_count)
    vocab = vec.get_feature_names_out()[keep]
    del vec
    return X[:, keep].tocsr(), vocab


def _score(X, y, metric: str) -> np.ndarray:
    match metric:
        case 'anova_f':
            values = f_classif(X, y)[0]
        case 'chi2':
            values = chi2(X, y)[0]
        case 'mutual_info':
            values = mutual_info_classif(X, y, discrete_features=True, random_state=42)
        case _:
            raise ValueError("metric must be 'anova_f', 'mutual_info' or 'chi2'")
    return np.nan_to_num(values)


def compute_ngram_metrics(texts_tokenized: list[list[str]], labels: Sequence[int], n: int = 1,
                          metric: str = 'anova_f', min_count: int = 1) -> dict[str, float]:
    """Как в шаблоне: оценка информативности n-грамм (1..n) для классификации, по убыванию."""
    X, vocab = _ngram_matrix(texts_tokenized, n, min_count)
    values = _score(X, np.asarray(labels), metric)
    order = np.argsort(-values)
    return {vocab[i]: float(values[i]) for i in order}


def count_based_analysis(texts_tokenized: list[list[str]], labels: Sequence[int], n: int = 1,
                         metric: str = 'anova_f', min_count: int = 1) -> pd.DataFrame:
    """Таблица n-грамм: значение метрики, частоты в каждом классе и log ratio частот (pos / neg).
    Частоты по классам считаются суммированием строк той же разреженной матрицы (один проход по корпусу)."""
    labels = np.asarray(labels)
    X, vocab = _ngram_matrix(texts_tokenized, n, min_count)
    df = pd.DataFrame({'token': vocab, 'metric': _score(X, labels, metric)})
    classes = np.unique(labels)
    for cls in classes:
        counts = np.asarray(X[labels == cls].sum(axis=0, dtype=np.int64)).ravel()
        df[f'count_{cls}'] = counts
        df[f'freq_{cls}'] = counts / counts.sum()
    del X
    df['total_count'] = df[[f'count_{cls}' for cls in classes]].sum(axis=1)
    if set(classes) == {0, 1}:
        df['log_ratio'] = np.log(((df['count_1'] + 1) / (df['count_1'].sum() + 1)) /
                                 ((df['count_0'] + 1) / (df['count_0'].sum() + 1)))
    return df.sort_values('metric', ascending=False).reset_index(drop=True)
