"""Загрузка, очистка и разбиение IMDB — одинаково во всех ноутбуках."""
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 34
DATA_PATH = "IMDB Dataset.csv"
MIN_WORDS = 10


def load_imdb(path: str = DATA_PATH, verbose: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    # HTML-переносы <br /> убираем сразу, чтобы они не считались словами
    df["review"] = df["review"].str.replace(r"<br\s*/?>", " ", regex=True)
    df["n_words"] = df["review"].str.split().str.len()
    n0, min0 = len(df), df["n_words"].min()
    df = df.drop_duplicates(subset="review")
    n1 = len(df)
    df = df[df["n_words"] >= MIN_WORDS].dropna(subset=["review", "sentiment"]).reset_index(drop=True)
    df["label"] = (df["sentiment"] == "positive").astype(int)
    if verbose:
        print(f"всего: {n0} (min слов: {min0}) | без дубликатов: {n1} | "
              f"после фильтра >= {MIN_WORDS} слов: {len(df)} (min слов: {df['n_words'].min()})")
    return df


def split_imdb(df: pd.DataFrame, test_size: float = 0.2, seed: int = SEED):
    """Стратифицированное разбиение train/test 80/20. Тест используется только для финальной оценки."""
    return train_test_split(df["review"], df["label"], test_size=test_size,
                            random_state=seed, stratify=df["label"])


def split_validation(X_train, y_train, val_size: float = 0.1, seed: int = SEED):
    """Валидация для ранней остановки / подбора гиперпараметров — выделяется из train."""
    return train_test_split(X_train, y_train, test_size=val_size, random_state=seed, stratify=y_train)
