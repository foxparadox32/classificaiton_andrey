from collections import Counter
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class SequenceVectorizer(BaseEstimator, TransformerMixin):
    """Списки токенов -> матрица индексов фиксированной длины (из шаблона).
    Изменения: keep_oov — незнакомые слова заменяются на <OOV>, а не выбрасываются;
    по умолчанию паддинг справа (нужно для pack_padded_sequence в BiLSTM)."""

    def __init__(self, min_frequency: int = 1, max_vocab_size: Optional[int] = None,
                 pad_token: str = "<PAD>", oov_token: str = "<OOV>",
                 pad_left: bool = False, max_length: Optional[int] = None,
                 keep_oov: bool = True, dtype: str = "int64", vocabulary: Optional[dict] = None) -> None:
        self.min_frequency = min_frequency
        self.max_vocab_size = max_vocab_size
        self.pad_token = pad_token
        self.oov_token = oov_token
        self.pad_left = pad_left
        self.max_length = max_length
        self.keep_oov = keep_oov
        self.dtype = dtype
        self.vocabulary = vocabulary

    pad_token_id_ = 0
    oov_token_id_ = 1

    def _set_vocabulary(self, vocabulary: dict) -> None:
        self.token_to_id_ = vocabulary
        max_id = max(vocabulary.values())
        id_to_token = [self.oov_token] * (max_id + 1)
        for token, index in vocabulary.items():
            id_to_token[index] = token
        self.id_to_token_ = id_to_token
        self.vocab_size_ = max_id + 1

    @staticmethod
    def _iter_sequences(X: Any):
        if isinstance(X, (pd.Series, pd.DataFrame)):
            X = X.squeeze(axis=1).tolist() if isinstance(X, pd.DataFrame) else X.tolist()
        for item in X:
            yield [] if item is None else list(item)

    def fit(self, X: Any, y: Optional[Any] = None):
        if self.vocabulary is not None:
            self._set_vocabulary(self.vocabulary)
        else:
            counts = Counter(t for seq in self._iter_sequences(X) for t in seq)
            for special in (self.pad_token, self.oov_token):
                counts.pop(special, None)
            items = [(t, c) for t, c in counts.items() if c >= self.min_frequency]
            items.sort(key=lambda x: (-x[1], x[0]))
            if self.max_vocab_size is not None:
                items = items[:max(0, self.max_vocab_size - 2)]
            vocab = {self.pad_token: self.pad_token_id_, self.oov_token: self.oov_token_id_}
            for index, (token, _) in enumerate(items, start=2):
                vocab[token] = index
            self._set_vocabulary(vocab)
        self.sequence_length_ = self.max_length or max(len(s) for s in self._iter_sequences(X))
        return self

    def transform(self, X: Any) -> np.ndarray:
        to_id, oov_id, max_length = self.token_to_id_, self.oov_token_id_, self.sequence_length_
        out = np.full((sum(1 for _ in self._iter_sequences(X)), max_length), self.pad_token_id_, dtype=self.dtype)
        for row, tokens in enumerate(self._iter_sequences(X)):
            if self.keep_oov:
                seq = [to_id.get(t, oov_id) for t in tokens]
            else:
                seq = [to_id[t] for t in tokens if t in to_id]
            seq = seq[-max_length:] if self.pad_left else seq[:max_length]
            if self.pad_left:
                out[row, max_length - len(seq):] = seq
            else:
                out[row, :len(seq)] = seq
        return out

    def inverse_transform(self, X: Any) -> list[list[str]]:
        return [[self.id_to_token_[i] for i in row if i != self.pad_token_id_] for row in np.asarray(X)]

    def get_vocabulary(self) -> dict:
        return dict(self.token_to_id_)


class BertVectorizer(BaseEstimator, TransformerMixin):
    """Тексты -> input_ids токенизатора HuggingFace (из шаблона; fit теперь возвращает self)."""

    def __init__(self, model_name: str, max_length: Optional[int] = None):
        self.model_name = model_name
        self.max_length = max_length

    def fit(self, X: Any, y: Optional[Any] = None):
        from transformers import AutoTokenizer
        self.tokenizer_ = AutoTokenizer.from_pretrained(self.model_name)
        return self

    def transform(self, texts: Any) -> np.ndarray:
        encoded = self.tokenizer_(list(texts), padding="max_length", truncation=True,
                                  max_length=self.max_length, return_tensors="np")
        return encoded["input_ids"]
