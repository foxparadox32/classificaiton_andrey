"""Архитектуры нейросетей. Как в шаблоне, forward принимает только матрицу индексов X,
длины последовательностей и маска считаются по pad_token_id. Все модели (кроме BERT)
поддерживают инициализацию эмбеддингов готовыми векторами (GloVe)."""
from typing import Optional

import torch
import torch.nn as nn


def make_embedding(vocab_size: int, embedding_dim: int, pretrained: Optional[torch.Tensor] = None,
                   freeze: bool = False, pad_token_id: int = 0) -> nn.Embedding:
    emb = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_id)
    if pretrained is not None:
        emb.weight.data.copy_(torch.as_tensor(pretrained, dtype=torch.float32))
        emb.weight.requires_grad = not freeze
    return emb


class FastTextLike(nn.Module):
    """Эмбеддинги слов, усреднённые по тексту (без паддинга), + линейный слой."""

    def __init__(self, vocab_size: int, embedding_dim: int = 128, dropout_rate: float = 0.3,
                 pretrained: Optional[torch.Tensor] = None, freeze: bool = False, pad_token_id: int = 0):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.embedding = make_embedding(vocab_size, embedding_dim, pretrained, freeze, pad_token_id)
        self.fc = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(embedding_dim, 1))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        e = self.embedding(X)                                        # (B, T, D)
        mask = (X != self.pad_token_id).unsqueeze(-1).float()
        mean = (e * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.fc(mean).squeeze(1)


class TextCNN(nn.Module):
    """TextCNN (Kim, 2014): свёртки с окнами 3/4/5 + max-pooling по времени.
    Расширение CNNForHallucinationDetection из шаблона (там одно окно)."""

    def __init__(self, vocab_size: int, embedding_dim: int = 128, num_filters: int = 128,
                 kernel_sizes: tuple[int, ...] = (3, 4, 5), dropout_rate: float = 0.5,
                 pretrained: Optional[torch.Tensor] = None, freeze: bool = False, pad_token_id: int = 0):
        super().__init__()
        self.embedding = make_embedding(vocab_size, embedding_dim, pretrained, freeze, pad_token_id)
        self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, num_filters, k) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        e = self.dropout(self.embedding(X)).transpose(1, 2)          # (B, D, T)
        h = [torch.relu(conv(e)).max(dim=2).values for conv in self.convs]
        return self.fc(self.dropout(torch.cat(h, dim=1))).squeeze(1)


class BiLSTM(nn.Module):
    """Двунаправленная LSTM. В отличие от LSTMForHallucinationDetection из шаблона использует
    pack_padded_sequence (паддинг не попадает в рекуррентность) и конкатенирует последние состояния
    обоих направлений с max-pooling по выходам."""

    def __init__(self, vocab_size: int, embedding_dim: int = 128, lstm_units: int = 128, dropout_rate: float = 0.4,
                 pretrained: Optional[torch.Tensor] = None, freeze: bool = False, pad_token_id: int = 0):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.embedding = make_embedding(vocab_size, embedding_dim, pretrained, freeze, pad_token_id)
        self.lstm = nn.LSTM(embedding_dim, lstm_units, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(lstm_units * 4, 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        lengths = (X != self.pad_token_id).sum(dim=1).clamp(min=1)
        e = self.dropout(self.embedding(X))
        packed = nn.utils.rnn.pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, (h, _) = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, padding_value=-1e4,
                                                  total_length=X.shape[1])
        last = torch.cat([h[-2], h[-1]], dim=1)                      # последние состояния прямого и обратного прохода
        maxpool = out.max(dim=1).values
        return self.fc(self.dropout(torch.cat([last, maxpool], dim=1))).squeeze(1)


class BertClassifier(nn.Module):
    """Трансформер + линейная голова на [CLS] (BertForHallucinationDetection из шаблона)."""

    def __init__(self, model_name: str, freeze_bert: bool = False, dropout_rate: float = 0.1, pad_token_id: int = 0):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(model_name)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)
        self.pad_token_id = pad_token_id

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        attention_mask = (X != self.pad_token_id).long()
        cls_output = self.bert(input_ids=X, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        return self.fc(self.dropout(cls_output)).squeeze(1)
