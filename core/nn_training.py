"""Обучение PyTorch-моделей. По сравнению с fine_tune_and_validate из шаблона:
- отдельная валидация и ранняя остановка по ROC-AUC (восстанавливаются веса лучшей эпохи);
- оценка на отложенном тесте, а не на обучающих данных;
- опционально: линейный warmup/decay и bf16 autocast (для трансформера)."""
import copy
from time import perf_counter
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, accuracy_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from core.utils import get_device


def _to_loader(X: Any, y: Optional[Any], batch_size: int, shuffle: bool) -> DataLoader:
    tensors = [torch.as_tensor(np.asarray(X), dtype=torch.long)]
    if y is not None:
        tensors.append(torch.as_tensor(np.asarray(y), dtype=torch.float32))
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def predict_proba(model: nn.Module, X: Any, batch_size: int = 256, device=None, use_amp: bool = False) -> np.ndarray:
    device = device or get_device()
    model.eval()
    out = []
    for (X_batch,) in _to_loader(X, None, batch_size, shuffle=False):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp and device.type == "cuda"):
            logits = model(X_batch.to(device))
        out.append(torch.sigmoid(logits.float()).cpu())
    return torch.cat(out).numpy()


def train_model(model: nn.Module, X_train: Any, y_train: Any, X_val: Any, y_val: Any, *,
                num_epochs: int = 10, batch_size: int = 128, lr: float = 1e-3, weight_decay: float = 1e-4,
                patience: int = 3, warmup_ratio: Optional[float] = None, use_amp: bool = False,
                device=None, verbose: bool = True) -> tuple[nn.Module, pd.DataFrame, float]:
    """Возвращает модель с весами лучшей по val ROC-AUC эпохи, историю обучения и время обучения (с)."""
    device = device or get_device()
    model = model.to(device)
    train_loader = _to_loader(X_train, y_train, batch_size, shuffle=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=lr, weight_decay=weight_decay)
    scheduler = None
    if warmup_ratio is not None:
        from transformers import get_linear_schedule_with_warmup
        total = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, int(warmup_ratio * total), total)
    criterion = nn.BCEWithLogitsLoss()
    y_val = np.asarray(y_val)

    history, best_auc, best_state, bad = [], -1.0, None, 0
    start = perf_counter()
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp and device.type == "cuda"):
                logits = model(X_batch)
            loss = criterion(logits.float(), y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.item() * len(y_batch)

        val_proba = predict_proba(model, X_val, device=device, use_amp=use_amp)
        val_auc = roc_auc_score(y_val, val_proba)
        val_acc = accuracy_score(y_val, val_proba >= 0.5)
        history.append({"epoch": epoch, "train_loss": total_loss / len(train_loader.dataset),
                        "val_acc": val_acc, "val_auc": val_auc, "time_s": perf_counter() - start})
        if verbose:
            print(f"epoch {epoch}: train_loss={history[-1]['train_loss']:.4f}  val_acc={val_acc:.4f}  "
                  f"val_auc={val_auc:.4f}  ({history[-1]['time_s']:.0f}s)")
        if val_auc > best_auc:
            best_auc, bad = val_auc, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print("ранняя остановка")
                break
    model.load_state_dict(best_state)
    if verbose:
        print(f"лучшая эпоха: {int(pd.DataFrame(history)['val_auc'].idxmax()) + 1}")
    return model, pd.DataFrame(history), perf_counter() - start
