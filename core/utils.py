import gc
import json
import random

import numpy as np
import torch
from torch import nn


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def init_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_glove_from_hub(repo_id: str = "sentence-transformers/average_word_embeddings_glove.6B.300d"
                        ) -> tuple[dict, torch.Tensor]:
    """GloVe 6B из HuggingFace Hub (вместо ручного скачивания архива, как load_glove_fixed_vocab в шаблоне).
    Возвращает словарь {слово: индекс} и матрицу весов."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    glove_dir = snapshot_download(repo_id) + "/0_WordEmbeddings"
    with open(f"{glove_dir}/whitespacetokenizer_config.json", encoding="utf-8") as f:
        glove_vocab = json.load(f)["vocab"]
    weights = load_file(f"{glove_dir}/model.safetensors")["emb_layer.weight"]
    return {w: i for i, w in enumerate(glove_vocab)}, weights


def build_embedding_matrix(token_to_id: dict, glove_index: dict, glove_weights: torch.Tensor,
                           pad_id: int = 0, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Матрица эмбеддингов под словарь SequenceVectorizer: векторы GloVe для найденных слов,
    случайные — для остальных, нули — для PAD. Второй результат — маска «слово есть в GloVe»."""
    gen = torch.Generator().manual_seed(seed)
    vocab_size = max(token_to_id.values()) + 1
    matrix = torch.randn(vocab_size, glove_weights.shape[1], generator=gen) * glove_weights.std().item()
    matrix[pad_id] = 0
    in_glove = torch.zeros(vocab_size, dtype=torch.bool)
    for token, idx in token_to_id.items():
        j = glove_index.get(token)
        if j is not None and idx > 1:
            matrix[idx] = glove_weights[j]
            in_glove[idx] = True
    return matrix, in_glove
