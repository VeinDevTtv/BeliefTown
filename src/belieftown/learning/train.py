"""Train rankers deterministically and select checkpoints on validation loss."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .data import load_cases, training_graphs
from .model import MODEL_KINDS, GraphRanker, MLPRanker, Predictor, collate, configure


def train_model(
    data_dir: str | Path,
    output: str | Path,
    *,
    kind: str = "graph",
    seed: int = 0,
    epochs: int = 18,
    batch_size: int = 64,
    learning_rate: float = 0.002,
) -> Predictor:
    """Train one model variant using exact execution labels.

    Scenes define the split unit. Validation chooses the saved epoch; test and
    out-of-distribution splits are never loaded by this training function.
    """
    if kind not in MODEL_KINDS:
        raise ValueError("Unknown model kind")
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("Invalid training hyperparameters")

    configure(seed)
    start_time = time.perf_counter()
    data_dir = Path(data_dir)
    strip_provenance = kind == "graph_no_provenance"

    train_graphs = training_graphs(
        load_cases(data_dir / "train.jsonl"),
        strip_provenance=strip_provenance,
    )
    validation_graphs = training_graphs(
        load_cases(data_dir / "validation.jsonl"),
        strip_provenance=strip_provenance,
    )
    if not train_graphs or not validation_graphs:
        raise ValueError("Nonempty train and validation splits are required")

    model: nn.Module = MLPRanker() if kind == "mlp" else GraphRanker()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    loss_function = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    history: list[dict[str, float | int]] = []
    best_validation_loss = float("inf")
    best_state: dict[str, Tensor] | None = None
    best_epoch = 0

    # Validation batches are deterministic: the model uses no dropout or
    # stochastic augmentation, so they can be prepared once.
    validation_batches = [
        collate(validation_graphs[start : start + batch_size])
        for start in range(0, len(validation_graphs), batch_size)
    ]

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        shuffled_indices = rng.permutation(len(train_graphs))

        for start in range(0, len(shuffled_indices), batch_size):
            batch_indices = shuffled_indices[start : start + batch_size]
            graph_batch = [train_graphs[index] for index in batch_indices]
            batch = collate(graph_batch)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = loss_function(logits, batch[-1])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(graph_batch)

        model.eval()
        validation_loss = 0.0
        correct_predictions = 0
        with torch.inference_mode():
            for batch in validation_batches:
                logits = model(batch)
                labels = batch[-1]
                validation_loss += float(loss_function(logits, labels)) * len(labels)
                correct_predictions += int(((logits > 0) == labels.bool()).sum())

        validation_loss /= len(validation_graphs)
        history.append(
            {
                "epoch": epoch,
                "train_bce": epoch_loss / len(train_graphs),
                "validation_bce": validation_loss,
                "validation_accuracy": correct_predictions / len(validation_graphs),
            }
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

    assert best_state is not None
    model.load_state_dict(best_state)

    metadata: dict[str, Any] = {
        "seed": seed,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "train_samples": len(train_graphs),
        "validation_samples": len(validation_graphs),
        "validation_bce": best_validation_loss,
        "selection": "lowest validation BCE only; test and OOD splits never used in optimization",
        "torch_version": str(torch.__version__),
        "training_seconds": time.perf_counter() - start_time,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "manifest_sha256": hashlib.sha256(
            (data_dir / "manifest.json").read_bytes()
        ).hexdigest(),
        "history": history,
    }

    predictor = Predictor(model, kind, metadata)
    predictor.save(output)
    Path(output).with_suffix(".json").write_text(
        json.dumps({"kind": kind, **metadata}, indent=2) + "\n",
        encoding="utf-8",
    )
    return predictor
