"""Small graph and tabular rankers for exact-label candidate ordering."""
from __future__ import annotations

from pathlib import Path
import random
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from ..interventions import Delivery, PatchProblem
from ..world import Snapshot
from .graph import FEATURE_DIM, GLOBAL_DIM, N_RELATIONS, GraphData, encode_graph, tabular_features

Batch = tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
MODEL_KINDS = {"graph", "mlp", "graph_no_provenance"}


def configure(seed: int) -> None:
    """Set the deterministic CPU training environment used by the release."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def collate(graphs: Sequence[GraphData]) -> Batch:
    """Pad graphs in a batch and build mean-normalized relation matrices."""
    batch_size = len(graphs)
    max_nodes = max(len(graph.x) for graph in graphs)

    features = torch.zeros((batch_size, max_nodes, FEATURE_DIM), dtype=torch.float32)
    adjacency = torch.zeros(
        (batch_size, N_RELATIONS, max_nodes, max_nodes), dtype=torch.float32
    )
    node_mask = torch.zeros((batch_size, max_nodes), dtype=torch.float32)
    global_features = torch.from_numpy(np.stack([graph.global_features for graph in graphs]))
    target_indices = torch.tensor([graph.target_index for graph in graphs], dtype=torch.long)
    destination_indices = torch.tensor(
        [graph.destination_index for graph in graphs], dtype=torch.long
    )
    labels = torch.tensor([graph.y for graph in graphs], dtype=torch.float32)
    tabular = torch.from_numpy(np.stack([tabular_features(graph) for graph in graphs]))

    for batch_index, graph in enumerate(graphs):
        node_count = len(graph.x)
        features[batch_index, :node_count] = torch.from_numpy(graph.x)
        node_mask[batch_index, :node_count] = 1.0

        if len(graph.edges):
            edge_data = graph.edges
            adjacency[
                batch_index,
                edge_data[:, 0],
                edge_data[:, 1],
                edge_data[:, 2],
            ] = 1.0

    # Each destination averages its incoming neighbors separately by relation.
    adjacency = adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return (
        features,
        adjacency,
        node_mask,
        global_features,
        target_indices,
        destination_indices,
        tabular,
        labels,
    )


class GraphRanker(nn.Module):
    """Relational message-passing model that scores one proposed patch."""

    def __init__(self, hidden: int = 32, layers: int = 2) -> None:
        super().__init__()
        self.hidden = hidden
        self.layers = layers
        self.input = nn.Linear(FEATURE_DIM, hidden)
        self.self_layers = nn.ModuleList(
            nn.Linear(hidden, hidden) for _ in range(layers)
        )
        self.relations = nn.ParameterList(
            nn.Parameter(torch.empty(N_RELATIONS, hidden, hidden)) for _ in range(layers)
        )
        for relation_weights in self.relations:
            nn.init.xavier_uniform_(relation_weights)

        self.head = nn.Sequential(
            nn.Linear(hidden * 3 + GLOBAL_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, batch: Batch) -> Tensor:
        features, adjacency, node_mask, global_features, targets, destinations, _, _ = batch
        hidden = torch.relu(self.input(features)) * node_mask.unsqueeze(-1)

        for self_layer, relation_weights in zip(self.self_layers, self.relations):
            # adjacency[b, r, v, u] sends node u's features to node v.
            messages = torch.einsum("brij,bjh->brih", adjacency, hidden)
            aggregated = torch.einsum("brih,rhk->bik", messages, relation_weights)
            hidden = torch.relu(self_layer(hidden) + aggregated) * node_mask.unsqueeze(-1)

        graph_embedding = hidden.sum(dim=1) / node_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        batch_indices = torch.arange(len(hidden))
        selected_embeddings = torch.cat(
            (
                graph_embedding,
                hidden[batch_indices, targets],
                hidden[batch_indices, destinations],
                global_features,
            ),
            dim=-1,
        )
        return self.head(selected_embeddings).squeeze(-1)


class MLPRanker(nn.Module):
    """Feature-matched baseline that receives no graph edges."""

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.hidden = hidden
        self.net = nn.Sequential(
            nn.Linear(GLOBAL_DIM + FEATURE_DIM * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, batch: Batch) -> Tensor:
        tabular = batch[6]
        return self.net(tabular).squeeze(-1)


class Predictor:
    """Encode candidate patches and return model logits without verifying them."""

    def __init__(self, model: nn.Module, kind: str, metadata: dict[str, Any] | None = None):
        self.model = model.eval()
        self.kind = kind
        self.metadata = metadata or {}

    def score(
        self,
        snapshot: Snapshot,
        problem: PatchProblem,
        patches: Sequence[Sequence[Delivery]],
    ) -> list[float]:
        if not patches:
            return []

        scores: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(patches), 64):
                patch_batch = patches[start : start + 64]
                graphs = [
                    encode_graph(
                        snapshot,
                        problem,
                        patch,
                        strip_provenance=self.kind == "graph_no_provenance",
                    )
                    for patch in patch_batch
                ]
                scores.extend(self.model(collate(graphs)).cpu().tolist())
        return scores

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "kind": self.kind,
            "metadata": self.metadata,
            "schema_version": 1,
            "hidden": getattr(self.model, "hidden", 32),
            "layers": getattr(self.model, "layers", 2),
            "mlp_hidden": getattr(self.model, "hidden", 64) if self.kind == "mlp" else 64,
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str | Path) -> Predictor:
        # weights_only=True avoids arbitrary Python execution during deserialization.
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("schema_version") != 1:
            raise ValueError("Unsupported checkpoint version")

        kind = checkpoint["kind"]
        if kind not in MODEL_KINDS:
            raise ValueError("Unknown checkpoint kind")

        if kind == "mlp":
            model = MLPRanker(checkpoint.get("mlp_hidden", 64))
        else:
            model = GraphRanker(checkpoint["hidden"], checkpoint["layers"])
        model.load_state_dict(checkpoint["state_dict"])
        return cls(model, kind, checkpoint.get("metadata", {}))
