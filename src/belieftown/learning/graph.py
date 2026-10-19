"""Encode designer-visible evidence as graphs for candidate ranking.

Hidden parcel truth is deliberately excluded from every feature. Relation
indices and feature positions are part of the checkpoint schema, so keep their
order stable when changing this encoder.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..interventions import Delivery, PatchProblem, heuristic_scores
from ..planner import distances
from ..world import Snapshot

FEATURE_DIM = 20
GLOBAL_DIM = 14

# Directed relation types. The reverse relation is explicit where useful.
RELATIONS = (
    "room_edge",
    "actor_at",
    "at_actor",
    "observed_by",
    "observer_of",
    "reports_room",
    "reported_by",
    "root_of_message",
    "message_of_root",
    "forwarded_from",
    "forwarded_to",
    "received_by",
    "received_message",
    "sender_of",
    "sent_by",
    "patch_root",
    "root_patch",
    "patch_recipient",
    "recipient_patch",
    "patch_sender",
    "sender_patch",
)
N_RELATIONS = len(RELATIONS)
PROVENANCE_RELATIONS = {7, 8, 9, 10}

# Node type coordinates within the shared feature vector.
ROOM_NODE = 0
ACTOR_NODE = 1
ROOT_NODE = 2
MESSAGE_NODE = 3
DELIVERY_NODE = 4

# Named feature coordinates make the checkpoint layout easier to audit.
ROOM_BELIEF = 5
TARGET_MARKER = 6
ACTOR_POSITION = 7
ROOM_DISTANCE = 8
ACTOR_ENTROPY = 9
IS_PROTECTED = 10
ROOT_RELIABILITY = 11
EVENT_AGE = 12
ROOT_ALREADY_KNOWN = 13
TARGET_ROOM_LIKELIHOOD = 14
REPORTS_TARGET_ROOM = 15
IS_INSPECTION = 16


@dataclass
class GraphData:
    """One variable-sized evidence graph and its ranking metadata."""

    x: np.ndarray
    edges: np.ndarray  # [edge_count, 3]: (relation, destination, source)
    global_features: np.ndarray
    target_index: int
    destination_index: int
    y: float = 0.0


def encode_graph(
    snapshot: Snapshot,
    problem: PatchProblem,
    patch: Sequence[Delivery],
    *,
    strip_provenance: bool = False,
) -> GraphData:
    """Build an order-invariant graph for one proposed evidence patch."""
    nodes: list[np.ndarray] = []
    indices: dict[tuple[str, object], int] = {}
    edges: list[tuple[int, int, int]] = []

    def add_node(key: tuple[str, object], node_type: int) -> int:
        node_index = len(nodes)
        indices[key] = node_index
        features = np.zeros(FEATURE_DIM, dtype=np.float32)
        features[node_type] = 1.0
        nodes.append(features)
        return node_index

    def add_edge(relation: int, source: int, destination: int) -> None:
        # Adjacency is stored as (relation, destination, source) for batching.
        edges.append((relation, destination, source))

    room_count = len(snapshot.scene.rooms)
    target_belief = snapshot.belief(problem.target_actor)
    distance_matrix, _ = distances(room_count, snapshot.scene.edges)
    distance_scale = max(1.0, float(distance_matrix.max()))
    target_position = snapshot.positions[problem.target_actor]
    actors = sorted(snapshot.scene.actors, key=lambda actor: actor.id)
    known_roots = snapshot.known(problem.target_actor)
    roots = sorted(snapshot.roots.values(), key=lambda root: root.id)
    messages = sorted(snapshot.messages.values(), key=lambda message: message.id)

    for room in range(room_count):
        node_index = add_node(("room", room), ROOM_NODE)
        features = nodes[node_index]
        features[ROOM_BELIEF] = target_belief[room]
        features[TARGET_MARKER] = float(room == problem.target_room)
        features[ACTOR_POSITION] = float(room == target_position)
        features[ROOM_DISTANCE] = distance_matrix[target_position, room] / distance_scale

    for actor in actors:
        node_index = add_node(("actor", actor.id), ACTOR_NODE)
        features = nodes[node_index]
        features[TARGET_MARKER] = float(actor.id == problem.target_actor)
        belief = snapshot.belief(actor.id)
        positive = belief[belief > 0]
        features[ACTOR_ENTROPY] = float(-np.sum(positive * np.log(positive))) / np.log(room_count)
        features[IS_PROTECTED] = float(actor.id in problem.protected_actors)

        room_index = indices[("room", snapshot.positions[actor.id])]
        add_edge(1, node_index, room_index)
        add_edge(2, room_index, node_index)

    for room_a, room_b, _cost in snapshot.scene.edges:
        index_a = indices[("room", room_a)]
        index_b = indices[("room", room_b)]
        add_edge(0, index_a, index_b)
        add_edge(0, index_b, index_a)

    for root in roots:
        node_index = add_node(("root", root.id), ROOT_NODE)
        features = nodes[node_index]
        age = snapshot.time - root.time
        features[ROOT_RELIABILITY] = root.reliability
        features[EVENT_AGE] = age / max(1, snapshot.time)
        features[ROOT_ALREADY_KNOWN] = float(root.id in known_roots)

        likelihoods = root.probabilities(room_count)
        features[TARGET_ROOM_LIKELIHOOD] = likelihoods[problem.target_room]
        features[REPORTS_TARGET_ROOM] = float(root.reported_room == problem.target_room)
        features[IS_INSPECTION] = float(root.kind.startswith("inspection"))

        actor_index = indices[("actor", root.observer)]
        room_index = indices[("room", root.reported_room)]
        add_edge(3, node_index, actor_index)
        add_edge(4, actor_index, node_index)
        add_edge(5, node_index, room_index)
        add_edge(6, room_index, node_index)

    for message in messages:
        node_index = add_node(("message", message.id), MESSAGE_NODE)
        nodes[node_index][EVENT_AGE] = (snapshot.time - message.time) / max(1, snapshot.time)

        root_index = indices[("root", message.root_id)]
        sender_index = indices[("actor", message.sender)]
        recipient_index = indices[("actor", message.recipient)]
        add_edge(7, root_index, node_index)
        add_edge(8, node_index, root_index)
        add_edge(11, node_index, recipient_index)
        add_edge(12, recipient_index, node_index)
        add_edge(13, sender_index, node_index)
        add_edge(14, node_index, sender_index)

    for message in messages:
        parent_key = ("message", message.parent_id)
        if message.parent_id and parent_key in indices:
            message_index = indices[("message", message.id)]
            parent_index = indices[parent_key]
            add_edge(9, parent_index, message_index)
            add_edge(10, message_index, parent_index)

    for patch_index, delivery in enumerate(sorted(patch)):
        node_index = add_node(("delivery", patch_index), DELIVERY_NODE)
        root = snapshot.roots[delivery.root_id]
        features = nodes[node_index]
        features[ROOT_ALREADY_KNOWN] = float(delivery.root_id in known_roots)
        features[REPORTS_TARGET_ROOM] = float(root.reported_room == problem.target_room)
        features[IS_PROTECTED] = float(delivery.recipient in problem.protected_actors)

        root_index = indices[("root", delivery.root_id)]
        recipient_index = indices[("actor", delivery.recipient)]
        sender_index = indices[("actor", delivery.sender)]
        add_edge(15, root_index, node_index)
        add_edge(16, node_index, root_index)
        add_edge(17, node_index, recipient_index)
        add_edge(18, recipient_index, node_index)
        add_edge(19, sender_index, node_index)
        add_edge(20, node_index, sender_index)

    heuristic_score = heuristic_scores(snapshot, problem, [patch])[0]
    new_roots = {delivery.root_id for delivery in patch} - known_roots
    global_features = np.asarray(
        [
            len(patch) / 3,
            room_count / 7,
            len(roots) / 20,
            len(messages) / 30,
            len(new_roots) / 3,
            target_belief[problem.target_room],
            float(np.max(target_belief)),
            float(problem.target_actor in problem.protected_actors),
            np.tanh(heuristic_score / 5),
            distance_matrix[target_position, problem.target_room] / distance_scale,
            snapshot.time / 20,
            float(snapshot.plan(problem.target_actor).destination == problem.target_room),
            float(bool(patch) and not new_roots),
            sum(snapshot.roots[root_id].reliability for root_id in new_roots) / 3,
        ],
        dtype=np.float32,
    )

    if strip_provenance:
        edges = [edge for edge in edges if edge[0] not in PROVENANCE_RELATIONS]
        # Also remove direct duplicate indicators and related shortcut signals.
        for features in nodes:
            features[ROOT_ALREADY_KNOWN] = 0.0
        global_features[[4, 8, 12, 13]] = 0.0

    edge_array = np.asarray(edges, dtype=np.int64).reshape(-1, 3)
    return GraphData(
        x=np.asarray(nodes, dtype=np.float32),
        edges=edge_array,
        global_features=global_features,
        target_index=indices[("actor", problem.target_actor)],
        destination_index=indices[("room", problem.target_room)],
    )


def tabular_features(graph: GraphData) -> np.ndarray:
    """Return global features and node means/maxima for the MLP baseline."""
    pooled = np.concatenate((graph.x.mean(axis=0), graph.x.max(axis=0)))
    return np.concatenate((graph.global_features, pooled)).astype(np.float32)
