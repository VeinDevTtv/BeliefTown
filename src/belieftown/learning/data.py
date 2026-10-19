"""Generate whole-scene splits and labels from exact patch execution."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

from ..interventions import PatchProblem, candidate_deliveries, verify_patch
from ..randomness import keyed_rng
from ..scenarios import FAMILIES, procedural_scene
from ..schema import Scene
from ..world import Episode
from .graph import GraphData, encode_graph


@dataclass
class Case:
    """One complete scene, its intervention goal, and exact patch labels."""

    scene: Scene
    problem: PatchProblem
    split: str
    labels: list[dict[str, Any]]

    @property
    def snapshot(self):
        """Return the scene state at the first active search decision."""
        return Episode(self.scene).at(self.scene.search_start)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene": self.scene.to_dict(),
            "problem": asdict(self.problem),
            "split": self.split,
            "labels": self.labels,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Case:
        problem_data = dict(value["problem"])
        problem_data["protected_actors"] = tuple(problem_data.get("protected_actors", ()))
        return cls(
            scene=Scene.from_dict(value["scene"]),
            problem=PatchProblem(**problem_data),
            split=value["split"],
            labels=value["labels"],
        )


def make_dataset(
    directory: str | Path,
    *,
    train: int = 120,
    validation: int = 24,
    test: int = 48,
    ood: int = 24,
    seed: int = 20261018,
) -> dict[str, Any]:
    """Write scene-disjoint splits with labels from the exact verifier."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "seed": seed,
        "split_unit": "entire scene, all candidate patches together",
        "protocol": "All patches up to cardinality 3 labeled by exact execution; no sample-level random split.",
        "splits": {},
    }
    scene_hashes: set[str] = set()
    split_specs = (
        ("train", train, 0),
        ("validation", validation, 100000),
        ("test", test, 200000),
        ("ood", ood, 300000),
    )

    for split, scene_count, seed_offset in split_specs:
        path = directory / f"{split}.jsonl"
        stats: dict[str, Any] = {
            "scenes": scene_count,
            "candidates": 0,
            "positive_candidates": 0,
            "family_counts": {},
        }

        with path.open("w", encoding="utf-8") as output:
            for index in range(scene_count):
                family = FAMILIES[index % len(FAMILIES)]
                case_seed = seed + seed_offset + index
                scene = procedural_scene(
                    case_seed,
                    family,
                    n_rooms=(5 + index % 2 if split == "ood" else 4),
                    ood=split == "ood",
                )
                snapshot = Episode(scene).at(scene.search_start)

                target_rng = keyed_rng(case_seed, "target")
                target_room = int(target_rng.integers(len(scene.rooms)))
                # Keep some zero-delivery targets while ensuring most cases
                # ask the character to change its current action.
                current_room = snapshot.plan("courier").destination
                if index % 10 and target_room == current_room:
                    target_room = (target_room + 1) % len(scene.rooms)

                protected_actors = (
                    ("courier",)
                    if family == "constrained_patch"
                    and (index // len(FAMILIES)) % 2
                    else ()
                )
                problem = PatchProblem("courier", target_room, 3, protected_actors, 24)
                candidates = candidate_deliveries(snapshot, problem)
                labels = []

                for size in range(min(3, len(candidates)) + 1):
                    for candidate_indices in itertools.combinations(range(len(candidates)), size):
                        patch = tuple(candidates[candidate] for candidate in candidate_indices)
                        result = verify_patch(snapshot, problem, patch)
                        labels.append(
                            {
                                "indices": list(candidate_indices),
                                "feasible": result.feasible,
                            }
                        )

                case = Case(scene, problem, split, labels)
                output.write(
                    json.dumps(
                        case.to_dict(),
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )

                stats["candidates"] += len(labels)
                stats["positive_candidates"] += sum(label["feasible"] for label in labels)
                family_counts = stats["family_counts"]
                family_counts[family] = family_counts.get(family, 0) + 1

                if scene.hash in scene_hashes:
                    raise RuntimeError("Scene split leakage")
                scene_hashes.add(scene.hash)

        stats["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["splits"][split] = stats

    manifest["scene_hash_count"] = len(scene_hashes)
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_cases(path: str | Path) -> list[Case]:
    """Read JSONL cases while preserving one row per complete scene."""
    with Path(path).open(encoding="utf-8") as source:
        return [Case.from_dict(json.loads(line)) for line in source if line.strip()]


def training_graphs(
    cases: list[Case], *, strip_provenance: bool = False
) -> list[GraphData]:
    """Encode every labeled patch; no candidate is sampled or dropped."""
    graphs: list[GraphData] = []
    for case in cases:
        snapshot = case.snapshot
        candidates = candidate_deliveries(snapshot, case.problem)

        for label in case.labels:
            patch = tuple(candidates[index] for index in label["indices"])
            graph = encode_graph(
                snapshot,
                case.problem,
                patch,
                strip_provenance=strip_provenance,
            )
            graph.y = float(label["feasible"])
            graphs.append(graph)

    return graphs
