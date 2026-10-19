"""Paired scene-level evaluation for belief policies and patch rankers."""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Iterable

import numpy as np

from .interventions import search
from .learning.data import load_cases
from .learning.model import Predictor, configure


def bootstrap_mean(
    values: Iterable[float], *, seed: int = 0, repetitions: int = 2000
) -> dict[str, float | int | None]:
    """Return a mean and percentile interval from bootstrap resamples."""
    observations = np.asarray(list(values), dtype=float)
    if not len(observations):
        return {"mean": None, "low": None, "high": None, "n_scenes": 0}

    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(
        0,
        len(observations),
        size=(repetitions, len(observations)),
    )
    bootstrap_means = observations[sample_indices].mean(axis=1)
    return {
        "mean": float(observations.mean()),
        "low": float(np.quantile(bootstrap_means, 0.025)),
        "high": float(np.quantile(bootstrap_means, 0.975)),
        "n_scenes": len(observations),
    }


def _scene_summary(rows: list[dict[str, Any]], metric: str) -> dict[str, float | int | None]:
    """Average seed outcomes within each scene before bootstrapping scenes."""
    by_scene: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if value is not None:
            by_scene[row["scene_id"]].append(float(value))

    scene_means = [np.mean(values) for values in by_scene.values()]
    return bootstrap_mean(scene_means)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize search cost and correctness by split and method."""
    summaries = []
    splits = sorted({row["split"] for row in rows})
    methods = sorted({row["method"] for row in rows})

    for split in splits:
        for method in methods:
            group = [
                row for row in rows
                if row["split"] == split and row["method"] == method
            ]
            if not group:
                continue

            solvable = [row for row in group if row["oracle_minimum"] is not None]
            summaries.append(
                {
                    "split": split,
                    "method": method,
                    "runs": len(group),
                    "verifier_calls": _scene_summary(group, "verifier_calls"),
                    "latency_ms": _scene_summary(group, "latency_ms"),
                    "ranking_ms": _scene_summary(group, "ranking_ms"),
                    "solvable_latency_ms": _scene_summary(solvable, "latency_ms"),
                    "solvable_calls": _scene_summary(solvable, "verifier_calls"),
                    "gap": _scene_summary(solvable, "minimum_gap"),
                    "correctness": _scene_summary(group, "correct"),
                    "success_at_4": _scene_summary(solvable, "success_at_4"),
                    "success_at_16": _scene_summary(solvable, "success_at_16"),
                }
            )

    return summaries


def run_benchmark(
    data_dir: str | Path,
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    *,
    limit: int | None = None,
    seeds: tuple[int, ...] = (0, 1, 2),
) -> dict[str, Any]:
    """Evaluate learned ordering against exact and heuristic search orders."""
    if limit is not None and (
        not isinstance(limit, int) or isinstance(limit, bool) or limit < 1
    ):
        raise ValueError("Benchmark limit must be a positive integer")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide nonempty, distinct model seeds")

    configure(0)
    data_dir = Path(data_dir)
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models: dict[tuple[str, int], Predictor] = {}
    model_kinds = ("graph", "mlp", "graph_no_provenance")
    for kind in model_kinds:
        for seed in seeds:
            checkpoint_path = checkpoint_dir / f"{kind}_seed{seed}.pt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Missing required checkpoint: {checkpoint_path}")
            models[(kind, seed)] = Predictor.load(checkpoint_path)

    rows: list[dict[str, Any]] = []
    belief_rows: list[dict[str, Any]] = []
    start_time = time.perf_counter()
    checkpoint_hashes = {
        f"{kind}_seed{seed}": hashlib.sha256(
            (checkpoint_dir / f"{kind}_seed{seed}.pt").read_bytes()
        ).hexdigest()
        for kind, seed in models
    }

    # Warm tensor operations before timing requests. Model loading is treated
    # as startup cost and is excluded from the steady-state latency measure.
    test_cases = load_cases(data_dir / "test.jsonl")
    if not test_cases or not load_cases(data_dir / "ood.jsonl"):
        raise ValueError("Benchmark requires nonempty test and OOD splits")
    warm_case = test_cases[0]
    for model in models.values():
        model.score(warm_case.snapshot, warm_case.problem, [()])

    for split in ("test", "ood"):
        cases = load_cases(data_dir / f"{split}.jsonl")
        if limit is not None:
            cases = cases[:limit]

        for case_index, case in enumerate(cases):
            snapshot = case.snapshot
            oracle_minimum = min(
                (len(label["indices"]) for label in case.labels if label["feasible"]),
                default=None,
            )
            truth = case.scene.truth[snapshot.time]
            target_distribution = np.eye(len(case.scene.rooms))[truth]

            for mode in ("provenance", "naive", "last_report"):
                belief = snapshot.belief("courier", mode)
                belief_rows.append(
                    {
                        "scene_id": case.scene.id,
                        "split": split,
                        "family": case.scene.family,
                        "mode": mode,
                        "brier": float(np.square(belief - target_distribution).sum()),
                        "true_room_probability": float(belief[truth]),
                        "chosen_room": snapshot.plan("courier", mode).destination,
                    }
                )

            methods = [("uninformed", -1, None), ("heuristic", -1, None)]
            methods.extend(
                (kind, seed, models[(kind, seed)])
                for kind in model_kinds
                for seed in seeds
            )

            for method, model_seed, ranker in methods:
                result = search(
                    snapshot,
                    case.problem,
                    method=method,
                    ranker=ranker,
                    trace=False,
                )
                found_size = None if result.patch is None else len(result.patch)
                exact_result = (
                    result.status == "no_solution_within_bounds"
                    if oracle_minimum is None
                    else result.status == "verified_minimum_within_bounds"
                )
                correct = oracle_minimum == found_size and exact_result

                row = {
                    "scene_id": case.scene.id,
                    "split": split,
                    "family": case.scene.family,
                    "rooms": len(case.scene.rooms),
                    "method": method,
                    "model_seed": model_seed,
                    "oracle_minimum": oracle_minimum,
                    "found_size": found_size,
                    "minimum_gap": (
                        None
                        if oracle_minimum is None or found_size is None
                        else found_size - oracle_minimum
                    ),
                    "status": result.status,
                    "correct": int(correct),
                    "verifier_calls": result.verifier_calls,
                    "candidate_space": result.candidate_space_size,
                    "latency_ms": result.elapsed_seconds * 1000,
                    "ranking_ms": result.ranking_seconds * 1000,
                    # These prefixes measure verifier calls, not elapsed time.
                    "success_at_4": int(
                        found_size is not None and result.verifier_calls <= 4
                    ),
                    "success_at_16": int(
                        found_size is not None and result.verifier_calls <= 16
                    ),
                }
                rows.append(row)
                if not correct:
                    raise AssertionError(f"Exactness regression: {row}")

            if (case_index + 1) % 12 == 0:
                print(f"{split}: {case_index + 1}/{len(cases)} scenes evaluated", flush=True)

    summary = summarize(rows)
    belief_summary = []
    for split in ("test", "ood"):
        for mode in ("provenance", "naive", "last_report"):
            group = [
                row
                for row in belief_rows
                if row["split"] == split and row["mode"] == mode
            ]
            belief_summary.append(
                {
                    "split": split,
                    "mode": mode,
                    "brier": _scene_summary(group, "brier"),
                }
            )

    paired_comparisons = []
    for split in ("test", "ood"):
        heuristic_by_scene = {
            row["scene_id"]: row
            for row in rows
            if row["split"] == split and row["method"] == "heuristic"
        }
        for method in ("uninformed", "graph", "mlp", "graph_no_provenance"):
            group = [
                row for row in rows
                if row["split"] == split and row["method"] == method
            ]
            by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in group:
                by_scene[row["scene_id"]].append(row)

            call_differences = []
            latency_differences = []
            for scene_id, scene_rows in by_scene.items():
                reference = heuristic_by_scene[scene_id]
                call_differences.append(
                    np.mean([row["verifier_calls"] for row in scene_rows])
                    - reference["verifier_calls"]
                )
                latency_differences.append(
                    np.mean([row["latency_ms"] for row in scene_rows])
                    - reference["latency_ms"]
                )

            paired_comparisons.append(
                {
                    "split": split,
                    "method": method,
                    "reference": "heuristic",
                    "calls_difference": bootstrap_mean(call_differences),
                    "latency_ms_difference": bootstrap_mean(latency_differences),
                }
            )

    try:
        import resource

        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_memory_mb = float(peak_rss) / (
            1024**2 if sys.platform == "darwin" else 1024
        )
    except ImportError:
        peak_memory_mb = None

    report = {
        "memory": {
            "peak_process_rss_mb": peak_memory_mb,
            "scope": "whole evaluation process including all loaded checkpoints; not a per-method measurement",
        },
        "schema_version": 1,
        "protocol": {
            "split_unit": "whole scene",
            "training_seeds": list(seeds),
            "confidence_intervals": "95% nonparametric bootstrap over scene means; seeds averaged within scene",
            "latency": "steady-state CPU search, includes graph construction and ranking, excludes model loading and scene creation",
            "budget": "verifier-call prefixes, not measured deadline success",
            "minimum_scope": "recorded candidate universe; max patch size 3",
            "limit": limit,
        },
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_threads": 1,
        },
        "checkpoint_hashes": checkpoint_hashes,
        "dataset_manifest_sha256": hashlib.sha256(
            (data_dir / "manifest.json").read_bytes()
        ).hexdigest(),
        "elapsed_seconds": time.perf_counter() - start_time,
        "summary": summary,
        "belief_summary": belief_summary,
        "paired_comparisons": paired_comparisons,
        "all_searches_correct": all(row["correct"] for row in rows),
        "searches": len(rows),
    }

    (output_dir / "results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    for name, data in (("search_results", rows), ("belief_results", belief_rows)):
        with (output_dir / f"{name}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as output:
            writer = csv.DictWriter(output, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)

    write_report(report, output_dir / "RESULTS.md")
    return report


def write_report(report: dict[str, Any], path: str | Path) -> None:
    """Write a compact Markdown summary alongside the machine-readable data."""
    lines = [
        "# Measured benchmark results",
        "",
        "These are measurements from the bundled finite synthetic benchmark, not claims about arbitrary games.",
        f"The statistical unit is an independent scene. {len(report['protocol']['training_seeds'])} trained seed(s) are averaged within scene before bootstrap confidence intervals.",
        "",
        "| Split | Method | Calls, mean | Latency, mean (ms) | Solvable goals found ≤4 calls | Exactness |",
        "|---|---|---:|---:|---:|---:|",
    ]

    for row in report["summary"]:
        success_rate = row["success_at_4"]["mean"]
        success_text = "n/a" if success_rate is None else f"{success_rate:.1%}"
        lines.append(
            f"| {row['split']} | {row['method']} | "
            f"{row['verifier_calls']['mean']:.2f} | "
            f"{row['latency_ms']['mean']:.2f} | {success_text} | "
            f"{row['correctness']['mean']:.1%} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A smaller verifier-call count does not establish a speedup. Neural encoding and scoring are included in request latency. No-solution certificates still require exhausting the candidate universe.",
            "",
            "The released benchmark is deliberately small. Out-of-distribution (OOD) cases change room count, graph density, source reliability, and mobility. This is a bundled distribution shift, not a controlled estimate of each factor separately.",
            "",
            "Full confidence intervals, paired differences against the heuristic, individual scene rows, manifest hashes, and checkpoint hashes are in the JSON and CSV files.",
            "",
            "## Belief calibration proxy",
            "",
            "| Split | Belief method | Brier score, mean |",
            "|---|---|---:|",
        ]
    )
    for row in report["belief_summary"]:
        lines.append(
            f"| {row['split']} | {row['mode']} | {row['brier']['mean']:.5f} |"
        )
    lines.extend(
        [
            "",
            "Belief scores use the same received histories at the initial search checkpoint. They do not compare divergent rollouts or measure human believability.",
            "",
        ]
    )
    Path(path).write_text("\n".join(lines), encoding="utf-8")
