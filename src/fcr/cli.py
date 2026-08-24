"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .experiment import evaluate_models, save_results
from .synthetic import generate_synthetic_connectome, train_test_split


def _add_microns_parser(subparsers: argparse._SubParsersAction) -> None:
    microns = subparsers.add_parser(
        "microns-export",
        help="query a small version-pinned MICrONS pilot and export candidate pairs",
    )
    microns.add_argument("--version", type=int, required=True)
    microns.add_argument("--max-nodes", type=int, default=500)
    microns.add_argument("--max-candidate-pairs", type=int, default=5_000_000)
    microns.add_argument(
        "--strategy",
        action="append",
        dest="strategies",
        help="axon proofreading strategy; repeat to allow multiple (default: axon_fully_extended)",
    )
    microns.add_argument("--output", default="data/cache/microns_pilot.npz")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="fcr", description="Functional Connectome Residual research toolkit"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    synthetic = sub.add_parser(
        "synthetic-baselines",
        help="run structural coding baselines on a synthetic graph",
    )
    synthetic.add_argument("--nodes", type=int, default=500)
    synthetic.add_argument("--pairs", type=int, default=40_000)
    synthetic.add_argument("--seed", type=int, default=7)
    synthetic.add_argument("--output", default="results/synthetic_baselines.json")
    _add_microns_parser(sub)
    args = parser.parse_args()

    if args.command == "synthetic-baselines":
        sample = generate_synthetic_connectome(
            n_nodes=args.nodes, n_pairs=args.pairs, seed=args.seed
        )
        train, test = train_test_split(sample, seed=args.seed + 1)
        results = evaluate_models(train, test)
        save_results(results, args.output)
        print(json.dumps([asdict(r) for r in results], indent=2))
        return

    if args.command == "microns-export":
        from .data.microns import MICrONSConfig, export_microns_pilot

        strategies = tuple(args.strategies) if args.strategies else ("axon_fully_extended",)
        config = MICrONSConfig(
            materialization_version=args.version,
            max_nodes=args.max_nodes,
            max_candidate_pairs=args.max_candidate_pairs,
            proofread_strategies=strategies,
        )
        npz_path, provenance_path = export_microns_pilot(config, args.output)
        print(f"wrote {npz_path}")
        print(f"wrote {provenance_path}")
