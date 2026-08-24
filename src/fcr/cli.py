"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .experiment import evaluate_models, save_results
from .synthetic import generate_synthetic_connectome, train_test_split


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
    args = parser.parse_args()

    if args.command == "synthetic-baselines":
        sample = generate_synthetic_connectome(
            n_nodes=args.nodes, n_pairs=args.pairs, seed=args.seed
        )
        train, test = train_test_split(sample, seed=args.seed + 1)
        results = evaluate_models(train, test)
        save_results(results, args.output)
        print(json.dumps([asdict(r) for r in results], indent=2))
