"""Experiment 002: estimate residual structural bits after learned regularities."""

from fcr.experiment import evaluate_models
from fcr.synthetic import generate_synthetic_connectome, train_test_split

sample = generate_synthetic_connectome(n_nodes=800, n_pairs=100_000, seed=101)
train, test = train_test_split(sample, seed=102)
results = evaluate_models(train, test)
best = min(results, key=lambda item: item.two_part_bits)
print(f"best_model={best.model}")
print(f"two_part_bits={best.two_part_bits:.2f}")
print(f"bits_per_pair={best.bits_per_pair:.6f}")
print(f"bits_per_positive={best.bits_per_positive:.6f}")
