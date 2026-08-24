"""Experiment 001: predictive structural code-length baselines."""

import json
from dataclasses import asdict

from fcr.experiment import evaluate_models
from fcr.synthetic import generate_synthetic_connectome, train_test_split

sample = generate_synthetic_connectome()
train, test = train_test_split(sample)
results = evaluate_models(train, test)
print(json.dumps([asdict(result) for result in results], indent=2))
