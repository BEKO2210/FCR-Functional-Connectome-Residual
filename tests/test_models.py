from fcr.experiment import evaluate_models
from fcr.synthetic import generate_synthetic_connectome, train_test_split


def test_structured_models_beat_global_on_structured_synthetic_data():
    sample = generate_synthetic_connectome(n_nodes=350, n_pairs=60_000, seed=31)
    train, test = train_test_split(sample, seed=32)
    results = {r.model: r for r in evaluate_models(train, test)}
    assert results["distance"].residual_bits < results["global"].residual_bits
    assert results["type+distance"].residual_bits < results["global"].residual_bits


def test_two_part_code_includes_model_overhead():
    sample = generate_synthetic_connectome(n_nodes=80, n_pairs=3_000, seed=41)
    train, test = train_test_split(sample, seed=42)
    for result in evaluate_models(train, test):
        assert result.two_part_bits == result.residual_bits + result.model_bits
