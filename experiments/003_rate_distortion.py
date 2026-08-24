"""Experiment 003: surrogate functional distortion under structural edge retention."""

from fcr.models import TypeDistanceModel
from fcr.rate_distortion import structural_rate_distortion_curve
from fcr.synthetic import generate_synthetic_connectome, train_test_split

sample = generate_synthetic_connectome(n_nodes=120, n_pairs=10_000, seed=201)
train, test = train_test_split(sample, seed=202)
model = TypeDistanceModel().fit(train)
p = model.predict_proba(test)

for mode in ("residual-first", "typical-first", "random"):
    curve = structural_rate_distortion_curve(
        n_nodes=120,
        source=test.source,
        target=test.target,
        connected=test.connected,
        probabilities=p,
        mode=mode,
        seed=203,
    )
    for point in curve:
        print(mode, point.retained_fraction, point.retained_edges, f"{point.distortion:.6f}")
