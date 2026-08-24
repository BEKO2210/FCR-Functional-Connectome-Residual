# FCR — Functional Connectome Residual

**Research question:** How many bits are actually required to preserve the *relevant function* of a biological neural network, rather than its exact electron-microscopy image?

FCR is an experimental research project for measuring the gap between **physical connectome complexity** and **functionally necessary information**.

> Status: **v0.1 research scaffold.** No novelty, biological-equivalence, mind-uploading, or whole-brain-compression claim is made by this repository.

## Core hypothesis

A connectome may contain large amounts of structure that are predictable from shared biological regularities such as geometry and cell type. FCR asks whether the information that remains after those regularities are modeled — the **functional connectome residual** — is much smaller than the raw structural representation, and which parts of that residual matter for function.

For a held-out graph sample `G` under model `M`, the prototype measures predictive residual code length as:

```text
L_residual(G | M) = -log2 P(G | M)
L_two_part        = L_model + L_residual
```

The long-term target is a functional rate-distortion objective:

```text
R(epsilon) = min_C |C|
             subject to D(F(G), F(decode(C))) <= epsilon
```

where `F` is a functional measurement and `D` is functional distortion.

## What v0.1 contains

- transparent structural coding baselines: global density, distance, cell-type pair, and type+distance;
- held-out information metrics in bits;
- explicit model-overhead accounting for prototype two-part codes;
- deterministic synthetic null data so CI never depends on petabyte-scale datasets;
- a surrogate recurrent-network rate-distortion experiment;
- a staged protocol for MICrONS and later H01;
- tests and GitHub Actions CI.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
ruff check .
fcr synthetic-baselines
```

The JSON output is written to `results/synthetic_baselines.json` by default.

## Experiments

```bash
python experiments/001_baselines.py
python experiments/002_residual_entropy.py
python experiments/003_rate_distortion.py
```

Experiments 001–003 use synthetic data only. They validate the machinery; they **cannot** establish a biological result.

## MICrONS live adapter

FCR includes a version-pinned MICrONS export path. Live access is optional and never handles token values itself. Install the optional dependency and configure CAVE authentication locally using the official MICrONS instructions.

```bash
python -m pip install -e '.[dev,microns]'
fcr microns-export --version 1822 --max-nodes 100 --output data/cache/microns_v1822_pilot_100.npz
```

The default selection accepts only `axon_fully_extended` neurons, requires dendrite proofreading status, rejects stale `valid_id` mappings, requests positions in nanometers, aggregates multiple synapses per directed pair, and writes a compressed NPZ plus provenance JSON. The deterministic root-ID limit is **plumbing only**, not a scientific sampling design.

See [`docs/MICRONS_RUNBOOK.md`](docs/MICRONS_RUNBOOK.md).

## Real-data plan

The first target is **MICrONS**, because it combines dense synaptic structure with matched functional measurements. The public resource reports roughly 200,000 cells, 75,000 neurons with physiology, and 523 million synapses. FCR will query bounded subsets and pin a materialization version rather than download the full dataset.

The later human structural replication target is **H01**, a roughly 1 mm³ human cortex reconstruction containing about 57,000 cells and 150 million synapses in a 1.4 PB representation.

See [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md) and [`docs/DATASETS.md`](docs/DATASETS.md).

## Scientific guardrails

A positive result must survive all of these:

1. spatially disjoint train/validation/test regions;
2. strong geometry and cell-type baselines;
3. model-size accounting;
4. fixed data-version and inclusion rules;
5. boundary/censoring controls;
6. negative-sampling sensitivity analysis;
7. null and permutation models;
8. functional validation on measurements not used to fit the structural codec;
9. replication on a second region/dataset before any broad claim.

The hypothesis is considered unsupported if FCR does not beat strong predictive/MDL baselines on untouched spatial regions or if compression gains disappear after correct model-overhead and observation-bias accounting.

## Repository layout

```text
src/fcr/                    reusable metrics, models and experiment code
experiments/                numbered executable experiments
tests/                      deterministic unit tests
docs/THEORY.md              definitions and claims boundary
docs/EXPERIMENT_PROTOCOL.md preregistration-style protocol
docs/DATASETS.md            real-data access and versioning plan
docs/THREATS_TO_VALIDITY.md failure modes and controls
results/                    generated local outputs (not committed)
```

## Reproducibility

The project treats every numerical claim as provisional until it can be reproduced from a pinned dataset version and committed experiment configuration. Generated result files are ignored by default so accidental outputs are not confused with frozen evidence.

## References / data resources

- MICrONS Explorer: https://www.microns-explorer.org/cortical-mm3
- MICrONS tutorials: https://tutorial.microns-explorer.org/
- Google Research Neural Mapping datasets (H01): https://sites.research.google/gr/neural-mapping/datasets/

## License

No license has been selected yet. Until one is added, normal copyright rules apply. This is intentional for the bootstrap commit so the project owner can choose the research/code license explicitly.
