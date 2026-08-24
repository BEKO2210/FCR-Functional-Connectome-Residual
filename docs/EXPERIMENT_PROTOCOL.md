# Experiment Protocol

## Goal

Measure whether a shared structural model plus residual code can describe untouched connectome regions compactly, then test whether retained residual information predicts preservation of independently measured function.

## Stage 0 — machinery validation

Dataset: deterministic synthetic graphs.

Required outcomes:

- a fair-coin predictor costs exactly ~1 bit per binary observation;
- structured models beat the global-density baseline when the generator contains type/geometry structure;
- model overhead is included in the reported two-part score;
- full edge retention produces zero surrogate functional distortion.

No biological claim is allowed at Stage 0.

## Stage 1 — MICrONS structural pilot

### Freeze before looking at test results

Record:

- MICrONS materialization version or timestamp;
- queried bounding boxes;
- cell inclusion criteria;
- edge/synapse aggregation rule;
- candidate negative definition;
- minimum proofreading/quality state;
- random seeds;
- model hyperparameters;
- train/validation/test spatial split;
- primary metric and kill threshold.

### Primary structural metric

Held-out predictive bits per candidate pair, with model overhead reported separately and together.

### Required baselines

1. global Bernoulli density;
2. distance-only;
3. cell-type pair;
4. type + distance;
5. degree-aware baseline;
6. stochastic block / hierarchical block model where feasible;
7. a proper MDL or prequential baseline before publication claims.

### Split rule

Never use a random edge split as the primary result. Use spatially disjoint tissue blocks. A random split may appear only as a diagnostic because shared cells and local geometry can leak information.

### Negative observations

Absence of a synapse is not automatically a true biological negative. Candidate-pair construction must account for volume boundaries, incomplete processes, proof-reading state, and opportunity for contact. Report sensitivity to at least two defensible candidate definitions.

## Stage 2 — MICrONS functional test

Use function that was not optimized by the structural codec.

Candidate observables include:

- matched visual response vectors;
- orientation/direction tuning;
- response-correlation structure;
- digital-twin predicted response properties, clearly labeled as model-derived rather than direct physiology;
- population decoding metrics.

Primary question:

> At equal transmitted structural bits, does a residual-aware reconstruction preserve held-out functional observables better than random and strong structural baselines?

## Stage 3 — H01 structural replication

Repeat structural-only analysis on the human H01 volume. Because H01 does not provide the same matched living functional measurements, H01 cannot by itself validate functional equivalence.

## Kill criteria

Stop or rewrite the central hypothesis if any of the following holds:

- gains vanish on spatial holdout;
- gains vanish after accounting for model description cost;
- gains depend on one arbitrary negative-sampling definition;
- residual-aware retention does not beat random retention on independent functional metrics;
- effect does not replicate across at least two disjoint regions or datasets.

## Evidence levels

- **E0:** unit/synthetic only;
- **E1:** retrospective structural result;
- **E2:** frozen held-out structural result;
- **E3:** frozen held-out structure + independent function;
- **E4:** independent replication.

Repository documentation must attach one of these labels to every headline result.
