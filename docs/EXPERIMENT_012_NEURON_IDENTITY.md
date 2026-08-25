# Experiment 012 — MICrONS v185 Neuron identity bridge

Experiment 012 is an **identity-only E0 bridge** between the official MICrONS
functional-vignette pyramidal-cell list and FCR's already frozen 334-node public-v185
structural graph.

Preregistration: GitHub Issue #33.

## Authorized data access

The official Allen MICrONS reference describes `Neuron.pkl` as the list of pyramidal-cell
segment IDs and uses `Neuron["segment_id"]` as the identity vector. Experiment 012 may read
only that identity value after the complete archive/member hash chain has passed.

The following remain unopened for scientific values:

- `EASETrace.pkl`;
- `EASETuning.pkl`;
- `Stimulus.pkl`.

No calcium trace, tuning property, stimulus response, response correlation, connectivity
outcome, or FCR model outcome may influence membership.

## Matching rule

The primary rule is exact integer equality only:

`functional Neuron.segment_id == frozen structural node_id`

No coordinate matching, tolerance, nearest-neighbor lookup, remapping, fuzzy identity
repair, or outcome-dependent filtering is allowed.

The output is a sorted one-column `segment_id` CSV containing the exact intersection and a
SHA-256 hash of that manifest.

## Interpretation

Experiment 012 can establish a reproducible structure↔functional-cell identity cohort. It
cannot establish functional fidelity. Any later use of calcium/tuning values requires a new
preregistered experiment that freezes the functional metric, controls, statistical design,
and adequacy criteria before those values are opened.
