# Token-free MICrONS layer-2/3 structural pilot

## Purpose

This path exists so FCR can touch real connectomic data without requiring a CAVE account,
Google login, personal token, or secret. It uses the public static MICrONS layer-2/3 v185
release hosted on Zenodo.

This is deliberately **not** a substitute for the current `minnie65_public` / v1822 pilot.
It is an earlier, smaller dataset that is useful for validating FCR's real-data structural
coding pipeline before privileged/dynamic CAVE access is available.

## Frozen public inputs

Zenodo record: `10.5281/zenodo.7510511`

The workflow downloads only:

| File | Published size | Published MD5 |
|---|---:|---|
| `soma_valence_v185.csv` | ~34 kB | `b7c349b1ecbdad46185bb98d88b96d20` |
| `soma_subgraph_synapses_spines_v185.csv` | ~274 kB | `5bbb8ff59dcad4ccea6d46930920cd04` |

The hashes are checked before analysis. A mismatch is a hard failure.

## Graph construction

1. Read soma root IDs, coarse cell class, and soma XYZ coordinates in nanometers.
2. Identify the historical pre/post root-ID columns in the proofread subgraph table.
3. Keep unique soma roots that appear as endpoints in that proofread table.
4. Aggregate repeated synapse rows into an observed synapse count per directed pair.
5. Enumerate every ordered non-self pair among retained nodes exactly once.
6. Set `connected = 1` iff at least one proofread subgraph synapse is observed for that pair.
7. Compute Euclidean soma distance from the released coordinates.

## Leakage boundary

Nodes are sorted spatially along coordinate axis 0 and split into contiguous slabs:

- train: 70%
- validation: 15%
- test: 15%

A node belongs to exactly one split. Primary within-split pair sets therefore share no
neurons across train/validation/test.

## Models

The existing frozen structural baselines are evaluated without post-result tuning:

- global Bernoulli;
- distance-binned;
- type-pair;
- type + distance.

Both residual code length and two-part code length (`model bits + residual bits`) are
reported.

## Evidence label

`E1-structural-public-static-pilot`

A successful run can establish that the FCR measurement machinery works on a real,
proofread connectomic subgraph and can quantify held-out structural code lengths.

It **cannot** establish:

- performance on current MICrONS v1822;
- functional equivalence after compression;
- a minimum sufficient neural description;
- whole-brain storage requirements;
- preservation of cognition, consciousness, memory, or identity.

## GitHub execution

Workflow: `MICrONS Public L2/3 FCR Pilot`

It runs automatically when its analysis code first lands on `main` and can also be
re-run manually with `workflow_dispatch`. No repository secret is required.
