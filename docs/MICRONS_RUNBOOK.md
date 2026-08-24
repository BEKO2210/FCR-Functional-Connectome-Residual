# MICrONS Pilot Runbook

This runbook is for **data plumbing only**. The root-ID-limited pilot is not a preregistered biological sample and must not be reported as evidence for H1–H3.

## Why the adapter is deliberately strict

The MICrONS proofreading documentation distinguishes axons that are clean from axons that are close to complete. FCR defaults to `axon_fully_extended`, requires dendrite proofreading status, and rejects rows whose `valid_id` no longer equals the current `pt_root_id`.

Cell-type annotations are queried in bounded batches. Roots with more than one automated cell-type row are excluded rather than assigned an arbitrary label. `--max-nodes` is applied only after these eligibility checks.

A zero in the exported candidate graph means only:

> no synapse was returned by the pinned CAVE query for this selected pair.

It does **not** prove a biological non-connection.

## Supported live-access environment

FCR's core package supports Python 3.11–3.13. The optional live MICrONS adapter is pinned to `caveclient==8.2.1`; upstream currently documents official CAVEclient support through Python 3.12. Use Python 3.11 or 3.12 for live CAVE access.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,microns]'
```

## One-time CAVE authentication

Do this on your own machine. **Never paste a CAVE token into GitHub, FCR config, an issue, a chat, or a command-line argument.** The token functions as a credential.

Follow the official setup:

- https://tutorial.microns-explorer.org/quickstart_notebooks/01-caveclient-setup.html

The official CAVEclient flow stores the token in its local credential store. FCR intentionally has no `--token` option and never serializes authentication data.

## Mandatory preflight

Before any export, run the bounded read-only doctor check:

```bash
fcr microns-doctor --version 1822
```

The doctor verifies that:

- the requested materialization version exists;
- the proofreading, cell-type and configured synapse tables are present;
- the proofreading schema contains every field required by the strict filters;
- the cell-type schema exposes `pt_root_id` and `cell_type`;
- a one-row synapse probe can be executed;
- the materialization timestamp can be read.

The doctor uses at most five proofreading rows, one cell-type row and one synapse row. Its JSON result contains no authentication state or credential values. Do not continue if `"ok": true` is not returned.

## First safe plumbing run

The initial adapter validation target is materialization **1822**. Versions are never selected implicitly.

After the doctor passes, start small:

```bash
fcr microns-export \
  --version 1822 \
  --max-nodes 100 \
  --output data/cache/microns_v1822_pilot_100.npz
```

Expected outputs:

```text
data/cache/microns_v1822_pilot_100.npz
data/cache/microns_v1822_pilot_100.provenance.json
```

The export command validates its own output before returning. Run the validator explicitly as a second check:

```bash
fcr microns-validate data/cache/microns_v1822_pilot_100.npz
```

A successful validation returns JSON with `"valid": true` and the artifact SHA-256, materialization version, node count, candidate-pair count, connected-pair count and total synapse count.

The NPZ contains both node-level source data and the derived candidate graph:

```text
node_id
node_type
node_xyz_nm
node_strategy_axon
source
target
source_type
target_type
distance_nm
connected
synapse_count
```

The validator checks, among other invariants:

- the NPZ SHA-256 matches the provenance sidecar;
- node IDs are unique and node coordinates are finite;
- every directed non-self pair is present exactly once;
- pair IDs refer only to exported nodes;
- `connected == (synapse_count > 0)`;
- pair distances are finite, positive, and exactly reconstructable from exported node coordinates within numerical tolerance;
- source/target cell types match node metadata;
- observed proofreading strategies agree with provenance configuration;
- provenance counts agree with the NPZ.

The provenance sidecar records version, selection settings, counts, coordinate units, caveats and the NPZ SHA-256. It contains no token.

For exactly 100 valid nodes, the candidate graph must contain **9,900** directed non-self pairs.

## Scale-up plumbing run

Only after the doctor and the 100-node export both pass:

```bash
fcr microns-export \
  --version 1822 \
  --max-nodes 500 \
  --output data/cache/microns_v1822_pilot_500.npz

fcr microns-validate data/cache/microns_v1822_pilot_500.npz
```

500 nodes produce 249,500 directed non-self candidate pairs. FCR has a hard candidate-pair limit to prevent accidental quadratic memory explosions.

## Do not use this selection for the paper result

The pilot sorts eligible root IDs and takes the first `N` only to make the plumbing deterministic. The registered biological experiment must instead freeze a spatially meaningful region and a spatial holdout before evaluating the final test set.

## Before E1/E2

The next data milestone requires all of the following:

- freeze a spatial bounding box / region selection;
- freeze materialization version;
- define biological candidate-negative eligibility;
- compare strict vs less strict proofreading strategies as a sensitivity analysis;
- add a degree-aware baseline;
- add a block-model baseline;
- replace the prototype model-overhead count with a defensible MDL/prequential code;
- record hashes for exported analysis artifacts.

Until then, MICrONS exports are **E0 data plumbing**.
