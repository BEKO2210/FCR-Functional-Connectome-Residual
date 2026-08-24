# MICrONS Pilot Runbook

This runbook is for **data plumbing only**. The root-ID-limited pilot is not a preregistered biological sample and must not be reported as evidence for H1–H3.

## Why the adapter is deliberately strict

The MICrONS proofreading documentation distinguishes axons that are clean from axons that are close to complete. FCR defaults to `axon_fully_extended`, requires dendrite proofreading status, and rejects rows whose `valid_id` no longer equals the current `pt_root_id`.

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

## First safe plumbing run

The current public release used for the initial adapter validation target is materialization **1822**. Versions are never selected implicitly.

Start small:

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

The NPZ contains:

```text
source
target
source_type
target_type
distance_nm
connected
synapse_count
```

The provenance sidecar records version, selection settings, counts, coordinate units, and caveats. It contains no token.

## Scale-up plumbing run

Only after the 100-node export passes sanity checks:

```bash
fcr microns-export \
  --version 1822 \
  --max-nodes 500 \
  --output data/cache/microns_v1822_pilot_500.npz
```

500 nodes produce at most 249,500 directed non-self candidate pairs. FCR has a hard candidate-pair limit to prevent accidental quadratic memory explosions.

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
