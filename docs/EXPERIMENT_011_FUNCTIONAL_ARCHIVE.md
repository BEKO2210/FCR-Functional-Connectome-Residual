# Experiment 011 — MICrONS v185 functional archive identity inventory

Experiment 011 is preregistered in issue #31. It is an E0 provenance/schema experiment only.

## Goal

Inventory the official MICrONS Layer 2/3 functional-vignette archive used by the Allen Institute `MicronsBinder` reference implementation and determine whether the functional subset exposes an identity-only source that can later be joined to the already frozen public v185 structural graph.

## Frozen provenance

- Allen reference repository: `AllenInstitute/MicronsBinder`
- immutable reference commit: `3f53e35e2bfd3063469dcbab0e1dedce5a82e3ca`
- archive: `https://zenodo.org/record/6363348/files/211019_vignette_functional_analysis_data.tgz`
- published MD5: `bcb0d4f678909fbd481ac0b01242ae5c`

The official vignette documents `Neuron["segment_id"]` for the structural pyramidal-cell set and uses `segment_id` plus `scan_id` for functional records.

## What this experiment does

The live workflow downloads the frozen archive, verifies the published MD5, records a SHA-256, inventories outer and nested tar members, and hashes identity-relevant files as opaque bytes.

It does not analyze calcium traces, tuning curves, visual responses, stimulus-response statistics, connectivity outcomes, or FCR outcomes.

## Interpretation

The report ends in exactly one next-step classification:

- `identity-source-available`: an identity-only functional key source is separately present;
- `identity-source-entangled-with-functional-values`: the documented functional keys remain packaged together with functional data objects;
- `hard-stop`: the expected official archive layout is not present.

All three are valid scientific inventory outcomes. A successful GitHub workflow means the archive was handled reproducibly and within the information boundary; it does not force a favorable classification.

## Merge rule

Do not merge the Experiment 011 pull request until both the normal repository CI and the dedicated Experiment 011 live workflow pass on the same PR head. After merge, the push-to-main live workflow must pass again before closing issue #31.
