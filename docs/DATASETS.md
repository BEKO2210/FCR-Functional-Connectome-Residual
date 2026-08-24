# Dataset Plan

## MICrONS — first target

Why first: it combines dense synaptic reconstruction with functional measurements and matched structural/functional subsets.

Public entry points:

- https://www.microns-explorer.org/cortical-mm3
- https://tutorial.microns-explorer.org/

The public documentation reports roughly 200,000 cells, about 75,000 neurons with physiology, and 523 million synapses.

### Access policy for FCR

Do **not** download the full imagery/segmentation for the first experiments. Query bounded structural tables and derived properties. Pin a materialization version or timestamp because annotations and proofreading continue to change.

The implemented MICrONS adapter requires an explicit version in configuration; `latest` is not accepted for frozen experiments.

### Minimal first real-data slice

Target a bounded region with approximately 2,000–5,000 eligible neurons after quality filters. Start with aggregated binary connectivity and synapse count, not raw EM voxels.

The first export should contain only fields needed for the registered models, for example:

```text
source_id
target_id
source_type
target_type
source_xyz
target_xyz
synapse_count
proofreading/status fields
spatial_block
```

Functional properties are joined only after the structural analysis configuration is frozen.

### Implemented plumbing adapter

The `fcr microns-export` command performs a deliberately bounded query through CAVEclient. It requires an explicit materialization version, defaults to `axon_fully_extended`, uses `desired_resolution=[1,1,1]` so candidate-pair distances are stored in nanometers, and never accepts a token argument.

The initial deterministic root-ID-limited selection is an E0 plumbing test only. See `docs/MICRONS_RUNBOOK.md` before running it.

## H01 — later human structural target

Public entry point:

- https://sites.research.google/gr/neural-mapping/datasets/

Google reports a ~1 mm³ human cortex sample represented by ~1.4 PB of EM-derived data, with ~57,000 cells and ~150 million synapses.

H01 is valuable for testing structural residual coding in human tissue but is not a substitute for matched functional validation.

## Data never committed here

Raw or derived large datasets belong in ignored local/cloud storage, not Git history. `data/raw/` and `data/cache/` are ignored.

## Provenance requirement

Every real-data result must record:

- dataset/release name;
- materialization version or timestamp;
- query code commit SHA;
- spatial bounding box;
- filters;
- row counts before and after filters;
- content hashes for exported analysis tables when practical.
