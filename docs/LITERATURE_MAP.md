# Literature / Novelty Map

This file exists to prevent accidental novelty inflation. FCR is a research direction, not a claim that nobody has previously combined information theory and connectomics.

## Clearly related prior directions

### Minimum description length and graph compression

Compression-based motif inference already uses the MDL principle to compare graph explanations by total description length. This means "compress a network to reveal regularity" is established methodology, not an FCR invention.

Reference:

- *Compression-based inference of network motif sets*, PLOS Computational Biology: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012460

### Generative wiring rules in connectomics

Generative and simulation-based models already predict synaptic connectivity from morphology, geometry and cell-type-related structure.

Reference:

- *Simulation-based inference for efficient identification of generative models in computational connectomics*, PLOS Computational Biology: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011406

### Connectome structure + MDL + dynamics

A 2026 Drosophila optic-lobe study reports that cell-type symmetries reduce structural description length by roughly 9–16% relative to independent-edge baselines, then separately studies connectome-constrained dynamics and information/energy trade-offs.

Reference:

- *Energy-efficient information processing and eligibility-trace plasticity in the Drosophila optic lobe connectome*, Scientific Reports: https://www.nature.com/articles/s41598-026-52140-3

This is especially important prior art for FCR and must be treated as a strong baseline/conceptual neighbor.

### Rate-distortion and brain-network communication

Rate-distortion ideas have already been used at the macroscale to study lossy communication and compression efficiency in human structural connectomes.

Reference:

- *Efficient coding in the economics of human brain connectomics*: https://pubmed.ncbi.nlm.nih.gov/36605887/

## Working gap to test — not yet a novelty claim

The specific direction FCR intends to test is narrower:

1. synapse-resolved structural prediction on spatially untouched tissue;
2. explicit model + residual bit accounting;
3. a residual-aware lossy reconstruction policy;
4. distortion scored using independently measured, matched neural function;
5. strong comparison against geometry, type, degree, block-model, MDL and random-retention baselines;
6. replication across regions/datasets.

MICrONS is unusually suited to this because dense structural connectivity can be linked to functional recordings/model-derived properties for matched neurons.

The project must perform a broader systematic literature and patent search before describing this exact combination as novel.

## Search questions before any paper/preprint

- Has "functional rate-distortion" been defined at synapse resolution on an EM connectome?
- Has predictive surprisal of individual synapses been related to matched physiological importance?
- Has an MDL residual been used as the transmission priority in lossy connectome reconstruction?
- Has this been evaluated on spatial holdout rather than random edge holdout?
- Have model-description cost, boundary censoring and candidate-negative construction been handled simultaneously?

Every discovered near-neighbor goes into this file, even if it weakens the novelty story.
