# FCR Theory v0.1

## 1. Problem statement

Electron microscopy measures a physical representation. FCR instead asks for a compact description that preserves a predefined functional observable.

Let:

- `G` be an observed connectome graph plus permitted structural attributes;
- `M` be a predictive model learned without access to the held-out test region;
- `C` be a code or compressed description;
- `decode(C)` be the reconstructed graph/system;
- `F(.)` be a functional measurement operator;
- `D(., .)` be a distortion metric.

The target quantity is not raw image entropy. It is the minimum description length compatible with an allowed functional distortion:

```text
R_F(epsilon) = min_C L(C)
               such that D(F(G), F(decode(C))) <= epsilon.
```

## 2. Structural residual

For binary candidate-pair connectivity `y_i` with model probabilities `p_i`:

```text
L_residual = -sum_i [y_i log2(p_i) + (1-y_i) log2(1-p_i)]
```

This is ideal predictive code length on held-out observations.

A prototype two-part score is:

```text
L_two_part = L_model + L_residual
```

In v0.1, `L_model` is deliberately conservative and simple. Publication-grade MDL work must replace this with a principled coding scheme or prequential code.

## 3. Working metrics

FCR reports both:

```text
bits_per_candidate_pair = L_two_part / N_pairs
bits_per_positive_edge   = L_two_part / N_positive
```

The second value is intuitive but depends strongly on candidate-pair definition and sparsity, so it must never be reported alone.

## 4. Functional Connectome Residual

The phrase **Functional Connectome Residual (FCR)** is a working project term for structural information not already predicted by the chosen shared model and that is relevant to the selected functional measurements.

It is not assumed that every surprising edge is functionally important. That relationship is an empirical question.

## 5. Brain Residual Entropy

`Brain Residual Entropy` is retained only as an informal motivating phrase. The repository will prefer operational quantities such as held-out code length and functional rate-distortion because a single intrinsic entropy of a biological brain is not directly identified by these experiments.

## 6. Central falsifiable hypotheses

**H1 — Predictability:** type+geometry models reduce held-out structural code length relative to density-only models on spatially untouched tissue.

**H2 — Residual concentration:** a minority of structural observations account for a disproportionate fraction of predictive surprisal.

**H3 — Functional concentration:** preserving high-residual structural information yields more functional fidelity per encoded bit than preserving equally sized random subsets.

H3 is the strongest and most important claim. It requires real matched functional measurements; synthetic recurrent-network experiments are only implementation checks.

## 7. Claims boundary

FCR v0.1 does not claim:

- that a connectome is sufficient to reproduce a mind;
- that synapses are the only state variables required for neural function;
- that an EM reconstruction can be losslessly replaced by this code;
- that personality, memories, or consciousness can be recovered;
- that the project concept is historically novel.

Novelty must be established by a dedicated literature and patent review before publication language is used.
