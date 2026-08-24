# AGENTS.md — FCR Research Rules

These rules apply to every coding/research agent working in this repository.

## Mission

Test whether a connectome can be represented as a shared predictive model plus a smaller residual while preserving explicitly defined functional observables.

## Non-negotiable rules

1. **Never manufacture novelty.** Search and document prior art before using words such as novel, first, unprecedented or breakthrough.
2. **Never turn synthetic results into neuroscience claims.** Synthetic runs are E0 evidence only.
3. **Never use the final spatial test block for tuning.** Hyperparameters and inclusion rules freeze first.
4. **Never report residual bits without model cost.** Show both separately and together.
5. **Never treat all missing synapses as clean negatives without an explicit observation rule.** Boundary and proofreading state matter.
6. **Never use random edge split as the primary biological result.** Primary evaluation is spatially disjoint.
7. **Never use functional targets to rank bits and then claim independent functional validation.** That is circular unless explicitly labeled supervised.
8. **Never download petabyte-scale raw data by default.** Query bounded tables/derived data first.
9. **Pin dataset materialization/version/timestamp for frozen experiments.** Moving `latest` is not reproducible evidence.
10. **Every headline result gets an evidence level E0–E4.** See `docs/EXPERIMENT_PROTOCOL.md`.

## Required before merging code

```bash
ruff check .
pytest
```

New metrics require tests with analytically known cases. New models require a null/synthetic test. New real-data experiments require provenance metadata and a frozen configuration.

## Claims language

Preferred:

- "the held-out code length decreased by ..."
- "under this candidate-pair definition ..."
- "this supports/does not support H1 ..."
- "functional distortion under the selected observable ..."

Forbidden without extraordinary evidence:

- "we compressed a human mind"
- "the connectome fully determines the person"
- "this proves consciousness can be stored"
- "nobody has ever thought of this"

## Repository hygiene

- keep raw/large data out of Git;
- deterministic seeds for tests;
- no secrets or dataset credentials in commits;
- small reviewable commits;
- document scientific decisions, not just code changes.
