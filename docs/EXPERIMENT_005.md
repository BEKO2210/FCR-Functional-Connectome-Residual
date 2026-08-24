# Experiment 005 — Nested Spatial CV

Experiment 005 is the preregistered development follow-up to the negative held-out result from Experiment 004.

The authoritative preregistration is GitHub Issue #11. This file records the implementation boundary in the repository.

## Fixed protocol

- public MICrONS layer-2/3 v185 static release;
- five contiguous soma-X outer slabs;
- each node appears in exactly one outer test slab;
- inner node-disjoint spatial CV uses only the four non-test slabs;
- model families: global, continuous distance, relative geometry, spatial wiring propensity + geometry;
- fixed L2 candidate grids from Issue #11;
- Degree-product oracle is diagnostic only and pays explicit degree-vector side information;
- primary criterion: at least 5% aggregate residual-bit reduction versus global **and** at least four of five outer-fold wins.

## Evidence boundary

This is `E1-development-nested-spatial-cv`, not independent confirmation. The dataset was already inspected in Experiment 004. A positive result must therefore be carried to a different dataset/materialization before any general claim.
