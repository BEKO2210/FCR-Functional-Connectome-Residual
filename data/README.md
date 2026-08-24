# Data directory

Large datasets are intentionally not stored in Git.

- `data/raw/` — local read-only source exports (ignored)
- `data/cache/` — derived/query cache (ignored)

Every real-data export used for a result must have provenance recorded in the experiment configuration or result metadata.
