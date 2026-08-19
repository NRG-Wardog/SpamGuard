# ML Engineering Controls

SpamGuard treats dataset preparation and evaluation as part of the model contract rather than as disposable notebook setup.

## Invariants

- Source provenance and allowed-use metadata are retained during ingestion.
- Identical text carrying conflicting labels is excluded from clean training/evaluation candidates.
- Exact duplicate text is deduplicated before split construction.
- Normalized template clusters are assigned as whole clusters to a single split.
- Train, validation, and test are checked for exact-text and template overlap.
- Split construction is deterministic for a fixed seed.
- Leakage verification fails the build step rather than emitting a warning and continuing.

## Evaluation Boundary

The repository separates three concerns:

1. **Data validity** — whether records are structurally usable and policy-compatible.
2. **Split validity** — whether evaluation partitions are isolated from repeated content/templates.
3. **Model quality** — how a trained classifier performs after the first two conditions hold.

This separation is deliberate: a high model score is not meaningful if the evaluation data contains repeated content from training or if source licensing/policy is unclear.

## CI Scope

Public CI intentionally validates the deterministic, dependency-light data pipeline. Model training and artifact-dependent inference are not run in every PR because they require large dependencies, model files, and potentially external data. They remain separate validation layers.
