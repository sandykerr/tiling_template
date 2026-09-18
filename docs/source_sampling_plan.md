# Source Selection and Sampling Plan

## Goal

Implement Step 5 of the tiling pipeline as a deterministic, independently
callable component that selects `SourceRecord` objects before grid and tile
planning. The component should support filtering, simple random sampling,
systematic sampling, and stratified sampling without depending on a pipeline
orchestrator, executor, progress display, or CLI.

The implementation should operate on explicit records and configurations,
avoid global state, and return enough information to explain why every input
source was selected, excluded, or left unselected.

## Proposed Package Layout

```text
configs/
└── sampling/
    ├── __init__.py
    ├── base.py
    ├── random.py
    ├── systematic.py
    └── stratified.py

sampling/
├── __init__.py
├── base.py
├── features.py
├── filtering.py
├── records.py
└── techniques/
    ├── __init__.py
    ├── simple_random.py
    ├── systematic.py
    └── stratified.py
```

Do not create a `features/` package initially. Promote `features.py` to a
package only if concrete feature implementations make the single module hard
to navigate.

## Design Principles

- Treat filtering, feature extraction, and sampling as separate operations.
- Use immutable, serializable configuration and result records.
- Establish canonical source ordering by stable `source_id` before sampling.
- Use technique-local random-number generators; never modify global random
  state.
- Make results independent of discovery, worker, or input completion order.
- Do not mutate input `SourceRecord` objects.
- Keep logging, progress reporting, multiprocessing, retries, and artifact
  publication outside the component.
- Allow custom technique and feature implementations without requiring changes
  to the built-in techniques.

## Sequential Implementation Steps

### 1. Define the Public Sampling Contract

Create an abstract `SourceSamplingTechnique` with a narrow method that accepts
an ordered sequence of eligible `SourceRecord` objects and returns selected
and unselected sources. Keep filtering and feature extraction outside this
base method unless a technique explicitly requires features.

Define whether technique instances compose immutable configuration or receive
configuration as a method argument. Prefer one consistent pattern across all
techniques.

### 2. Define Sampling Records

Add immutable sampling-specific records in `sampling/records.py`, including:

- `SourceExclusion`: source ID, stable reason code, and optional message.
- `SourceSamplingResult`:
    - Selected sources.
    - Eligible but unselected sources.
    - Excluded sources and their reasons.
    - Technique identifier.
    - Resolved seed, when applicable.
    - Optional per-stratum counts or other technique summary values.

Preserve source records rather than returning only IDs, but use IDs in issue
records to avoid duplicating large structures unnecessarily.

### 3. Define Common Size Configuration

Create a shared sampling-size configuration supporting:

- Exact count.
- Fraction of eligible sources.
- All eligible sources when no reduction is requested.

Require at most one size mode, reject negative counts and invalid fractions,
and define fraction rounding explicitly. Resolve requested size only after
filtering so feasibility is evaluated against eligible sources.

### 4. Implement Canonical Ordering and Seed Handling

Add pure helpers that:

- Reject duplicate `source_id` values.
- Sort candidates by stable `source_id`.
- Resolve and validate optional seeds.
- Create a local random-number generator.

Test that permuting the input sequence does not change the result for the same
configuration and seed.

### 5. Implement Eligibility Filtering

Define a `SourceFilter` interface that evaluates one source and returns either
an accepted result or a structured exclusion. Support composition of multiple
filters with deterministic evaluation order.

Initial built-in filters should remain lightweight and rely only on existing
`SourceRecord` and `AssetRef` fields, such as:

- Source ID inclusion or exclusion.
- Role/modality presence.
- Association-key membership.
- Optional user-provided filter implementations.

Metadata-, geography-, time-, and value-aware filters can be added when their
input contracts exist. Missing required feature data must follow an explicit
error or exclusion policy rather than being silently ignored.

### 6. Implement Simple Random Sampling

Create `SimpleRandomSamplingConfig` and `SimpleRandomSampler`.

The sampler should:

- Resolve the requested count after filtering.
- Validate feasibility before sampling.
- Sample without replacement.
- Use only its local seeded generator.
- Return selected and eligible-but-unselected sources in canonical order,
  regardless of the generator's internal draw order.

### 7. Implement Systematic Sampling

Create `SystematicSamplingConfig` and `SystematicSampler`. Use the term
"systematic" rather than "uniform" to distinguish every-kth selection from
equal-probability random sampling.

Define and test:

- Interval or requested-size configuration.
- Starting offset behavior.
- Whether the offset is fixed or seed-derived.
- Behavior when the requested size exceeds the eligible population.
- Deterministic handling when the population size is not evenly divisible by
  the interval.

### 8. Define Source Feature Extraction

Add a `SourceFeature` interface that returns a hashable value for a source.
Feature extraction may receive a read-only sampling context containing
optional metadata or precomputed summaries, but should not open assets or read
pixel values itself.

Initial useful features may include:

- Association-key components.
- Role or modality presence.
- Stable fields already stored on source or asset records.
- Composite features formed from multiple feature values.

Define an explicit missing-feature policy: error, exclude the source, or place
it in a named missing-value stratum.

### 9. Define Stratified Sampling Configuration

Create `StratifiedSamplingConfig` with an explicit feature identifier and
allocation policy. Initially support only allocation policies with clear,
testable semantics:

- Proportional allocation.
- Equal allocation across strata.
- Explicit per-stratum counts or fractions.

Specify rounding and remainder allocation deterministically. Validate that
required strata exist and that requested allocations are feasible before any
random draws occur.

### 10. Implement Stratified Sampling

Create `StratifiedSampler` that:

1. Extracts one stratum key per eligible source.
2. Groups sources deterministically by key.
3. Resolves and validates all stratum allocations.
4. Samples each stratum using a stable seed derivation that does not depend on
   dictionary or worker order.
5. Merges selected and unselected sources into canonical order.
6. Returns per-stratum candidate and selection counts in the result summary.

Do not partially return a stratified sample when allocation validation fails.

### 11. Compose Filtering and Techniques

Add a small `SourceSampler` component that performs the complete standalone
stage:

```text
SourceRecords
    -> validate identities and canonicalize order
    -> apply eligibility filters
    -> extract features when required
    -> apply sampling technique
    -> return SourceSamplingResult
```

This component is not the pipeline orchestrator. It should remain a normal
serial call that a future orchestrator can invoke inside sequential or worker
execution.

### 12. Add Extension and Validation Hooks

Document how projects can supply custom filters, features, techniques, and
post-selection validators. Application-specific spatial or temporal separation
rules may validate or implement their own selection behavior through these
interfaces.

Do not add generic clustering. Projects that need clustering should implement
their own feature construction and `SourceSamplingTechnique` according to
their domain semantics.

### 13. Complete Unit and Integration Coverage

Test at least:

- Empty populations.
- Duplicate source IDs.
- Count, fraction, and all-source size modes.
- Infeasible requests.
- Determinism across repeated runs and permuted input order.
- Different seeds producing valid alternative samples.
- Exclusion reasons and eligible-but-unselected reporting.
- Systematic interval and offset edge cases.
- Missing feature policies.
- Stratified proportional, equal, and explicit allocation.
- Small and empty strata.
- Stable remainder distribution.
- Configuration and result serialization or process pickling.
- User-defined technique and feature implementations.

Add an integration test using realistic `SourceRecord` objects produced by
the association component. Do not require multiprocessing to test sampling
correctness.

## Deferred Capabilities

- Pixel- or label-value-aware sampling until characterization summaries exist.
- Built-in geographic and temporal separation algorithms until their spatial
  and temporal record contracts are defined.
- Tile-level sampling, which belongs to Step 9 after `TilePlan` exists.
- Pipeline executors, progress reporting, retries, and manifest publication.
- Generic clustering or automatic cluster construction.

## Completion Criteria

Step 5 can be marked **[Component Complete]** when:

- Filtering, simple random, systematic, and stratified sampling are
  implemented through stable public contracts.
- All built-in techniques are deterministic with respect to canonical source
  identity and configured seed.
- Requested sizes and strata are validated before selection.
- Results distinguish selected, eligible-but-unselected, and excluded sources.
- Custom features, filters, and sampling techniques can be supplied without
  modifying built-in implementations.
- Configurations and results are suitable for later multiprocessing and
  orchestration.
- The complete repository test suite passes.
