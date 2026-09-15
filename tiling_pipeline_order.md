# Tiling Pipeline Order

## Coordinate and Window Conventions

- Pixel-center versus pixel-edge conventions:
    - Define whether the configured grid origin represents the upper-left pixel edge or the center of the first pixel.
    - Prefer an edge-based origin internally because raster bounds and windows naturally describe pixel footprints.
    - For a pixel at row r and column c:
        - Upper-left edge: transform * (c, r)
        - Pixel center: transform * (c + 0.5, r + 0.5)
    - Apply the convention consistently when rasterizing vectors, converting coordinates to indices, comparing rasters, and calculating output transforms.
- Half-open window bounds:
    - Represent pixel-index windows as [start, stop), including the start and excluding the stop.
    - Example: rows [0, 256) contain rows 0 through 255 and have a height of 256.
    - This prevents gaps and duplicate pixels between adjacent windows.
- Read window versus output/core window:
    - The core window defines the official tile area and output transform.
    - The read window may be larger to provide context/halo pixels for reprojection, interpolation, filtering, or downstream modeling.
    - Clip or pad read windows that extend beyond source bounds.
    - Record both windows when they differ.

## Grid Approaches

- Image-relative grid:
    - Anchored to each source image's origin and dimensions.
    - Requires source metadata before source-specific grid creation.
- Canonical spatial grid:
    - Defined independently using a CRS, grid origin, resolution, bounds, tile dimensions, and stride.
    - The GridDefinition can be created before inspecting imagery.
    - Source metadata is still required to determine intersection, coverage, alignment, reprojection, and source read windows.
- Known standardized input grid:
    - Constructed from a known product or external grid specification.
    - Actual source alignment should still be validated because matching CRS and resolution do not guarantee matching origins.
- Vector- or AOI-driven grid:
    - Created from external regions, polygons, shapes, or another spatial index.
    - Source metadata is required when associating source coverage and read windows with cells.
- Adaptive or content-driven grid:
    - Created or filtered using source values, labels, or other content statistics.
    - Requires a preliminary content-inspection or statistics pass.

## Orchestration

1) Asset discovery: find assets on disk, in cloud storage, or through another catalog.
    - Determine filename and path parsing needs, such as product ID, date, region, product, or modality.
    - Identify candidate inputs, labels, metadata, and sidecar files.
    - Support multiple modalities.
    - Account for logical assets split across multiple physical files, such as an ENVI header and binary data pair.
    - Discovery answers which assets exist; it does not need to prove that a complete logical source exists.
2) Asset association: group discovered assets into logical SourceRecords.
    - Assign each asset a role, such as input, label, QA, metadata, or sidecar.
    - Verify that required matches exist.
    - Detect missing, duplicate, and ambiguous associations.
    - Create a stable source ID that is not based only on a potentially duplicated basename.
3) Source metadata inspection and structural validation.
    - Open and parse file headers without reading all pixel data.
    - Validate CRS/projection, transform, resolution, dimensions, extent, data type, nodata declaration, and available indices.
    - Confirm that expected channels, bands, date/time slices, or other dimensions exist.
    - Determine whether modalities can be aligned to the intended grid.
    - Perform source-level inspection in independent workers when useful, with each worker opening its own file handles.
    - Distinguish header/structural validation from complete content validation; a valid header does not prove that every data block can be decoded.
4) Optional source characterization.
    - Read representative blocks or existing sidecar summaries when selection depends on input or label values.
    - Detect sampled decode failures, implausible values, non-finite values, and approximate nodata or class distributions.
    - Perform a complete source scan only when explicitly requested or required.
    - Store reusable summaries in the SourceRecord or a separate characterization artifact.
5) Source selection and sampling.
    - Possible variables:
        - Geography
        - Date/time
        - Label value or distribution
        - Input value or distribution
        - Modality, sensor, product, or another configured field
    - Possible techniques:
        - Stratified sampling
        - Simple random sampling
        - Systematic/uniform sampling, such as every kth source
        - Cluster or group sampling
        - Application-specific sampling
    - Validate that requested counts and percentages are feasible across bins, strata, and groups.
    - Ensure required strata have eligible candidates.
    - Make results deterministic using a configured seed or stable hash, independent of worker completion order.
    - Respect configured exclusions and spatial/temporal separation requirements.
    - Source selection based on data values requires metadata summaries or the optional characterization phase.
6) Define or load the GridDefinition.
    - Select the grid approach: image-relative, canonical, standardized, AOI-driven, or adaptive.
    - Define cell width and height.
    - Define grid origin using an explicit pixel-edge or pixel-center convention.
    - Define pixel or geographic/map units.
    - Define output resolution, axis/dimension order, and canonical CRS/transform.
    - Define stride and overlap.
    - Define edge behavior:
        - Drop partial cells.
        - Emit smaller cells.
        - Resize cells.
        - Pad cells using a configured mode and value.
    - Define the output/core window and any additional read context or halo.
    - Define expected output shape.
    - Define co-registration requirements between modalities.
    - Keep the GridDefinition independent from train/validation/test membership.
7) Create candidate TilePlans.
    - Intersect selected sources with grid cells.
    - Calculate source read windows, core/output windows, spatial bounds, and output transforms.
    - Determine expected padding, coverage, reprojection, and alignment behavior.
    - Associate each TilePlan with its SourceRecord and leakage group.
    - Derive a stable tile ID from a stable source ID, grid/configuration ID, and tile row/column or equivalent spatial identity.
    - Do not include the dataset split in the stable tile ID.
8) Split assignment for ML datasets.
    - Permit an explicit no-split state for non-ML applications, preliminary runs, or downstream assignment.
    - Prefer assignment by the highest-level leakage group, such as source, scene, region, patient, site, or time period.
    - Keep related and overlapping tiles in the same split unless leakage is intentionally allowed.
    - Permit split assignment before candidate grid creation when it is entirely source/group based.
    - Require candidate TilePlans first when split assignment deliberately depends on tile-level properties.
    - Store split assignments independently from the reusable GridDefinition and stable tile identity.
9) Tile selection and sampling.
    - Filter or sample candidate TilePlans within the previously assigned leakage and split constraints.
    - Support geographic, temporal, label-aware, value-aware, systematic, random, stratified, clustered, or custom approaches.
    - Collect inexpensive tile summaries first when content-aware selection requires them.
    - Validate requested counts, percentages, strata, exclusions, and minimum separation rules.
    - Ensure deterministic results for the configured seed.
10) Optional fitted-preprocessing statistics phase.
    - Treat fitted preprocessing as distinct from model-training diagnostics.
    - Support publishing raw tiles without fitted preprocessing as the default generic behavior.
    - If dataset-level normalization is requested, calculate statistics from selected training data only.
    - Use workers to calculate mergeable partial statistics and combine them in the orchestrator.
    - Distinguish per-tile, per-source, dataset-level, and fixed/user-provided normalization explicitly.
    - Freeze and version fitted values before workers apply them to all splits.
    - Optionally publish statistics as artifacts without modifying tile values.
11) Freeze the run plan and initialize worker staging.
    - Freeze the resolved source catalog, configuration, grid, TilePlans, split assignments, selections, and fitted parameters.
    - Create a unique run ID.
    - Create an isolated staging area and any required final output directories.
    - Ensure staging paths and filenames cannot collide between workers.
    - Keep staging on the same filesystem as final local outputs when atomic rename is required.
    - Initialize a coordinator-owned result journal or provisional manifest for crash recovery.
    - Do not allow workers to append concurrently to one JSON or CSV manifest.

## Worker Execution

12) Create source-oriented TileBatches.
    - Prefer batches that process multiple tiles from one source so each source is not repeatedly opened for every tile.
    - Keep batches small enough for reasonable load balancing and memory use.
    - Pass serializable source references, TilePlans, and frozen configuration to workers.
    - Do not pass open Rasterio/GDAL datasets or other non-process-safe handles between processes.
    - Use the same worker contract for sequential and multiprocessing executors.
13) Start worker timing and optional resource tracking.
    - Record batch and tile timing where useful.
    - Ensure optional instrumentation does not change processing results.
14) Open worker-owned source resources and perform content checks.
    - Open each required source asset inside the worker.
    - Confirm that current metadata still agrees with the frozen SourceRecord when required.
    - Treat source-opening or source-level failures as BatchResult information when individual TileResults cannot be created.
15) Read data and source masks for each tile.
    - Read only required windows and requested indices, such as bands, channels, or date/time slices.
    - Read available nodata, QA, cloud, coverage, and label-validity information.
    - Use chunked/windowed access appropriate for source layout and storage backend.
16) Perform mask-aware tile processing.
    - Create source-space validity masks before operations that need them.
    - Warp, reproject, merge, or clip data as required.
    - Use role-appropriate resampling:
        - Bilinear or cubic interpolation for suitable continuous data.
        - Nearest-neighbor interpolation for categorical labels and masks.
        - Custom handling for probabilities, densities, or other special semantics.
    - Update masks after warping, resolution changes, padding, and coverage changes.
    - Reorder dimensions when required, considering current and downstream access efficiency.
    - Apply frozen fitted preprocessing when configured.
    - Apply other explicit per-tile processing, such as fixed clipping or intentionally per-tile scaling.
    - Keep stochastic model-training augmentation downstream by default.
17) Validate and classify each processed tile.
    - Verify expected shape, data type, CRS, transform, resolution, extent, and modality alignment.
    - Verify sufficient valid coverage and application-specific label/input acceptance rules.
    - Assign a structured state:
        - Accepted: processing succeeded and acceptance rules passed.
        - Rejected: processing succeeded, but the tile did not meet an acceptance rule, such as minimum valid coverage.
        - Failed: processing could not be completed because of an error.
        - Skipped: the tile was intentionally omitted, already completed, or filtered by an earlier rule.
18) Stage and validate accepted tile outputs.
    - Write each accepted tile to a unique temporary/staging path.
    - Configure output format, compression, data type, bit depth, metadata, and naming.
    - Close and reopen staged files to verify readability and expected metadata.
    - Do not treat staged output as finally published.
    - For local files, final commit can use atomic rename when staging and final destinations share a filesystem.
    - For cloud/object storage, let the publication backend define commit semantics because rename may be copy-and-delete rather than atomic.
19) Finish worker timing/resource tracking and close resources.
    - Close all worker-owned datasets, handles, and other resources.
    - Flush worker-local output and result fragments when used.
20) Return structured results.
    - Return one TileResult per tile containing:
        - Tile ID
        - Source ID
        - Batch ID
        - Split or explicit no-split value
        - Status
        - Staged output paths
        - Intended final output paths
        - Quality metrics
        - Warnings
        - Error category and message
        - Retryability, when known
        - Timing/resource information
    - Return a BatchResult containing its TileResults plus batch-level warnings, source/opening failures, and timing information.
    - Let the coordinator own the canonical manifest and consume results continuously when result volume is large.

## Finalization

21) Reconcile planned and returned results.
    - Confirm that every planned tile has an accepted, rejected, failed, or skipped result.
    - Detect duplicate or missing results.
    - Collect worker exceptions, process crashes, timeouts, and incomplete batches into structured failure records.
22) Evaluate the configured run-level failure policy.
    - Decide which conditions are errors versus warnings.
    - Support thresholds such as maximum failure percentage, maximum rejection percentage, and minimum accepted counts per split or stratum.
    - Detect systemic failures, such as every tile failing from expired credentials or a missing modality.
    - Permit tolerated individual failures without automatically invalidating the complete dataset.
23) Commit accepted staged outputs.
    - Commit each verified staged output using the selected publication backend.
    - Record the committed state and final path in the coordinator-owned manifest or journal.
    - Make publication idempotent where practical so interrupted runs can be resumed safely.
24) Validate the final dataset.
    - Confirm final file readability.
    - Confirm output metadata, dimensions, transforms, CRS, data types, and checksums when configured.
    - Ensure manifest and file counts reconcile.
    - Ensure split groups do not cross prohibited boundaries.
    - Ensure required classes, strata, modalities, and splits satisfy configured minimums.
    - Confirm that no required output remains only in staging.
25) Finalize artifacts.
    - Write the canonical manifest and any requested plots, metrics, and reports.
    - Include or reference:
        - Code and dependency version information
        - Run ID and configuration
        - Source catalog
        - Grid configuration
        - Nodata, mask, channel, and processing configuration
        - Fitted preprocessing statistics, when applicable
        - Output files and group/split memberships
        - Accepted, rejected, failed, and skipped records
        - Summary quality and performance metrics
    - Preserve enough information to reproduce or diagnose failed runs.
26) Perform narrowly scoped cleanup.
    - Remove temporary files owned by the current run when policy permits.
    - Remove incomplete atomic-write artifacts.
    - Preserve failure reports, logs, frozen configuration, and other artifacts needed for diagnosis or resume.
    - Never delete previously successful outputs unless explicitly requested.
    - Restrict deletion to known run paths or paths containing an expected ownership marker.
27) Write the final run marker.
    - Write _SUCCESS last only when final validation satisfies the configured run-level policy.
    - _SUCCESS can permit documented, tolerated tile failures when the configured policy allows them.
    - On failure, write a structured failure report before optionally writing a _FAIL marker.

## Dry-Run and Testing Support

- Keep core planning and worker functions callable without multiprocessing.
- Support deterministic sequential execution for unit tests and debugging.
- Consider explicit dry-run levels:
    - Plan-only: discovery through frozen TilePlans without pixel processing or final output creation.
    - Read-check: open and inspect representative windows without publishing tiles.
    - Staging run: execute processing and validation without committing final outputs.
    - Limited run: process a deterministic subset of planned tiles.
- Ensure a dry run does not leave final outputs, success markers, or unowned temporary files.
