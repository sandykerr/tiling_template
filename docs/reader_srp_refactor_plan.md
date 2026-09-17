# Reader Single-Responsibility Refactor Plan

## Goal

Reduce the visual and behavioral complexity of the Rasterio and Xarray reader
implementations without changing their existing I/O behavior. The refactor
should isolate spatial-grid interpretation, window geometry, selection,
validity-mask construction, padding, and metadata normalization so each part
can be tested and modified independently.

The refactor should remain lightweight. Prefer small immutable records and
pure functions over additional stateful service classes.

## 1. Correct Existing Small Inconsistencies

Make the low-risk corrections before moving code:

- Correct the `dimension_indices` comment in `WindowReadRequest`. Xarray
  dimension indices are zero-based and may be negative; they are not
  one-based.
- Replace the custom Rasterio tiled-layout calculation with
  `DatasetReader.is_tiled`.
- Add focused regression tests for both corrections.

Completing these first prevents known inconsistencies from being carried into
new abstractions.

## 2. Introduce Backend-Neutral Window Geometry

Create a small module such as `readers/windowing.py` containing an immutable
description of a requested window's relationship to its source:

```python
@dataclass(frozen=True, slots=True)
class WindowGeometry:
    requested: PixelWindow
    source: PixelWindow | None
    destination_row_offset: int
    destination_column_offset: int
    extends_beyond_source: bool
```

Add a pure resolver:

```python
def resolve_window_geometry(
    requested: PixelWindow,
    source_height: int,
    source_width: int,
) -> WindowGeometry:
    ...
```

The resolver should define and test:

- Half-open row and column bounds.
- Fully contained windows.
- Partial intersections on every edge.
- Negative offsets.
- Windows fully outside the source.
- Destination placement for boundless results.

Update Rasterio to use this result for strict-boundary validation. Update
Xarray to use it for source slicing and boundless placement. Do not add
padding or backend-specific I/O to this module.

## 3. Introduce an Xarray Spatial-Grid Record

Move coordinate discovery and regular-grid interpretation out of
`XarrayMetadataReader` into a module such as `readers/xarray_grid.py`:

```python
@dataclass(frozen=True, slots=True)
class XarraySpatialGrid:
    x_dimension: str
    y_dimension: str
    width: int
    height: int
    transform: tuple[float, ...]
    bounds: tuple[float, ...]
    resolution: tuple[float, ...]
```

Provide a public resolver:

```python
def resolve_xarray_spatial_grid(
    dataset: xr.Dataset,
) -> XarraySpatialGrid | None:
    ...
```

Move the following behavior into the resolver and its private helpers:

- X and Y coordinate discovery.
- Coordinate-to-dimension mapping.
- Finite and regularly spaced coordinate validation.
- Pixel-edge transform derivation.
- Bounds and resolution derivation.

Use this abstraction from the metadata reader, window reader, and real-data
check script. No caller should need to invoke private methods on
`XarrayMetadataReader`.

Initially, resolving the grid once per metadata or window operation is
acceptable. After behavior is stable, consider constructing it once per open
session and sharing it between the bound readers if profiling shows repeated
coordinate inspection is meaningful.

## 4. Extract Xarray Variable Selection Validation

Move request and variable validation out of `read_window()` into a pure
function. Its result should describe the validated variable and non-spatial
selection without loading data values.

The function should validate:

- `variable_name` is supplied and exists.
- Rasterio `source_indices` are not present.
- The variable contains both grid dimensions.
- `dimension_indices` do not select spatial dimensions.
- Every selected dimension exists on the variable.
- Every integer index is within the dimension's positive or negative range.

Keep the actual `DataArray.isel()` call in a separate, small selection
function so validation can be tested without performing a window read.

## 5. Extract Transform Translation

Move requested-window transform calculation into a pure helper, for example:

```python
def translate_transform(
    transform: tuple[float, ...],
    window: PixelWindow,
) -> tuple[float, ...]:
    ...
```

Test positive, negative, and zero offsets. Use the helper from Xarray and, if
it improves consistency without fighting Rasterio's API, from Rasterio.

The normalized public representation can remain a tuple even if an internal
implementation uses an affine-transform type.

## 6. Extract Xarray Validity-Mask Construction

Create a pure function responsible only for converting loaded values and
metadata into a Boolean validity mask:

```python
def xarray_valid_mask(
    values: np.ma.MaskedArray,
    nodata: int | float | None,
) -> NDArray[np.bool_]:
    ...
```

Cover these cases independently:

- Unmasked numeric arrays.
- Existing masked-array elements.
- Integer and floating nodata sentinels.
- NaN nodata.
- Other non-finite floating values.
- No configured nodata value.

Keep scale and offset application outside this function. The reader currently
returns stored values according to the Xarray backend configuration, and this
refactor should not silently change that policy.

## 7. Extract Fill Resolution and Boundless Padding

Separate default fill selection from output allocation and placement:

```python
def resolve_fill_value(dtype, requested_fill, nodata):
    ...


def pad_boundless_result(
    data,
    valid_mask,
    geometry,
    output_height,
    output_width,
    y_axis,
    x_axis,
    fill_value,
):
    ...
```

Test:

- Every partially intersecting edge.
- Every corner.
- Fully outside windows.
- Spatial dimensions in different axis positions.
- Preserved non-spatial dimensions.
- Incompatible fill values and dtypes.
- Invalid padding remaining false in the output mask.

After this step, `XarrayWindowReader.read_window()` should primarily
orchestrate validation, geometry resolution, selection, materialization,
masking, optional padding, and result construction.

## 8. Simplify Xarray Metadata Normalization

Extract repeated scalar metadata decoding into a helper shared by nodata,
scale, and offset extraction. Extract variable record construction into a
function such as:

```python
def xarray_variable_metadata(
    name: str,
    variable: xr.DataArray,
) -> VariableMetadata:
    ...
```

Keep CRS discovery separate from variable metadata construction. The final
`read_metadata()` method should visibly perform only these operations:

1. Resolve the spatial grid.
2. Normalize the data variables.
3. Resolve the CRS.
4. Construct `AssetMetadata`.

## 9. Simplify Rasterio Metadata Normalization

Replace the dense multi-sequence `zip()` comprehension with a focused band
normalizer:

```python
def rasterio_variable_metadata(
    dataset: DatasetReader,
    index: int,
    compression: str | None,
) -> VariableMetadata:
    ...
```

The public metadata reader should iterate `dataset.indexes` and assemble the
asset-level metadata. Preserve all current band descriptions, tags, nodata,
scale, offset, unit, block shape, compression, and overview behavior.

## 10. Make Window Result Dimensions Explicit

Add output dimension names to `WindowReadResult`:

```python
dimensions: tuple[str, ...]
```

Use:

- `("band", "y", "x")` for Rasterio results.
- The resulting `DataArray.dims` for Xarray results after integer dimension
  selection.

Validate that the number of dimension names matches `data.ndim`. This removes
the need for downstream code to infer Xarray axis order from the request and
source metadata.

This is a public record change, so complete it only after the internal reader
behavior has been decomposed and stabilized.

## 11. Separate Backend-Specific Selection Types

The current `WindowReadRequest` permits mutually exclusive Rasterio and Xarray
fields. Replace those loose fields with explicit selection records:

```python
@dataclass(frozen=True, slots=True)
class RasterBandSelection:
    source_indices: tuple[int, ...] | None = None


@dataclass(frozen=True, slots=True)
class XarrayVariableSelection:
    variable_name: str
    dimension_indices: tuple[tuple[str, int], ...] = ()
```

Then compose one selection into the shared request:

```python
@dataclass(frozen=True, slots=True)
class WindowReadRequest:
    window: PixelWindow
    selection: RasterBandSelection | XarrayVariableSelection
    boundless: bool = False
    fill_value: int | float | None = None
```

This keeps shared spatial and boundary behavior in one request while making
invalid cross-backend combinations unrepresentable. Update the reader ABC,
scripts, and tests together in this step.

Do this later than the pure-function extractions because it has a wider public
API impact and should not be mixed with changes to spatial or mask behavior.

## 12. Reassess File Boundaries

After the extractions, evaluate file sizes and dependency direction before
creating additional packages. A likely lightweight layout is:

```text
readers/
├── base.py
├── windowing.py
├── rasterio_reader.py
├── xarray_grid.py
├── xarray_values.py
└── xarray_reader.py
```

Only split into `readers/rasterio/` and `readers/xarray/` packages if the
remaining concrete reader modules are still difficult to navigate. Avoid a
generic `utils.py`; name modules after the responsibility they own.

## 13. Run Real-Data and Multiprocessing Regression Checks

After all unit tests pass, rerun the Slurm checks against representative large
GeoTIFF and NetCDF assets. Verify:

- Metadata remains unchanged.
- Window values, masks, dimensions, and transforms remain unchanged.
- Boundless reads retain their requested shape.
- Multiple windows reuse one open backend resource.
- Worker processes continue to open independent resources.
- Progress reporting remains parent-owned and stable.
- Repeated Xarray window reads do not repeatedly perform unnecessary spatial
  coordinate work if session-level grid reuse was introduced.

Benchmark before adding caching beyond the open backend session. The purpose
of this refactor is clearer responsibility and easier bug isolation, not new
I/O policy.

