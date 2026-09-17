# Reader Backend Refactor for Large GeoTIFFs

## Goal

Evolve the reader layer toward worker-local, asset-bound sessions that open each
large file once, read metadata once, serve many small windows through the same
backend handle, and close resources deterministically.

Opening a Rasterio dataset does **not** load the complete GeoTIFF into memory.
Only requested pixel windows and a bounded internal block cache consume image
data memory.

The target usage is:

```python
backend = registry.backend_for(asset)

with backend.open(asset) as session:
    metadata = session.metadata_reader.read_metadata()

    for request in window_requests:
        result = session.window_reader.read_window(request)
```

## Sequential Refactor

### 1. Define the lifecycle boundary

Use one open session for one asset within one worker:

```text
worker starts
    -> opens one asset
    -> reads metadata
    -> reads many windows
    -> closes the asset
```

Open Rasterio and xarray objects must be created inside the worker that uses
them. Do not pickle them, pass them between processes, or share them across
processes.

For a multimodal `SourceRecord`, a worker can use `contextlib.ExitStack` to
manage one session per asset.

### 2. Make `AssetReaderBackend` a backend factory

Replace the current container of already-created readers with an interface that
knows how to open an asset:

```python
class AssetReaderBackend(ABC):
    @abstractmethod
    def open(
        self,
        asset: AssetRef,
    ) -> ContextManager[AssetReadSession]:
        ...
```

Concrete implementations would initially be `RasterioBackend` and
`XarrayBackend`.

The registry continues to select a backend by file extension, but selecting a
backend does not open the file:

```python
backend = registry.backend_for(asset)
```

### 3. Introduce an asset-bound session

The backend should open its resource and return a session containing readers
bound to that resource:

```python
@dataclass(slots=True)
class AssetReadSession:
    metadata_reader: MetadataReader
    window_reader: WindowReader
```

A Rasterio backend would conceptually behave like this:

```python
class RasterioBackend(AssetReaderBackend):
    @contextmanager
    def open(self, asset: AssetRef):
        with rasterio.open(asset.path) as dataset:
            yield AssetReadSession(
                metadata_reader=RasterioMetadataReader(asset, dataset),
                window_reader=RasterioWindowReader(asset, dataset),
            )
```

### 4. Bind each reader to its asset and open handle

Remove the asset parameter from operations performed by a bound reader:

```python
metadata = session.metadata_reader.read_metadata()
result = session.window_reader.read_window(request)
```

This prevents a caller from accidentally supplying an asset that does not
correspond to the reader's open dataset.

### 5. Adapt the existing metadata readers

Move only resource opening and closing out of the metadata readers. Preserve
their current extraction logic.

For example:

```python
class RasterioMetadataReader(MetadataReader):
    def __init__(
        self,
        asset: AssetRef,
        dataset: rasterio.DatasetReader,
    ):
        self.asset = asset
        self.dataset = dataset

    def read_metadata(self) -> AssetMetadata:
        dataset = self.dataset
        ...
```

Apply the same lifecycle to xarray: the backend opens the dataset, and its
metadata and window readers share that open dataset.

### 6. Add immutable backend configuration

Store only file-access and backend-resource options on the backend:

```python
@dataclass(frozen=True, slots=True)
class RasterioBackendConfig:
    sharing: bool = False
    gdal_options: Mapping[str, object] = field(default_factory=dict)
```

A backend can apply those options while opening an asset:

```python
with rasterio.Env(**config.gdal_options):
    with rasterio.open(asset.path, sharing=config.sharing) as dataset:
        ...
```

Tile dimensions, overlap, padding, sampling, and processing policy do not
belong in backend configuration.

### 7. Design `WindowReadRequest`

Before implementing concrete window reads, define the request's exact pixel
and boundary conventions. A starting shape is:

```python
@dataclass(frozen=True, slots=True)
class PixelWindow:
    row_offset: int
    column_offset: int
    height: int
    width: int


@dataclass(frozen=True, slots=True)
class WindowReadRequest:
    window: PixelWindow
    source_indices: tuple[int, ...] | None = None
    boundless: bool = False
    fill_value: int | float | None = None
```

Use half-open pixel bounds consistently:

```text
rows    [row_offset, row_offset + height)
columns [column_offset, column_offset + width)
```

Potential later fields include output shape, resampling, masked-array behavior,
and separate read and core windows.

### 8. Return a structured window result

A bare array is unlikely to contain enough information for padding, masks, and
multimodal alignment. Define a structured result such as:

```python
@dataclass(frozen=True, slots=True)
class WindowReadResult:
    data: NDArray[np.generic]
    valid_mask: NDArray[np.bool_] | None
    transform: tuple[float, ...]
    request: WindowReadRequest
```

When present, `valid_mask` has the same shape as `data` and represents
per-element validity. A clipped `source_window` can be added later if
boundless edge handling needs to distinguish requested and intersecting areas.

### 9. Implement the Rasterio window reader first

The core operation will resemble:

```python
data = dataset.read(
    indexes=request.source_indices,
    window=window,
    boundless=request.boundless,
    fill_value=request.fill_value,
)

transform = dataset.window_transform(window)
```

Keep reprojection and general tile processing outside the reader. Read-time
resampling through `out_shape` can be added later when it is an intentional I/O
optimization.

### 10. Use source- or batch-oriented worker tasks

Prefer this model:

```text
worker receives a source
    -> opens each modality once
    -> reads many spatially nearby windows
    -> closes all assets
```

Avoid creating one process task per tile when that causes the same source file
to be opened and closed repeatedly.

Spatially ordering windows can improve reuse of TIFF blocks already held by
GDAL's cache.

### 11. Add lifecycle and window tests

Tests should verify:

- One `rasterio.open()` call serves multiple window reads.
- The dataset remains open for the session lifetime.
- The dataset closes after normal completion.
- The dataset closes when processing raises an exception.
- Metadata and window readers share the same dataset handle.
- A closed session cannot continue reading.
- Pixel windows return the expected values and transform.
- Band selection works.
- Edge, boundless, fill, and mask behavior are correct.
- Separate worker processes create separate dataset handles.

### 12. Benchmark before adding manual block logic

Use representative tiled and compressed GeoTIFFs to compare:

- Reopening per tile versus one persistent handle.
- Per-tile versus per-source worker tasks.
- Spatially ordered versus random window order.
- Different worker counts.
- Different GDAL cache sizes.
- Window dimensions relative to TIFF block dimensions.

Rasterio already reads and caches the underlying TIFF blocks required for a
window. Do not implement custom block alignment or caching until measurements
show a meaningful benefit.

## Implementation Status

Steps 1 through 9 are implemented for the large-GeoTIFF path. Rasterio uses
one worker-local handle for metadata and repeated window reads. Window reads
are band-first, preserve requested band order, return a per-element validity
mask, and report the transform of the requested half-open pixel window.

Non-boundless reads must be fully contained by the source. Boundless reads
preserve the requested output shape, use the configured fill value outside
the source, and mark padded elements invalid.

Step 10 is an orchestration concern: future workers should group work by
source or source batch and use `ExitStack` for multimodal sources. The reader
API already supports that lifecycle; no multiprocessing policy belongs in the
reader package itself.

Synthetic tests cover the lifecycle and window behavior listed in step 11,
except the separate-process integration case. Representative real-data,
multiprocessing, and performance tests from steps 11 and 12 remain to be
designed against actual project workloads.

Xarray window reading remains intentionally unimplemented. A useful NetCDF
request must select named variables and non-spatial dimensions; Rasterio's
one-based `source_indices` contract is not sufficient for that API.
