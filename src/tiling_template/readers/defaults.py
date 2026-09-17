from .base import ReaderRegistry


def default_reader_registry() -> ReaderRegistry:
    from .rasterio_reader import RasterioBackend
    from .xarray_reader import XarrayBackend

    registry = ReaderRegistry()
    registry.register(
        (".tif", ".tiff"),
        RasterioBackend,
    )
    registry.register(
        (".nc",),
        XarrayBackend,
    )
    return registry
