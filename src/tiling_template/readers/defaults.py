from .base import ReaderRegistry


def default_reader_registry() -> ReaderRegistry:
    from .rasterio import RasterioBackend
    from .xarray import XarrayBackend

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
