from .base import ReaderRegistry


def default_reader_registry() -> ReaderRegistry:
    from .rasterio_reader import rasterio_backend
    from .xarray_reader import xarray_backend

    registry = ReaderRegistry()
    registry.register(
        (".tif", ".tiff"),
        rasterio_backend,
    )
    registry.register(
        (".nc",),
        xarray_backend,
    )
    return registry
