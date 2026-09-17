from numpy.typing import NDArray
import numpy as np
import rasterio
from dataclasses import dataclass

from .base import AssetReaderBackend, MetadataReader, WindowReader
from ..records import (
    AssetMetadata,
    AssetRef,
    VariableMetadata,
    VariableStorageMetadata,
)


@dataclass(frozen=True, slots=True)
class RasterioMetadataReader(MetadataReader):
    """Read normalized metadata through Rasterio."""

    def read_metadata(self, asset: AssetRef) -> AssetMetadata:
        with rasterio.open(asset.path) as src:
            compression = (
                src.compression.value.lower()
                if src.compression is not None
                else None
            )
            variables = tuple(
                VariableMetadata(
                    name=band_name or f"band_{i}",
                    shape=(src.height, src.width),
                    dimensions=("y", "x"),
                    dtype=str(dtype),
                    source_index=i,
                    nodata=nodata,
                    scale=scale,
                    offset=offset,
                    unit=unit,
                    attributes=src.tags(i),
                    storage=VariableStorageMetadata(
                        chunk_shape=tuple(block_shape),
                        compression=compression,
                        overview_factors=tuple(src.overviews(i)),
                    ),
                )
                for (
                    i,
                    band_name,
                    dtype,
                    nodata,
                    scale,
                    offset,
                    unit,
                    block_shape,
                ) in zip(
                    src.indexes,
                    src.descriptions,
                    src.dtypes,
                    src.nodatavals,
                    src.scales,
                    src.offsets,
                    src.units,
                    src.block_shapes,
                    strict=True
                )
            )
            asset_metadata = AssetMetadata(
                asset=asset,
                variables=variables,
                crs=src.crs.to_string() if src.crs is not None else None,
                transform=tuple(src.transform),
                bounds=tuple(src.bounds),
                resolution=tuple(src.res),
                is_tiled=(
                    bool(src.block_shapes)
                    and all(
                        src.width != block_width
                        for _, block_width in src.block_shapes
                    )
                ),
                attributes=src.tags()
            )
            return asset_metadata


@dataclass(frozen=True, slots=True)
class RasterioWindowReader(WindowReader):
    """Read array windows through Rasterio."""

    def read_window(self, asset: AssetRef) -> NDArray[np.generic]:
        raise NotImplementedError(
            "Rasterio window reading has not been implemented."
        )


def rasterio_backend() -> AssetReaderBackend:
    """Construct the standard Rasterio reader backend."""

    return AssetReaderBackend(
        metadata_reader=RasterioMetadataReader(),
        window_reader=RasterioWindowReader(),
    )
