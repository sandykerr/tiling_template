from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import rasterio
from rasterio.io import DatasetReader
from rasterio.windows import Window

from ...configs.reader import RasterioBackendConfig
from ..base import (
    AssetReadSession,
    AssetReaderBackend,
    MetadataReader,
    WindowReader,
)
from ...records import (
    AssetMetadata,
    AssetRef,
    RasterBandSelection,
    VariableMetadata,
    VariableStorageMetadata,
    WindowReadRequest,
    WindowReadResult,
)
from ..windowing import resolve_window_geometry


def rasterio_variable_metadata(
    dataset: DatasetReader,
    index: int,
    compression: str | None,
) -> VariableMetadata:
    """Normalize one Rasterio band into a backend-neutral record."""

    position = dataset.indexes.index(index)
    band_name = dataset.descriptions[position]
    return VariableMetadata(
        name=band_name or f"band_{index}",
        shape=(dataset.height, dataset.width),
        dimensions=("y", "x"),
        dtype=str(dataset.dtypes[position]),
        source_index=index,
        nodata=dataset.nodatavals[position],
        scale=dataset.scales[position],
        offset=dataset.offsets[position],
        unit=dataset.units[position],
        attributes=dataset.tags(index),
        storage=VariableStorageMetadata(
            chunk_shape=tuple(dataset.block_shapes[position]),
            compression=compression,
            overview_factors=tuple(dataset.overviews(index)),
        ),
    )


@dataclass(frozen=True, slots=True)
class RasterioMetadataReader(MetadataReader):
    """Read normalized metadata from an open Rasterio dataset."""

    asset: AssetRef
    dataset: DatasetReader

    def read_metadata(self) -> AssetMetadata:
        src = self.dataset
        compression = (
            src.compression.value.lower()
            if src.compression is not None
            else None
        )
        variables = tuple(
            rasterio_variable_metadata(src, index, compression)
            for index in src.indexes
        )
        return AssetMetadata(
            asset=self.asset,
            variables=variables,
            crs=src.crs.to_string() if src.crs is not None else None,
            transform=tuple(src.transform),
            bounds=tuple(src.bounds),
            resolution=tuple(src.res),
            is_tiled=src.is_tiled,
            attributes=src.tags(),
        )


@dataclass(frozen=True, slots=True)
class RasterioWindowReader(WindowReader):
    """Read band-first pixel windows from an open Rasterio dataset."""

    asset: AssetRef
    dataset: DatasetReader

    def read_window(
        self,
        request: WindowReadRequest,
    ) -> WindowReadResult:
        if not isinstance(request.selection, RasterBandSelection):
            raise ValueError(
                "Rasterio window reads require RasterBandSelection, not "
                "XarrayVariableSelection."
            )

        src = self.dataset
        window = Window(
            col_off=request.window.column_offset,
            row_off=request.window.row_offset,
            width=request.window.width,
            height=request.window.height,
        )
        source_indices = (
            request.selection.source_indices or tuple(src.indexes)
        )

        invalid_indices = set(source_indices) - set(src.indexes)
        if invalid_indices:
            raise ValueError(
                "Source indices are not present in the dataset: "
                f"{sorted(invalid_indices)}"
            )

        geometry = resolve_window_geometry(
            request.window,
            source_height=src.height,
            source_width=src.width,
        )
        if geometry.extends_beyond_source and not request.boundless:
            raise ValueError(
                "Window extends beyond the dataset; set boundless=True "
                "to pad the requested extent."
            )

        read_options: dict[str, object] = {
            "indexes": source_indices,
            "window": window,
            "boundless": request.boundless,
        }
        if request.fill_value is not None:
            read_options["fill_value"] = request.fill_value

        data = np.asarray(src.read(**read_options))
        valid_mask = np.asarray(
            src.read_masks(
                indexes=source_indices,
                window=window,
                boundless=request.boundless,
            )
            > 0,
            dtype=np.bool_,
        )

        return WindowReadResult(
            data=data,
            valid_mask=valid_mask,
            dimensions=("band", "y", "x"),
            transform=tuple(src.window_transform(window)),
            request=request,
        )


@dataclass(frozen=True, slots=True)
class RasterioBackend(AssetReaderBackend):
    """Open worker-local Rasterio reader sessions."""

    config: RasterioBackendConfig = field(
        default_factory=RasterioBackendConfig
    )

    @contextmanager
    def open(self, asset: AssetRef) -> Iterator[AssetReadSession]:
        with rasterio.Env(**self.config.options_dict()):
            with rasterio.open(
                asset.path,
                sharing=self.config.sharing,
            ) as dataset:
                yield AssetReadSession(
                    metadata_reader=RasterioMetadataReader(asset, dataset),
                    window_reader=RasterioWindowReader(asset, dataset),
                )
