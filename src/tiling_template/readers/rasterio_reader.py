from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import rasterio
from rasterio.io import DatasetReader
from rasterio.windows import Window

from ..configs.reader import RasterioBackendConfig
from .base import (
    AssetReadSession,
    AssetReaderBackend,
    MetadataReader,
    WindowReader,
)
from ..records import (
    AssetMetadata,
    AssetRef,
    VariableMetadata,
    VariableStorageMetadata,
    WindowReadRequest,
    WindowReadResult,
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
                strict=True,
            )
        )
        return AssetMetadata(
            asset=self.asset,
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
        if request.variable_name is not None or request.dimension_indices:
            raise ValueError(
                "Rasterio window reads do not accept Xarray variable or "
                "dimension selectors."
            )

        src = self.dataset
        window = Window(
            col_off=request.window.column_offset,
            row_off=request.window.row_offset,
            width=request.window.width,
            height=request.window.height,
        )
        source_indices = request.source_indices or tuple(src.indexes)

        invalid_indices = set(source_indices) - set(src.indexes)
        if invalid_indices:
            raise ValueError(
                "Source indices are not present in the dataset: "
                f"{sorted(invalid_indices)}"
            )

        row_end = request.window.row_offset + request.window.height
        column_end = request.window.column_offset + request.window.width
        extends_beyond_dataset = (
            request.window.row_offset < 0
            or request.window.column_offset < 0
            or row_end > src.height
            or column_end > src.width
        )
        if extends_beyond_dataset and not request.boundless:
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
