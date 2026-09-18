from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib import import_module

import numpy as np
import xarray as xr

from ...configs.reader import XarrayBackendConfig
from ..base import (
    AssetReadSession,
    AssetReaderBackend,
    MetadataReader,
    WindowReader,
)
from ...records import (
    AssetMetadata,
    AssetRef,
    WindowReadRequest,
    WindowReadResult,
)
from ..windowing import resolve_window_geometry, translate_transform
from .grid import resolve_xarray_spatial_grid
from .metadata import (
    xarray_nodata_value,
    xarray_variable_metadata,
)
from .selection import (
    select_xarray_variable,
    validate_xarray_selection,
)
from .values import (
    pad_boundless_result,
    resolve_fill_value,
    xarray_valid_mask,
)


@dataclass(frozen=True, slots=True)
class XarrayMetadataReader(MetadataReader):
    """Read metadata from NetCDF-style xarray datasets."""

    asset: AssetRef
    dataset: xr.Dataset

    @staticmethod
    def _grid_mapping_names(dataset: xr.Dataset) -> set[str]:
        return {
            name
            for variable in dataset.data_vars.values()
            if isinstance(name := variable.attrs.get("grid_mapping"), str)
        }

    @classmethod
    def _metadata_sources(cls, dataset: xr.Dataset) -> tuple[Mapping, ...]:
        sources: list[Mapping] = [dataset.attrs]
        candidate_names = cls._grid_mapping_names(dataset) | {
            "spatial_ref",
            "crs",
        }

        for name in candidate_names:
            if name in dataset.variables:
                sources.append(dataset[name].attrs)

        return tuple(sources)

    @classmethod
    def _crs(cls, dataset: xr.Dataset) -> str | None:
        for metadata in cls._metadata_sources(dataset):
            for key in ("crs_wkt", "spatial_ref", "epsg_code", "crs"):
                value = metadata.get(key)
                if value is None:
                    continue

                if isinstance(value, np.generic):
                    value = value.item()
                if isinstance(value, bytes):
                    value = value.decode()

                text = str(value).strip()
                if not text:
                    continue
                if key == "epsg_code" and text.isdigit():
                    return f"EPSG:{text}"
                return text

        return None

    def read_metadata(self) -> AssetMetadata:
        dataset = self.dataset
        grid = resolve_xarray_spatial_grid(dataset)
        grid_mapping_names = self._grid_mapping_names(dataset)
        variables = tuple(
            xarray_variable_metadata(str(name), variable)
            for name, variable in dataset.data_vars.items()
            if name not in grid_mapping_names
        )
        crs = self._crs(dataset)

        return AssetMetadata(
            asset=self.asset,
            variables=variables,
            crs=crs,
            transform=grid.transform if grid is not None else None,
            bounds=grid.bounds if grid is not None else None,
            resolution=grid.resolution if grid is not None else None,
            attributes=dict(dataset.attrs),
        )


@dataclass(frozen=True, slots=True)
class XarrayWindowReader(WindowReader):
    """Read one named variable from an open xarray dataset."""

    asset: AssetRef
    dataset: xr.Dataset

    def read_window(
        self,
        request: WindowReadRequest,
    ) -> WindowReadResult:
        grid = resolve_xarray_spatial_grid(self.dataset)
        if grid is None:
            raise ValueError(
                "Xarray window reads require a regular spatial grid."
            )
        selection = validate_xarray_selection(self.dataset, request, grid)
        variable = selection.variable
        selected = select_xarray_variable(selection)
        geometry = resolve_window_geometry(
            request.window,
            source_height=grid.height,
            source_width=grid.width,
        )
        if geometry.extends_beyond_source and not request.boundless:
            raise ValueError(
                "Window extends beyond the dataset; set boundless=True "
                "to pad the requested extent."
            )

        source_window = geometry.source
        if source_window is None:
            row_slice = slice(0, 0)
            column_slice = slice(0, 0)
        else:
            row_slice = slice(
                source_window.row_offset,
                source_window.row_offset + source_window.height,
            )
            column_slice = slice(
                source_window.column_offset,
                source_window.column_offset + source_window.width,
            )
        windowed = selected.isel(
            {
                grid.y_dimension: row_slice,
                grid.x_dimension: column_slice,
            }
        )

        masked_data = np.ma.asarray(windowed.values)
        data = np.asarray(np.ma.getdata(masked_data))
        nodata = xarray_nodata_value(variable)
        valid_mask = xarray_valid_mask(masked_data, nodata)

        if request.boundless and geometry.extends_beyond_source:
            y_axis = windowed.dims.index(grid.y_dimension)
            x_axis = windowed.dims.index(grid.x_dimension)
            fill_value = resolve_fill_value(
                data.dtype,
                request.fill_value,
                nodata,
            )
            data, valid_mask = pad_boundless_result(
                data,
                valid_mask,
                geometry,
                output_height=request.window.height,
                output_width=request.window.width,
                y_axis=y_axis,
                x_axis=x_axis,
                fill_value=fill_value,
            )

        return WindowReadResult(
            data=data,
            valid_mask=valid_mask,
            dimensions=tuple(str(dimension) for dimension in windowed.dims),
            transform=translate_transform(grid.transform, request.window),
            request=request,
        )


@dataclass(frozen=True, slots=True)
class XarrayBackend(AssetReaderBackend):
    """Open worker-local xarray reader sessions."""

    config: XarrayBackendConfig = field(
        default_factory=XarrayBackendConfig
    )

    @contextmanager
    def open(self, asset: AssetRef) -> Iterator[AssetReadSession]:
        for module_name in self.config.plugin_modules:
            try:
                import_module(module_name)
            except ImportError as error:
                raise ImportError(
                    "Unable to initialize Xarray plugin module "
                    f"{module_name!r}."
                ) from error

        with xr.open_dataset(
            asset.path,
            engine=self.config.engine,
            group=self.config.group,
            decode_cf=self.config.decode_cf,
            mask_and_scale=self.config.mask_and_scale,
            cache=self.config.cache,
        ) as dataset:
            yield AssetReadSession(
                metadata_reader=XarrayMetadataReader(asset, dataset),
                window_reader=XarrayWindowReader(asset, dataset),
            )
