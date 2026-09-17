from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from ..configs.reader import XarrayBackendConfig
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
class XarrayMetadataReader(MetadataReader):
    """Read metadata from NetCDF-style xarray datasets."""

    asset: AssetRef
    dataset: xr.Dataset

    @staticmethod
    def _nodata_value(variable: xr.DataArray) -> int | float | None:
        for metadata in (variable.attrs, variable.encoding):
            for key in ("_FillValue", "missing_value"):
                value = metadata.get(key)
                if value is None:
                    continue

                array = np.asarray(value)
                if array.size != 1:
                    continue

                scalar = array.item()
                if isinstance(scalar, bool):
                    continue
                if isinstance(scalar, (int, float)):
                    return scalar

        return None

    @staticmethod
    def _numeric_metadata(
        variable: xr.DataArray,
        key: str,
    ) -> int | float | None:
        for metadata in (variable.attrs, variable.encoding):
            value = metadata.get(key)
            if value is None:
                continue

            array = np.asarray(value)
            if array.size != 1:
                continue

            scalar = array.item()
            if not isinstance(scalar, bool) and isinstance(scalar, (int, float)):
                return scalar

        return None

    @staticmethod
    def _storage_metadata(variable: xr.DataArray) -> VariableStorageMetadata:
        encoding = variable.encoding
        chunk_shape = encoding.get("chunksizes")
        if chunk_shape is not None:
            chunk_shape = tuple(int(size) for size in chunk_shape)

        compression = encoding.get("compression")
        if compression is None and encoding.get("zlib"):
            compression = "zlib"
        if compression is not None:
            compression = str(compression).lower()

        compression_level = encoding.get("complevel")
        if compression_level is not None:
            compression_level = int(compression_level)

        shuffle = encoding.get("shuffle")
        if shuffle is not None:
            shuffle = bool(shuffle)

        return VariableStorageMetadata(
            chunk_shape=chunk_shape,
            compression=compression,
            compression_level=compression_level,
            shuffle=shuffle,
        )

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

    @staticmethod
    def _coordinate_for_axis(
        dataset: xr.Dataset,
        axis: str,
    ) -> xr.DataArray | None:
        preferred_names = {
            "X": ("x", "lon", "longitude"),
            "Y": ("y", "lat", "latitude"),
        }
        standard_names = {
            "X": ("projection_x_coordinate", "longitude"),
            "Y": ("projection_y_coordinate", "latitude"),
        }

        for name in preferred_names[axis]:
            coordinate = dataset.coords.get(name)
            if coordinate is not None and coordinate.ndim == 1:
                return coordinate

        for coordinate in dataset.coords.values():
            if coordinate.ndim != 1:
                continue
            coordinate_axis = str(coordinate.attrs.get("axis", "")).upper()
            standard_name = coordinate.attrs.get("standard_name")
            if (
                coordinate_axis == axis
                or standard_name in standard_names[axis]
            ):
                return coordinate

        return None

    @classmethod
    def _spatial_metadata(
        cls,
        dataset: xr.Dataset,
    ) -> tuple[
        tuple[float, ...] | None,
        tuple[float, ...] | None,
        tuple[float, ...] | None,
    ]:
        x_coordinate = cls._coordinate_for_axis(dataset, "X")
        y_coordinate = cls._coordinate_for_axis(dataset, "Y")
        if x_coordinate is None or y_coordinate is None:
            return None, None, None

        x_dimension = x_coordinate.dims[0]
        y_dimension = y_coordinate.dims[0]
        if not any(
            x_dimension in variable.dims and y_dimension in variable.dims
            for variable in dataset.data_vars.values()
        ):
            return None, None, None

        try:
            x_values = np.asarray(x_coordinate.values, dtype=float)
            y_values = np.asarray(y_coordinate.values, dtype=float)
        except (TypeError, ValueError):
            return None, None, None

        if x_values.size < 2 or y_values.size < 2:
            return None, None, None

        x_differences = np.diff(x_values)
        y_differences = np.diff(y_values)
        if (
            not np.all(np.isfinite(x_values))
            or not np.all(np.isfinite(y_values))
            or not np.allclose(x_differences, x_differences[0])
            or not np.allclose(y_differences, y_differences[0])
            or x_differences[0] == 0
            or y_differences[0] == 0
        ):
            return None, None, None

        x_step = float(x_differences[0])
        y_step = float(y_differences[0])
        x_edges = (
            float(x_values[0] - x_step / 2),
            float(x_values[-1] + x_step / 2),
        )
        y_edges = (
            float(y_values[0] - y_step / 2),
            float(y_values[-1] + y_step / 2),
        )
        transform = (
            x_step,
            0.0,
            x_edges[0],
            0.0,
            y_step,
            y_edges[0],
            0.0,
            0.0,
            1.0,
        )
        bounds = (
            min(x_edges),
            min(y_edges),
            max(x_edges),
            max(y_edges),
        )
        resolution = (abs(x_step), abs(y_step))
        return transform, bounds, resolution

    def read_metadata(self) -> AssetMetadata:
        dataset = self.dataset
        grid_mapping_names = self._grid_mapping_names(dataset)
        variables = tuple(
            VariableMetadata(
                name=str(name),
                shape=tuple(int(size) for size in variable.shape),
                dimensions=tuple(
                    str(dimension) for dimension in variable.dims
                ),
                dtype=str(variable.dtype),
                nodata=self._nodata_value(variable),
                scale=self._numeric_metadata(variable, "scale_factor"),
                offset=self._numeric_metadata(variable, "add_offset"),
                unit=(
                    str(variable.attrs["units"])
                    if variable.attrs.get("units") is not None
                    else None
                ),
                attributes=dict(variable.attrs),
                storage=self._storage_metadata(variable),
            )
            for name, variable in dataset.data_vars.items()
            if name not in grid_mapping_names
        )
        transform, bounds, resolution = self._spatial_metadata(dataset)

        return AssetMetadata(
            asset=self.asset,
            variables=variables,
            crs=self._crs(dataset),
            transform=transform,
            bounds=bounds,
            resolution=resolution,
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
        if request.source_indices is not None:
            raise ValueError(
                "Xarray window reads use variable_name, not source_indices."
            )
        if request.variable_name is None:
            raise ValueError(
                "Xarray window reads require an explicit variable_name."
            )
        if request.variable_name not in self.dataset.data_vars:
            raise ValueError(
                f"Variable is not present in the dataset: "
                f"{request.variable_name}"
            )

        variable = self.dataset[request.variable_name]
        x_coordinate = XarrayMetadataReader._coordinate_for_axis(
            self.dataset,
            "X",
        )
        y_coordinate = XarrayMetadataReader._coordinate_for_axis(
            self.dataset,
            "Y",
        )
        if x_coordinate is None or y_coordinate is None:
            raise ValueError(
                "Dataset does not define one-dimensional spatial coordinates."
            )

        x_dimension = x_coordinate.dims[0]
        y_dimension = y_coordinate.dims[0]
        if x_dimension not in variable.dims or y_dimension not in variable.dims:
            raise ValueError(
                f"Variable {request.variable_name!r} does not contain both "
                "spatial dimensions."
            )

        transform, _, _ = XarrayMetadataReader._spatial_metadata(self.dataset)
        if transform is None:
            raise ValueError(
                "Xarray window reads require a regular spatial grid."
            )

        indexers = dict(request.dimension_indices)
        spatial_selectors = {x_dimension, y_dimension} & indexers.keys()
        if spatial_selectors:
            raise ValueError(
                "Spatial dimensions are selected by PixelWindow, not "
                f"dimension_indices: {sorted(spatial_selectors)}"
            )
        unknown_dimensions = set(indexers) - set(variable.dims)
        if unknown_dimensions:
            raise ValueError(
                "Dimensions are not present on the selected variable: "
                f"{sorted(unknown_dimensions)}"
            )
        for dimension, index in indexers.items():
            dimension_size = variable.sizes[dimension]
            if not -dimension_size <= index < dimension_size:
                raise ValueError(
                    f"Index {index} is outside dimension {dimension!r} "
                    f"with size {dimension_size}."
                )

        selected = variable.isel(indexers)
        source_height = variable.sizes[y_dimension]
        source_width = variable.sizes[x_dimension]
        row_start = request.window.row_offset
        column_start = request.window.column_offset
        row_end = row_start + request.window.height
        column_end = column_start + request.window.width
        extends_beyond_dataset = (
            row_start < 0
            or column_start < 0
            or row_end > source_height
            or column_end > source_width
        )
        if extends_beyond_dataset and not request.boundless:
            raise ValueError(
                "Window extends beyond the dataset; set boundless=True "
                "to pad the requested extent."
            )

        clipped_row_start = min(source_height, max(0, row_start))
        clipped_row_end = min(source_height, max(0, row_end))
        clipped_column_start = min(source_width, max(0, column_start))
        clipped_column_end = min(source_width, max(0, column_end))
        windowed = selected.isel(
            {
                y_dimension: slice(clipped_row_start, clipped_row_end),
                x_dimension: slice(
                    clipped_column_start,
                    clipped_column_end,
                ),
            }
        )

        masked_data = np.ma.asarray(windowed.values)
        data = np.asarray(np.ma.getdata(masked_data))
        valid_mask = np.asarray(
            ~np.ma.getmaskarray(masked_data),
            dtype=np.bool_,
        )
        if np.issubdtype(data.dtype, np.inexact):
            valid_mask &= np.isfinite(data)

        nodata = XarrayMetadataReader._nodata_value(variable)
        if nodata is not None:
            if isinstance(nodata, float) and np.isnan(nodata):
                valid_mask &= ~np.isnan(data)
            else:
                valid_mask &= data != nodata

        if request.boundless and extends_beyond_dataset:
            fill_value = request.fill_value
            if fill_value is None:
                if nodata is not None:
                    fill_value = nodata
                elif np.issubdtype(data.dtype, np.inexact):
                    fill_value = np.nan
                else:
                    fill_value = 0

            y_axis = windowed.dims.index(y_dimension)
            x_axis = windowed.dims.index(x_dimension)
            output_shape = list(data.shape)
            output_shape[y_axis] = request.window.height
            output_shape[x_axis] = request.window.width
            try:
                output_data = np.full(
                    output_shape,
                    fill_value,
                    dtype=data.dtype,
                )
            except (OverflowError, TypeError, ValueError) as error:
                raise ValueError(
                    f"fill_value {fill_value!r} is incompatible with "
                    f"variable dtype {data.dtype}."
                ) from error
            output_mask = np.zeros(output_shape, dtype=np.bool_)

            if data.shape[y_axis] > 0 and data.shape[x_axis] > 0:
                destination = [slice(None)] * data.ndim
                destination[y_axis] = slice(
                    clipped_row_start - row_start,
                    clipped_row_end - row_start,
                )
                destination[x_axis] = slice(
                    clipped_column_start - column_start,
                    clipped_column_end - column_start,
                )
                output_data[tuple(destination)] = data
                output_mask[tuple(destination)] = valid_mask

            data = output_data
            valid_mask = output_mask

        a, b, c, d, e, f, g, h, i = transform
        window_transform = (
            a,
            b,
            c + a * column_start + b * row_start,
            d,
            e,
            f + d * column_start + e * row_start,
            g,
            h,
            i,
        )
        return WindowReadResult(
            data=data,
            valid_mask=valid_mask,
            transform=window_transform,
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
