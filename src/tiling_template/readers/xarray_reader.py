from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray
import xarray as xr

from .base import AssetReaderBackend, MetadataReader, WindowReader
from ..records import (
    AssetMetadata,
    AssetRef,
    VariableMetadata,
    VariableStorageMetadata,
)


@dataclass(frozen=True, slots=True)
class XarrayMetadataReader(MetadataReader):
    """Read metadata from NetCDF-style xarray datasets."""

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

    def read_metadata(self, asset: AssetRef) -> AssetMetadata:
        with xr.open_dataset(
            asset.path,
            decode_cf=False,
            mask_and_scale=False,
            cache=False,
        ) as dataset:
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
                asset=asset,
                variables=variables,
                crs=self._crs(dataset),
                transform=transform,
                bounds=bounds,
                resolution=resolution,
                attributes=dict(dataset.attrs),
            )


@dataclass(frozen=True, slots=True)
class XarrayWindowReader(WindowReader):
    """Read array windows through xarray."""

    def read_window(self, asset: AssetRef) -> NDArray[np.generic]:
        raise NotImplementedError(
            "Xarray window reading has not been implemented."
        )


def xarray_backend() -> AssetReaderBackend:
    """Construct the standard xarray reader backend."""

    return AssetReaderBackend(
        metadata_reader=XarrayMetadataReader(),
        window_reader=XarrayWindowReader(),
    )
