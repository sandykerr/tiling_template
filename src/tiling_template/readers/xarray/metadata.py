import numpy as np
import xarray as xr

from ...records import VariableMetadata, VariableStorageMetadata


def xarray_numeric_metadata(
    variable: xr.DataArray,
    keys: tuple[str, ...],
) -> int | float | None:
    """Return the first scalar numeric value for ordered metadata keys."""

    for metadata in (variable.attrs, variable.encoding):
        for key in keys:
            value = metadata.get(key)
            if value is None:
                continue

            array = np.asarray(value)
            if array.size != 1:
                continue

            scalar = array.item()
            if not isinstance(scalar, bool) and isinstance(
                scalar,
                (int, float),
            ):
                return scalar

    return None


def xarray_nodata_value(
    variable: xr.DataArray,
) -> int | float | None:
    """Return a variable's normalized fill or missing-value sentinel."""

    return xarray_numeric_metadata(
        variable,
        ("_FillValue", "missing_value"),
    )


def _xarray_storage_metadata(
    variable: xr.DataArray,
) -> VariableStorageMetadata:
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


def xarray_variable_metadata(
    name: str,
    variable: xr.DataArray,
) -> VariableMetadata:
    """Normalize one Xarray variable into a backend-neutral record."""

    return VariableMetadata(
        name=str(name),
        shape=tuple(int(size) for size in variable.shape),
        dimensions=tuple(str(dimension) for dimension in variable.dims),
        dtype=str(variable.dtype),
        nodata=xarray_nodata_value(variable),
        scale=xarray_numeric_metadata(variable, ("scale_factor",)),
        offset=xarray_numeric_metadata(variable, ("add_offset",)),
        unit=(
            str(variable.attrs["units"])
            if variable.attrs.get("units") is not None
            else None
        ),
        attributes=dict(variable.attrs),
        storage=_xarray_storage_metadata(variable),
    )
