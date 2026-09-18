import warnings

import numpy as np
from numpy.typing import NDArray

from ..windowing import WindowGeometry


def xarray_valid_mask(
    values: np.ma.MaskedArray,
    nodata: int | float | None,
) -> NDArray[np.bool_]:
    """Build an element-wise validity mask for loaded Xarray values."""

    data = np.asarray(np.ma.getdata(values))
    valid_mask = np.asarray(
        ~np.ma.getmaskarray(values),
        dtype=np.bool_,
    )

    if np.issubdtype(data.dtype, np.inexact):
        valid_mask &= np.isfinite(data)

    if nodata is not None:
        if isinstance(nodata, float) and np.isnan(nodata):
            valid_mask &= ~np.isnan(data)
        else:
            valid_mask &= data != nodata

    return valid_mask


def resolve_fill_value(
    dtype: np.dtype,
    requested_fill: int | float | None,
    nodata: int | float | None,
) -> int | float:
    """Resolve an explicit or dtype-appropriate boundless fill value."""

    if requested_fill is not None:
        return requested_fill
    if nodata is not None:
        return nodata
    if np.issubdtype(dtype, np.inexact):
        return np.nan
    return 0


def pad_boundless_result(
    data: NDArray[np.generic],
    valid_mask: NDArray[np.bool_],
    geometry: WindowGeometry,
    output_height: int,
    output_width: int,
    y_axis: int,
    x_axis: int,
    fill_value: int | float,
) -> tuple[NDArray[np.generic], NDArray[np.bool_]]:
    """Place an intersecting read into its requested boundless extent."""

    if data.shape != valid_mask.shape:
        raise ValueError("valid_mask must have the same shape as data.")
    if not 0 <= y_axis < data.ndim or not 0 <= x_axis < data.ndim:
        raise ValueError("Spatial axis positions are outside the data shape.")
    if y_axis == x_axis:
        raise ValueError("X and Y axis positions must be different.")
    if output_height <= 0 or output_width <= 0:
        raise ValueError("Output height and width must be positive.")

    output_shape = list(data.shape)
    output_shape[y_axis] = output_height
    output_shape[x_axis] = output_width
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            output_data = np.full(
                output_shape,
                fill_value,
                dtype=data.dtype,
            )
    except (OverflowError, RuntimeWarning, TypeError, ValueError) as error:
        raise ValueError(
            f"fill_value {fill_value!r} is incompatible with "
            f"variable dtype {data.dtype}."
        ) from error
    output_mask = np.zeros(output_shape, dtype=np.bool_)

    source_window = geometry.source
    if source_window is None:
        return output_data, output_mask
    if (
        data.shape[y_axis] != source_window.height
        or data.shape[x_axis] != source_window.width
    ):
        raise ValueError(
            "Loaded spatial shape does not match the source intersection."
        )

    destination = [slice(None)] * data.ndim
    destination[y_axis] = slice(
        geometry.destination_row_offset,
        geometry.destination_row_offset + source_window.height,
    )
    destination[x_axis] = slice(
        geometry.destination_column_offset,
        geometry.destination_column_offset + source_window.width,
    )
    output_data[tuple(destination)] = data
    output_mask[tuple(destination)] = valid_mask
    return output_data, output_mask
