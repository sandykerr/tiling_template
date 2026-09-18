from dataclasses import dataclass

import numpy as np
import xarray as xr


@dataclass(frozen=True, slots=True)
class XarraySpatialGrid:
    """Describe a regular pixel grid derived from Xarray coordinates."""

    x_dimension: str
    y_dimension: str
    width: int
    height: int
    transform: tuple[float, ...]
    bounds: tuple[float, ...]
    resolution: tuple[float, ...]


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


def resolve_xarray_spatial_grid(
    dataset: xr.Dataset,
) -> XarraySpatialGrid | None:
    """Resolve a regular, pixel-centered Xarray spatial grid if present."""

    x_coordinate = _coordinate_for_axis(dataset, "X")
    y_coordinate = _coordinate_for_axis(dataset, "Y")
    if x_coordinate is None or y_coordinate is None:
        return None

    x_dimension = str(x_coordinate.dims[0])
    y_dimension = str(y_coordinate.dims[0])
    if x_dimension == y_dimension or not any(
        x_dimension in variable.dims and y_dimension in variable.dims
        for variable in dataset.data_vars.values()
    ):
        return None

    try:
        x_values = np.asarray(x_coordinate.values, dtype=float)
        y_values = np.asarray(y_coordinate.values, dtype=float)
    except (TypeError, ValueError):
        return None

    if x_values.size < 2 or y_values.size < 2:
        return None

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
        return None

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

    return XarraySpatialGrid(
        x_dimension=x_dimension,
        y_dimension=y_dimension,
        width=int(x_values.size),
        height=int(y_values.size),
        transform=(
            x_step,
            0.0,
            x_edges[0],
            0.0,
            y_step,
            y_edges[0],
            0.0,
            0.0,
            1.0,
        ),
        bounds=(
            min(x_edges),
            min(y_edges),
            max(x_edges),
            max(y_edges),
        ),
        resolution=(abs(x_step), abs(y_step)),
    )
