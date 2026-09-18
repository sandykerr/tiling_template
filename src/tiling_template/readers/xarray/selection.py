from dataclasses import dataclass

import xarray as xr

from ...records import WindowReadRequest, XarrayVariableSelection
from .grid import XarraySpatialGrid


@dataclass(frozen=True, slots=True)
class ValidatedXarraySelection:
    """Contain an Xarray variable and validated non-spatial selectors."""

    variable: xr.DataArray
    dimension_indices: tuple[tuple[str, int], ...]


def validate_xarray_selection(
    dataset: xr.Dataset,
    request: WindowReadRequest,
    grid: XarraySpatialGrid,
) -> ValidatedXarraySelection:
    """Validate an Xarray variable selection without loading its values."""

    if not isinstance(request.selection, XarrayVariableSelection):
        raise ValueError(
            "Xarray window reads require XarrayVariableSelection, not "
            "RasterBandSelection."
        )
    request_selection = request.selection
    if request_selection.variable_name not in dataset.data_vars:
        raise ValueError(
            "Variable is not present in the dataset: "
            f"{request_selection.variable_name}"
        )

    variable = dataset[request_selection.variable_name]
    spatial_dimensions = {grid.x_dimension, grid.y_dimension}
    if not spatial_dimensions.issubset(variable.dims):
        raise ValueError(
            f"Variable {request_selection.variable_name!r} does not contain "
            "both spatial dimensions."
        )

    indexers = dict(request_selection.dimension_indices)
    spatial_selectors = spatial_dimensions & indexers.keys()
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

    for dimension, index in request_selection.dimension_indices:
        dimension_size = variable.sizes[dimension]
        if not -dimension_size <= index < dimension_size:
            raise ValueError(
                f"Index {index} is outside dimension {dimension!r} "
                f"with size {dimension_size}."
            )

    return ValidatedXarraySelection(
        variable=variable,
        dimension_indices=request_selection.dimension_indices,
    )


def select_xarray_variable(
    selection: ValidatedXarraySelection,
) -> xr.DataArray:
    """Apply a previously validated non-spatial variable selection."""

    return selection.variable.isel(dict(selection.dimension_indices))
