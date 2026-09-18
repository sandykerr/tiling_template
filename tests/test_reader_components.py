import unittest

import numpy as np
import xarray as xr

from tiling_template.readers.windowing import (
    resolve_window_geometry,
    translate_transform,
)
from tiling_template.readers.xarray.grid import (
    XarraySpatialGrid,
    resolve_xarray_spatial_grid,
)
from tiling_template.readers.xarray.selection import (
    select_xarray_variable,
    validate_xarray_selection,
)
from tiling_template.readers.xarray.metadata import (
    xarray_nodata_value,
    xarray_numeric_metadata,
    xarray_variable_metadata,
)
from tiling_template.readers.xarray.values import (
    pad_boundless_result,
    resolve_fill_value,
    xarray_valid_mask,
)
from tiling_template.records import (
    PixelWindow,
    RasterBandSelection,
    WindowReadRequest,
    XarrayVariableSelection,
)


def make_spatial_dataset() -> xr.Dataset:
    return xr.Dataset(
        data_vars={
            "temperature": (
                ("time", "northing", "easting"),
                np.arange(24).reshape(2, 3, 4),
            ),
            "nonspatial": ("time", np.arange(2)),
        },
        coords={
            "time": [0, 1],
            "easting": (
                "easting",
                [11.0, 13.0, 15.0, 17.0],
                {"axis": "X"},
            ),
            "northing": (
                "northing",
                [25.0, 22.0, 19.0],
                {"standard_name": "projection_y_coordinate"},
            ),
        },
    )


class TestWindowGeometry(unittest.TestCase):
    def test_resolves_fully_contained_half_open_window(self):
        requested = PixelWindow(2, 3, 4, 5)

        geometry = resolve_window_geometry(requested, 10, 12)

        self.assertIs(geometry.requested, requested)
        self.assertEqual(geometry.source, requested)
        self.assertEqual(geometry.destination_row_offset, 0)
        self.assertEqual(geometry.destination_column_offset, 0)
        self.assertFalse(geometry.extends_beyond_source)

    def test_exact_source_bounds_are_contained(self):
        geometry = resolve_window_geometry(
            PixelWindow(0, 0, 10, 12),
            10,
            12,
        )

        self.assertFalse(geometry.extends_beyond_source)

    def test_resolves_each_partial_edge(self):
        cases = (
            (
                PixelWindow(-2, 3, 5, 4),
                PixelWindow(0, 3, 3, 4),
                (2, 0),
            ),
            (
                PixelWindow(8, 3, 5, 4),
                PixelWindow(8, 3, 2, 4),
                (0, 0),
            ),
            (
                PixelWindow(2, -2, 3, 5),
                PixelWindow(2, 0, 3, 3),
                (0, 2),
            ),
            (
                PixelWindow(2, 10, 3, 5),
                PixelWindow(2, 10, 3, 2),
                (0, 0),
            ),
        )

        for requested, expected_source, expected_destination in cases:
            with self.subTest(requested=requested):
                geometry = resolve_window_geometry(requested, 10, 12)

                self.assertEqual(geometry.source, expected_source)
                self.assertEqual(
                    (
                        geometry.destination_row_offset,
                        geometry.destination_column_offset,
                    ),
                    expected_destination,
                )
                self.assertTrue(geometry.extends_beyond_source)

    def test_resolves_negative_corner_offsets(self):
        geometry = resolve_window_geometry(
            PixelWindow(-2, -3, 5, 7),
            10,
            12,
        )

        self.assertEqual(geometry.source, PixelWindow(0, 0, 3, 4))
        self.assertEqual(geometry.destination_row_offset, 2)
        self.assertEqual(geometry.destination_column_offset, 3)

    def test_fully_outside_window_has_no_source_intersection(self):
        geometry = resolve_window_geometry(
            PixelWindow(10, 12, 2, 2),
            10,
            12,
        )

        self.assertIsNone(geometry.source)
        self.assertTrue(geometry.extends_beyond_source)

    def test_rejects_negative_source_dimensions(self):
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            resolve_window_geometry(PixelWindow(0, 0, 1, 1), -1, 1)


class TestTransformTranslation(unittest.TestCase):
    def setUp(self):
        self.transform = (
            2.0,
            0.5,
            10.0,
            -0.25,
            -3.0,
            20.0,
            0.0,
            0.0,
            1.0,
        )

    def test_zero_offsets_preserve_transform(self):
        translated = translate_transform(
            self.transform,
            PixelWindow(0, 0, 1, 1),
        )

        self.assertEqual(translated, self.transform)

    def test_positive_offsets_translate_origin(self):
        translated = translate_transform(
            self.transform,
            PixelWindow(2, 3, 1, 1),
        )

        self.assertEqual(
            translated,
            (2.0, 0.5, 17.0, -0.25, -3.0, 13.25, 0.0, 0.0, 1.0),
        )

    def test_negative_offsets_translate_origin(self):
        translated = translate_transform(
            self.transform,
            PixelWindow(-2, -3, 1, 1),
        )

        self.assertEqual(
            translated,
            (2.0, 0.5, 3.0, -0.25, -3.0, 26.75, 0.0, 0.0, 1.0),
        )


class TestXarraySpatialGrid(unittest.TestCase):
    def test_resolves_axis_and_standard_name_coordinates(self):
        grid = resolve_xarray_spatial_grid(make_spatial_dataset())

        self.assertEqual(
            grid,
            XarraySpatialGrid(
                x_dimension="easting",
                y_dimension="northing",
                width=4,
                height=3,
                transform=(
                    2.0,
                    0.0,
                    10.0,
                    0.0,
                    -3.0,
                    26.5,
                    0.0,
                    0.0,
                    1.0,
                ),
                bounds=(10.0, 17.5, 18.0, 26.5),
                resolution=(2.0, 3.0),
            ),
        )

    def test_prefers_conventional_coordinate_names(self):
        dataset = make_spatial_dataset().rename(
            {"easting": "x", "northing": "y"}
        )

        grid = resolve_xarray_spatial_grid(dataset)

        self.assertEqual(grid.x_dimension, "x")
        self.assertEqual(grid.y_dimension, "y")

    def test_returns_none_for_missing_or_irregular_grid(self):
        missing = xr.Dataset(
            {"value": (("row", "column"), np.ones((2, 2)))}
        )
        irregular = make_spatial_dataset().assign_coords(
            easting=("easting", [0.0, 1.0, 3.0, 4.0])
        )

        self.assertIsNone(resolve_xarray_spatial_grid(missing))
        self.assertIsNone(resolve_xarray_spatial_grid(irregular))

    def test_returns_none_for_nonfinite_coordinates(self):
        dataset = make_spatial_dataset().assign_coords(
            easting=("easting", [0.0, 1.0, np.nan, 3.0])
        )

        self.assertIsNone(resolve_xarray_spatial_grid(dataset))


class TestXarraySelection(unittest.TestCase):
    def setUp(self):
        self.dataset = make_spatial_dataset()
        self.grid = resolve_xarray_spatial_grid(self.dataset)
        self.assertIsNotNone(self.grid)

    def request(self, **kwargs) -> WindowReadRequest:
        return WindowReadRequest(
            window=PixelWindow(0, 0, 1, 1),
            selection=XarrayVariableSelection(
                variable_name="temperature",
                **kwargs,
            ),
        )

    def test_validates_without_loading_and_selects_separately(self):
        request = self.request(dimension_indices=(("time", -1),))

        selection = validate_xarray_selection(
            self.dataset,
            request,
            self.grid,
        )
        selected = select_xarray_variable(selection)

        self.assertTrue(
            selection.variable.identical(self.dataset["temperature"])
        )
        self.assertEqual(selection.dimension_indices, (("time", -1),))
        self.assertEqual(selected.dims, ("northing", "easting"))
        np.testing.assert_array_equal(
            selected.values,
            self.dataset["temperature"].values[-1],
        )

    def test_rejects_raster_selection(self):
        request = WindowReadRequest(
            window=PixelWindow(0, 0, 1, 1),
            selection=RasterBandSelection(source_indices=(1,)),
        )

        with self.assertRaisesRegex(ValueError, "XarrayVariableSelection"):
            validate_xarray_selection(self.dataset, request, self.grid)

    def test_rejects_unknown_variable(self):
        unknown = WindowReadRequest(
            window=PixelWindow(0, 0, 1, 1),
            selection=XarrayVariableSelection("missing"),
        )

        with self.assertRaisesRegex(ValueError, "not present"):
            validate_xarray_selection(self.dataset, unknown, self.grid)

    def test_rejects_variable_without_spatial_dimensions(self):
        request = WindowReadRequest(
            window=PixelWindow(0, 0, 1, 1),
            selection=XarrayVariableSelection("nonspatial"),
        )

        with self.assertRaisesRegex(ValueError, "both spatial"):
            validate_xarray_selection(self.dataset, request, self.grid)

    def test_rejects_spatial_and_unknown_dimension_indices(self):
        spatial = self.request(dimension_indices=(("easting", 0),))
        unknown = self.request(dimension_indices=(("level", 0),))

        with self.assertRaisesRegex(ValueError, "PixelWindow"):
            validate_xarray_selection(self.dataset, spatial, self.grid)
        with self.assertRaisesRegex(ValueError, "not present"):
            validate_xarray_selection(self.dataset, unknown, self.grid)

    def test_rejects_positive_and_negative_out_of_range_indices(self):
        for index in (2, -3):
            with self.subTest(index=index):
                request = self.request(
                    dimension_indices=(("time", index),)
                )
                with self.assertRaisesRegex(ValueError, "outside"):
                    validate_xarray_selection(
                        self.dataset,
                        request,
                        self.grid,
                    )


class TestXarrayMetadataNormalization(unittest.TestCase):
    def test_numeric_metadata_uses_attributes_before_encoding(self):
        variable = xr.DataArray(
            np.ones((2, 2), dtype=np.int16),
            dims=("y", "x"),
            attrs={"missing_value": -2, "scale_factor": 0.5},
        )
        variable.encoding["_FillValue"] = -1
        variable.encoding["scale_factor"] = 2.0

        self.assertEqual(xarray_nodata_value(variable), -2)
        self.assertEqual(
            xarray_numeric_metadata(variable, ("scale_factor",)),
            0.5,
        )

    def test_numeric_metadata_ignores_boolean_and_nonscalar_values(self):
        variable = xr.DataArray(
            np.ones((2, 2)),
            attrs={
                "scale_factor": True,
                "add_offset": [1, 2],
            },
        )

        self.assertIsNone(
            xarray_numeric_metadata(variable, ("scale_factor",))
        )
        self.assertIsNone(
            xarray_numeric_metadata(variable, ("add_offset",))
        )

    def test_constructs_complete_variable_record(self):
        variable = xr.DataArray(
            np.ones((2, 3), dtype=np.float32),
            dims=("y", "x"),
            attrs={
                "_FillValue": -9999.0,
                "scale_factor": 0.1,
                "add_offset": 1.0,
                "units": "reflectance",
                "description": "example",
            },
        )
        variable.encoding.update(
            {
                "chunksizes": (1, 3),
                "zlib": True,
                "complevel": 4,
                "shuffle": True,
            }
        )

        metadata = xarray_variable_metadata("band", variable)

        self.assertEqual(metadata.name, "band")
        self.assertEqual(metadata.shape, (2, 3))
        self.assertEqual(metadata.dimensions, ("y", "x"))
        self.assertEqual(metadata.dtype, "float32")
        self.assertEqual(metadata.nodata, -9999.0)
        self.assertEqual(metadata.scale, 0.1)
        self.assertEqual(metadata.offset, 1.0)
        self.assertEqual(metadata.unit, "reflectance")
        self.assertEqual(metadata.attributes["description"], "example")
        self.assertEqual(metadata.storage.chunk_shape, (1, 3))
        self.assertEqual(metadata.storage.compression, "zlib")
        self.assertEqual(metadata.storage.compression_level, 4)
        self.assertTrue(metadata.storage.shuffle)


class TestXarrayValues(unittest.TestCase):
    def test_valid_mask_accepts_unmasked_numeric_values(self):
        values = np.ma.asarray(np.array([[1, 2], [3, 4]], dtype=np.int16))

        mask = xarray_valid_mask(values, nodata=None)

        np.testing.assert_array_equal(mask, np.ones((2, 2), dtype=np.bool_))

    def test_valid_mask_combines_existing_mask_and_integer_nodata(self):
        values = np.ma.array(
            [[1, -1], [3, 4]],
            mask=[[False, False], [True, False]],
        )

        mask = xarray_valid_mask(values, nodata=-1)

        np.testing.assert_array_equal(
            mask,
            np.array([[True, False], [False, True]]),
        )

    def test_valid_mask_rejects_float_nodata_and_nonfinite_values(self):
        values = np.ma.asarray(
            np.array([[1.0, -9999.0], [np.nan, np.inf]])
        )

        mask = xarray_valid_mask(values, nodata=-9999.0)

        np.testing.assert_array_equal(
            mask,
            np.array([[True, False], [False, False]]),
        )

    def test_valid_mask_supports_nan_nodata(self):
        values = np.ma.asarray(np.array([[1.0, np.nan]]))

        mask = xarray_valid_mask(values, nodata=np.nan)

        np.testing.assert_array_equal(mask, np.array([[True, False]]))

    def test_fill_resolution_precedence_and_dtype_defaults(self):
        self.assertEqual(resolve_fill_value(np.dtype("int16"), 7, -1), 7)
        self.assertEqual(resolve_fill_value(np.dtype("int16"), None, -1), -1)
        self.assertEqual(resolve_fill_value(np.dtype("uint8"), None, None), 0)
        self.assertTrue(
            np.isnan(resolve_fill_value(np.dtype("float32"), None, None))
        )

    def test_padding_places_each_edge_and_corner_intersection(self):
        source = np.arange(20, dtype=np.int16).reshape(4, 5)
        requests = (
            PixelWindow(-1, 1, 3, 3),
            PixelWindow(3, 1, 3, 3),
            PixelWindow(1, -1, 2, 3),
            PixelWindow(1, 4, 2, 3),
            PixelWindow(-1, -1, 3, 3),
            PixelWindow(-1, 4, 3, 3),
            PixelWindow(3, -1, 3, 3),
            PixelWindow(3, 4, 3, 3),
        )

        for requested in requests:
            with self.subTest(requested=requested):
                geometry = resolve_window_geometry(requested, 4, 5)
                source_window = geometry.source
                data = source[
                    source_window.row_offset:
                    source_window.row_offset + source_window.height,
                    source_window.column_offset:
                    source_window.column_offset + source_window.width,
                ]
                valid_mask = np.ones_like(data, dtype=np.bool_)

                padded, padded_mask = pad_boundless_result(
                    data,
                    valid_mask,
                    geometry,
                    output_height=requested.height,
                    output_width=requested.width,
                    y_axis=0,
                    x_axis=1,
                    fill_value=-1,
                )

                expected = np.full(
                    (requested.height, requested.width),
                    -1,
                    dtype=np.int16,
                )
                expected_mask = np.zeros_like(expected, dtype=np.bool_)
                row_start = geometry.destination_row_offset
                column_start = geometry.destination_column_offset
                expected[
                    row_start:row_start + source_window.height,
                    column_start:column_start + source_window.width,
                ] = data
                expected_mask[
                    row_start:row_start + source_window.height,
                    column_start:column_start + source_window.width,
                ] = True

                np.testing.assert_array_equal(padded, expected)
                np.testing.assert_array_equal(padded_mask, expected_mask)

    def test_padding_preserves_nonspatial_dimensions_and_axis_order(self):
        geometry = resolve_window_geometry(
            PixelWindow(-1, -1, 3, 4),
            4,
            5,
        )
        data = np.arange(2 * 2 * 3 * 3, dtype=np.int16).reshape(2, 2, 3, 3)
        valid_mask = np.ones_like(data, dtype=np.bool_)

        padded, padded_mask = pad_boundless_result(
            data,
            valid_mask,
            geometry,
            output_height=3,
            output_width=4,
            y_axis=1,
            x_axis=3,
            fill_value=-1,
        )

        self.assertEqual(padded.shape, (2, 3, 3, 4))
        np.testing.assert_array_equal(padded[:, 1:, :, 1:], data)
        self.assertTrue(np.all(padded_mask[:, 1:, :, 1:]))
        self.assertFalse(np.any(padded_mask[:, 0, :, :]))
        self.assertFalse(np.any(padded_mask[:, :, :, 0]))

    def test_fully_outside_padding_is_entirely_invalid(self):
        geometry = resolve_window_geometry(PixelWindow(5, 6, 2, 3), 4, 5)

        padded, padded_mask = pad_boundless_result(
            np.empty((0, 0), dtype=np.uint8),
            np.empty((0, 0), dtype=np.bool_),
            geometry,
            output_height=2,
            output_width=3,
            y_axis=0,
            x_axis=1,
            fill_value=9,
        )

        np.testing.assert_array_equal(
            padded,
            np.full((2, 3), 9, dtype=np.uint8),
        )
        self.assertFalse(np.any(padded_mask))

    def test_padding_rejects_incompatible_fill_value(self):
        geometry = resolve_window_geometry(PixelWindow(-1, 0, 2, 1), 2, 2)

        with self.assertRaisesRegex(ValueError, "incompatible"):
            pad_boundless_result(
                np.ones((1, 1), dtype=np.int16),
                np.ones((1, 1), dtype=np.bool_),
                geometry,
                output_height=2,
                output_width=1,
                y_axis=0,
                x_axis=1,
                fill_value=np.nan,
            )


if __name__ == "__main__":
    unittest.main()
