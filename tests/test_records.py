import unittest
from typing import get_type_hints

import numpy as np

from tiling_template.records import (
    PixelWindow,
    WindowReadRequest,
    WindowReadResult,
)


class TestPixelWindow(unittest.TestCase):
    def test_supports_negative_offsets_for_boundless_reads(self):
        window = PixelWindow(
            row_offset=-2,
            column_offset=-3,
            height=8,
            width=9,
        )

        self.assertEqual(window.row_offset, -2)
        self.assertEqual(window.column_offset, -3)

    def test_rejects_nonpositive_dimensions(self):
        for height, width in ((0, 1), (1, 0), (-1, 1), (1, -1)):
            with self.subTest(height=height, width=width):
                with self.assertRaisesRegex(ValueError, "must be positive"):
                    PixelWindow(0, 0, height, width)


class TestWindowReadRequest(unittest.TestCase):
    def setUp(self):
        self.window = PixelWindow(0, 0, 4, 5)

    def test_accepts_unique_one_based_source_indices(self):
        request = WindowReadRequest(
            window=self.window,
            source_indices=(1, 3),
        )

        self.assertEqual(request.source_indices, (1, 3))

    def test_accepts_no_source_indices(self):
        request = WindowReadRequest(window=self.window)

        self.assertIsNone(request.source_indices)

    def test_accepts_xarray_variable_and_dimension_indices(self):
        request = WindowReadRequest(
            window=self.window,
            variable_name="temperature",
            dimension_indices=(("time", 2), ("level", 0)),
        )

        self.assertEqual(request.variable_name, "temperature")
        self.assertEqual(
            request.dimension_indices,
            (("time", 2), ("level", 0)),
        )

    def test_rejects_empty_source_indices(self):
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            WindowReadRequest(window=self.window, source_indices=())

    def test_rejects_nonpositive_source_indices(self):
        with self.assertRaisesRegex(ValueError, "one-based"):
            WindowReadRequest(window=self.window, source_indices=(0,))

    def test_rejects_duplicate_source_indices(self):
        with self.assertRaisesRegex(ValueError, "must be unique"):
            WindowReadRequest(window=self.window, source_indices=(1, 1))

    def test_rejects_blank_variable_name(self):
        with self.assertRaisesRegex(ValueError, "cannot be blank"):
            WindowReadRequest(window=self.window, variable_name="  ")

    def test_rejects_raster_and_xarray_selectors_together(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            WindowReadRequest(
                window=self.window,
                source_indices=(1,),
                variable_name="temperature",
            )

    def test_rejects_duplicate_dimension_names(self):
        with self.assertRaisesRegex(ValueError, "must be unique"):
            WindowReadRequest(
                window=self.window,
                variable_name="temperature",
                dimension_indices=(("time", 0), ("time", 1)),
            )


class TestWindowReadResult(unittest.TestCase):
    def setUp(self):
        self.request = WindowReadRequest(
            window=PixelWindow(0, 0, 2, 3),
            source_indices=(1,),
        )
        self.data = np.zeros((1, 2, 3), dtype=np.uint8)
        self.transform = (
            1.0,
            0.0,
            0.0,
            0.0,
            -1.0,
            2.0,
            0.0,
            0.0,
            1.0,
        )

    def test_retains_request_and_matching_per_element_mask(self):
        mask = np.ones_like(self.data, dtype=np.bool_)

        result = WindowReadResult(
            data=self.data,
            valid_mask=mask,
            transform=self.transform,
            request=self.request,
        )

        self.assertIs(result.request, self.request)
        self.assertEqual(result.valid_mask.shape, result.data.shape)

    def test_allows_no_validity_mask(self):
        result = WindowReadResult(
            data=self.data,
            valid_mask=None,
            transform=self.transform,
            request=self.request,
        )

        self.assertIsNone(result.valid_mask)

    def test_rejects_mismatched_validity_mask(self):
        with self.assertRaisesRegex(ValueError, "same shape"):
            WindowReadResult(
                data=self.data,
                valid_mask=np.ones((2, 3), dtype=np.bool_),
                transform=self.transform,
                request=self.request,
            )

    def test_type_hints_resolve(self):
        hints = get_type_hints(WindowReadResult)

        self.assertIs(hints["request"], WindowReadRequest)


if __name__ == "__main__":
    unittest.main()
