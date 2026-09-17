import tempfile
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.transform import from_origin
import xarray as xr

from tiling_template.configs.reader import (
    RasterioBackendConfig,
    XarrayBackendConfig,
)
from tiling_template.readers import (
    AssetReadSession,
    AssetReaderBackend,
    MetadataReader,
    ReaderRegistry,
    WindowReader,
    default_reader_registry,
)
from tiling_template.readers.rasterio_reader import (
    RasterioBackend,
    RasterioMetadataReader,
    RasterioWindowReader,
)
from tiling_template.readers.xarray_reader import (
    XarrayBackend,
    XarrayMetadataReader,
    XarrayWindowReader,
)
from tiling_template.records import (
    AssetMetadata,
    AssetRef,
    PixelWindow,
    VariableMetadata,
    WindowReadRequest,
    WindowReadResult,
)


def make_asset(filename: str) -> AssetRef:
    path = Path("/dataset") / filename
    return AssetRef(
        path=path,
        relative_path=Path(filename),
        spec_id="test-assets",
        role="input",
        modality="test",
        size_bytes=1,
        modified_time_ns=1,
    )


def make_window_request() -> WindowReadRequest:
    return WindowReadRequest(
        window=PixelWindow(
            row_offset=0,
            column_offset=0,
            height=1,
            width=1,
        ),
        source_indices=(1,),
    )


class FakeMetadataReader(MetadataReader):
    def __init__(self, asset: AssetRef):
        self.asset = asset

    def read_metadata(self) -> AssetMetadata:
        return AssetMetadata(
            asset=self.asset,
            variables=(
                VariableMetadata(
                    name="band_1",
                    shape=(1, 1),
                    dimensions=("y", "x"),
                    dtype="uint8",
                    source_index=1,
                ),
            ),
        )


class FakeWindowReader(WindowReader):
    def __init__(self, asset: AssetRef):
        self.asset = asset

    def read_window(self, request: WindowReadRequest) -> WindowReadResult:
        data = np.zeros((1, 1, 1), dtype=np.uint8)
        return WindowReadResult(
            data=data,
            valid_mask=np.ones_like(data, dtype=np.bool_),
            transform=(1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0),
            request=request,
        )


class AlternateMetadataReader(FakeMetadataReader):
    pass


class FakeBackend(AssetReaderBackend):
    metadata_reader_class = FakeMetadataReader

    @contextmanager
    def open(self, asset: AssetRef) -> Iterator[AssetReadSession]:
        yield AssetReadSession(
            metadata_reader=self.metadata_reader_class(asset),
            window_reader=FakeWindowReader(asset),
        )


class AlternateBackend(FakeBackend):
    metadata_reader_class = AlternateMetadataReader


class TestReaderInterfaces(unittest.TestCase):
    def test_abstract_metadata_reader_cannot_be_instantiated(self):
        with self.assertRaises(TypeError):
            MetadataReader()

    def test_abstract_window_reader_cannot_be_instantiated(self):
        with self.assertRaises(TypeError):
            WindowReader()

    def test_abstract_backend_cannot_be_instantiated(self):
        with self.assertRaises(TypeError):
            AssetReaderBackend()

    def test_reader_returns_current_metadata_record_shape(self):
        asset = make_asset("scene.tif")

        with FakeBackend().open(asset) as session:
            metadata = session.metadata_reader.read_metadata()

        self.assertIs(metadata.asset, asset)
        self.assertEqual(metadata.variables[0].name, "band_1")
        self.assertEqual(metadata.variables[0].shape, (1, 1))


class TestReaderRegistry(unittest.TestCase):
    def test_constructs_reader_for_each_registered_extension(self):
        registry = ReaderRegistry()
        registry.register((".tif", ".tiff"), FakeBackend)

        self.assertIsInstance(
            registry.backend_for(make_asset("scene.tif")),
            FakeBackend,
        )
        self.assertIsInstance(
            registry.backend_for(make_asset("scene.tiff")),
            FakeBackend,
        )

    def test_normalizes_case_whitespace_and_leading_period(self):
        registry = ReaderRegistry()
        registry.register((" TIF ",), FakeBackend)

        backend = registry.backend_for(make_asset("scene.TIF"))

        self.assertIsInstance(backend, FakeBackend)

    def test_creates_a_new_reader_for_each_lookup(self):
        registry = ReaderRegistry()
        registry.register((".tif",), FakeBackend)
        asset = make_asset("scene.tif")

        first = registry.backend_for(asset)
        second = registry.backend_for(asset)

        self.assertIsNot(first, second)

    def test_session_readers_are_bound_to_the_requested_asset(self):
        asset = make_asset("scene.tif")

        with FakeBackend().open(asset) as session:
            self.assertIs(session.metadata_reader.asset, asset)
            self.assertIs(session.window_reader.asset, asset)

    def test_rejects_empty_extension_collection(self):
        registry = ReaderRegistry()

        with self.assertRaisesRegex(ValueError, "At least one"):
            registry.register((), FakeBackend)

    def test_rejects_blank_extension(self):
        registry = ReaderRegistry()

        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            registry.register(("  ",), FakeBackend)

    def test_rejects_duplicate_normalized_extensions(self):
        registry = ReaderRegistry()

        with self.assertRaisesRegex(ValueError, "must be unique"):
            registry.register(("tif", ".TIF"), FakeBackend)

    def test_rejects_an_extension_registered_by_another_reader(self):
        registry = ReaderRegistry()
        registry.register((".tif",), FakeBackend)

        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register((".TIF",), AlternateBackend)

    def test_conflicting_registration_is_atomic(self):
        registry = ReaderRegistry()
        registry.register((".tif",), FakeBackend)

        with self.assertRaises(ValueError):
            registry.register((".nc", ".tif"), AlternateBackend)

        self.assertIsInstance(
            registry.backend_for(make_asset("scene.tif")),
            FakeBackend,
        )
        with self.assertRaisesRegex(ValueError, "No backend is registered"):
            registry.backend_for(make_asset("scene.nc"))

    def test_rejects_unregistered_extension(self):
        registry = ReaderRegistry()

        with self.assertRaisesRegex(ValueError, r"\.nc"):
            registry.backend_for(make_asset("scene.nc"))

    def test_rejects_asset_without_extension(self):
        registry = ReaderRegistry()

        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            registry.backend_for(make_asset("README"))


class TestDefaultReaderRegistry(unittest.TestCase):
    def test_registers_rasterio_for_geotiff_extensions(self):
        registry = default_reader_registry()

        self.assertIsInstance(
            registry.backend_for(make_asset("scene.tif")),
            RasterioBackend,
        )
        self.assertIsInstance(
            registry.backend_for(make_asset("scene.tiff")),
            RasterioBackend,
        )

    def test_registers_xarray_for_netcdf_extension(self):
        registry = default_reader_registry()

        self.assertIsInstance(
            registry.backend_for(make_asset("scene.nc")),
            XarrayBackend,
        )


class TestBackendConfigs(unittest.TestCase):
    def test_rasterio_config_is_frozen_and_validates_options(self):
        config = RasterioBackendConfig(
            sharing=True,
            gdal_options=(("GDAL_CACHEMAX", 64_000_000),),
        )

        self.assertEqual(
            config.options_dict(),
            {"GDAL_CACHEMAX": 64_000_000},
        )
        with self.assertRaises(FrozenInstanceError):
            config.sharing = False

        with self.assertRaisesRegex(ValueError, "must be unique"):
            RasterioBackendConfig(
                gdal_options=(("OPTION", 1), ("OPTION", 2)),
            )

    def test_xarray_config_is_frozen(self):
        config = XarrayBackendConfig(
            engine="netcdf4",
            group="observations",
            plugin_modules=("hdf5plugin",),
        )

        with self.assertRaises(FrozenInstanceError):
            config.engine = None

        with self.assertRaisesRegex(ValueError, "blank or padded"):
            XarrayBackendConfig(plugin_modules=("  ",))
        with self.assertRaisesRegex(ValueError, "must be unique"):
            XarrayBackendConfig(
                plugin_modules=("hdf5plugin", "hdf5plugin"),
            )


class TestRasterioReaders(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary_directory.name) / "scene.tif"
        self.transform = from_origin(10, 20, 2, 2)

        with rasterio.open(
            self.path,
            "w",
            driver="GTiff",
            width=3,
            height=2,
            count=2,
            dtype="uint16",
            crs="EPSG:4326",
            transform=self.transform,
            nodata=999,
            tiled=True,
            blockxsize=16,
            blockysize=16,
            compress="deflate",
        ) as dataset:
            values = np.arange(12, dtype=np.uint16).reshape(2, 2, 3)
            values[0, 1, 1] = 999
            dataset.write(values)
            dataset.set_band_description(1, "red")
            dataset.scales = (0.1, 1.0)
            dataset.offsets = (1.0, 0.0)
            dataset.units = ("reflectance", "class")
            dataset.update_tags(dataset_tag="asset-value")
            dataset.update_tags(1, band_tag="band-value")
            dataset.build_overviews([2], Resampling.nearest)

        file_stat = self.path.stat()
        self.asset = AssetRef(
            path=self.path,
            relative_path=Path("scene.tif"),
            spec_id="test-assets",
            role="input",
            modality="optical",
            size_bytes=file_stat.st_size,
            modified_time_ns=file_stat.st_mtime_ns,
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_reads_asset_and_per_band_metadata(self):
        backend = RasterioBackend()

        with backend.open(self.asset) as session:
            metadata_reader = session.metadata_reader
            window_reader = session.window_reader
            self.assertIsInstance(metadata_reader, RasterioMetadataReader)
            self.assertIsInstance(window_reader, RasterioWindowReader)
            self.assertIs(metadata_reader.dataset, window_reader.dataset)
            dataset = metadata_reader.dataset
            self.assertFalse(dataset.closed)
            metadata = metadata_reader.read_metadata()

        self.assertTrue(dataset.closed)

        self.assertIs(metadata.asset, self.asset)
        self.assertEqual(metadata.crs, "EPSG:4326")
        self.assertEqual(metadata.transform, tuple(self.transform))
        self.assertEqual(metadata.bounds, (10.0, 16.0, 16.0, 20.0))
        self.assertEqual(metadata.resolution, (2.0, 2.0))
        self.assertTrue(metadata.is_tiled)
        self.assertEqual(metadata.attributes["dataset_tag"], "asset-value")
        self.assertEqual(len(metadata.variables), 2)

        first, second = metadata.variables
        self.assertEqual(first.name, "red")
        self.assertEqual(first.source_index, 1)
        self.assertEqual(first.shape, (2, 3))
        self.assertEqual(first.dimensions, ("y", "x"))
        self.assertEqual(first.dtype, "uint16")
        self.assertEqual(first.nodata, 999)
        self.assertEqual(first.scale, 0.1)
        self.assertEqual(first.offset, 1.0)
        self.assertEqual(first.unit, "reflectance")
        self.assertEqual(first.attributes["band_tag"], "band-value")
        self.assertEqual(first.storage.chunk_shape, (16, 16))
        self.assertEqual(first.storage.compression, "deflate")
        self.assertEqual(first.storage.overview_factors, (2,))

        self.assertEqual(second.name, "band_2")
        self.assertEqual(second.source_index, 2)

    def test_reads_selected_bands_in_requested_order(self):
        request = WindowReadRequest(
            window=PixelWindow(0, 1, 2, 2),
            source_indices=(2, 1),
        )

        with RasterioBackend().open(self.asset) as session:
            result = session.window_reader.read_window(request)

        np.testing.assert_array_equal(
            result.data,
            np.array(
                [
                    [[7, 8], [10, 11]],
                    [[1, 2], [999, 5]],
                ],
                dtype=np.uint16,
            ),
        )
        np.testing.assert_array_equal(
            result.valid_mask,
            np.array(
                [
                    [[True, True], [True, True]],
                    [[True, True], [False, True]],
                ]
            ),
        )
        self.assertEqual(
            result.transform,
            tuple(from_origin(12, 20, 2, 2)),
        )
        self.assertIs(result.request, request)

    def test_defaults_to_all_bands(self):
        request = WindowReadRequest(window=PixelWindow(0, 0, 1, 1))

        with RasterioBackend().open(self.asset) as session:
            result = session.window_reader.read_window(request)

        np.testing.assert_array_equal(
            result.data,
            np.array([[[0]], [[6]]], dtype=np.uint16),
        )
        self.assertEqual(result.data.shape, (2, 1, 1))

    def test_boundless_read_pads_data_and_marks_padding_invalid(self):
        request = WindowReadRequest(
            window=PixelWindow(-1, -1, 3, 3),
            source_indices=(1,),
            boundless=True,
            fill_value=77,
        )

        with RasterioBackend().open(self.asset) as session:
            result = session.window_reader.read_window(request)

        np.testing.assert_array_equal(
            result.data,
            np.array(
                [
                    [
                        [77, 77, 77],
                        [77, 0, 1],
                        [77, 3, 999],
                    ]
                ],
                dtype=np.uint16,
            ),
        )
        np.testing.assert_array_equal(
            result.valid_mask,
            np.array(
                [
                    [
                        [False, False, False],
                        [False, True, True],
                        [False, True, False],
                    ]
                ]
            ),
        )
        self.assertEqual(
            result.transform,
            tuple(from_origin(8, 22, 2, 2)),
        )

    def test_rejects_out_of_bounds_window_without_boundless_reading(self):
        request = WindowReadRequest(
            window=PixelWindow(-1, 0, 2, 2),
        )

        with RasterioBackend().open(self.asset) as session:
            with self.assertRaisesRegex(ValueError, "boundless=True"):
                session.window_reader.read_window(request)

    def test_rejects_source_index_missing_from_dataset(self):
        request = WindowReadRequest(
            window=PixelWindow(0, 0, 1, 1),
            source_indices=(3,),
        )

        with RasterioBackend().open(self.asset) as session:
            with self.assertRaisesRegex(ValueError, r"\[3\]"):
                session.window_reader.read_window(request)

    def test_rejects_xarray_selectors(self):
        request = WindowReadRequest(
            window=PixelWindow(0, 0, 1, 1),
            variable_name="temperature",
        )

        with RasterioBackend().open(self.asset) as session:
            with self.assertRaisesRegex(ValueError, "Xarray"):
                session.window_reader.read_window(request)

    def test_backend_applies_config_and_reuses_one_open_handle(self):
        config = RasterioBackendConfig(
            sharing=True,
            gdal_options=(("GDAL_CACHEMAX", 64_000_000),),
        )
        backend = RasterioBackend(config)

        with patch(
            "tiling_template.readers.rasterio_reader.rasterio.open",
            wraps=rasterio.open,
        ) as open_mock:
            with backend.open(self.asset) as session:
                self.assertEqual(
                    rasterio.env.getenv()["GDAL_CACHEMAX"],
                    64_000_000,
                )
                first = session.metadata_reader.read_metadata()
                second = session.metadata_reader.read_metadata()
                first_window = session.window_reader.read_window(
                    make_window_request()
                )
                second_window = session.window_reader.read_window(
                    make_window_request()
                )

        open_mock.assert_called_once_with(self.asset.path, sharing=True)
        self.assertEqual(first, second)
        np.testing.assert_array_equal(first_window.data, second_window.data)

    def test_session_closes_dataset_after_exception(self):
        dataset = None

        with self.assertRaisesRegex(RuntimeError, "worker failure"):
            with RasterioBackend().open(self.asset) as session:
                dataset = session.metadata_reader.dataset
                raise RuntimeError("worker failure")

        self.assertIsNotNone(dataset)
        self.assertTrue(dataset.closed)

    def test_window_reader_cannot_read_after_session_closes(self):
        with RasterioBackend().open(self.asset) as session:
            window_reader = session.window_reader

        with self.assertRaisesRegex(RasterioIOError, "closed"):
            window_reader.read_window(make_window_request())


class TestXarrayReaders(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary_directory.name) / "scene.nc"
        temperature_values = np.arange(
            12,
            dtype=np.float32,
        ).reshape(2, 2, 3)
        temperature_values[0, 1, 1] = -9999.0
        dataset = xr.Dataset(
            data_vars={
                "temperature": (
                    ("time", "y", "x"),
                    temperature_values,
                    {
                        "units": "K",
                        "scale_factor": 0.1,
                        "add_offset": 273.15,
                        "grid_mapping": "spatial_ref",
                        "_FillValue": -9999.0,
                    },
                ),
                "quality": (
                    ("y", "x"),
                    np.ones((2, 3), dtype=np.uint8),
                    {"long_name": "quality flag"},
                ),
            },
            coords={
                "time": ("time", np.array([0, 1], dtype=np.int32)),
                "x": (
                    "x",
                    np.array([11.0, 13.0, 15.0]),
                    {"axis": "X"},
                ),
                "y": (
                    "y",
                    np.array([19.0, 17.0]),
                    {"axis": "Y"},
                ),
                "spatial_ref": (
                    (),
                    0,
                    {"spatial_ref": "EPSG:4326"},
                ),
            },
            attrs={"title": "example dataset"},
        )
        dataset.to_netcdf(
            self.path,
            engine="netcdf4",
            encoding={
                "temperature": {
                    "zlib": True,
                    "complevel": 4,
                    "shuffle": True,
                    "chunksizes": (1, 2, 3),
                },
            },
        )
        dataset.close()

        file_stat = self.path.stat()
        self.asset = AssetRef(
            path=self.path,
            relative_path=Path("scene.nc"),
            spec_id="test-assets",
            role="input",
            modality="climate",
            size_bytes=file_stat.st_size,
            modified_time_ns=file_stat.st_mtime_ns,
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_reads_netcdf_asset_and_variable_metadata(self):
        with XarrayBackend().open(self.asset) as session:
            self.assertIsInstance(
                session.metadata_reader,
                XarrayMetadataReader,
            )
            self.assertIsInstance(session.window_reader, XarrayWindowReader)
            self.assertIs(
                session.metadata_reader.dataset,
                session.window_reader.dataset,
            )
            metadata = session.metadata_reader.read_metadata()

        self.assertIs(metadata.asset, self.asset)
        self.assertEqual(metadata.crs, "EPSG:4326")
        self.assertEqual(
            metadata.transform,
            (2.0, 0.0, 10.0, 0.0, -2.0, 20.0, 0.0, 0.0, 1.0),
        )
        self.assertEqual(metadata.bounds, (10.0, 16.0, 16.0, 20.0))
        self.assertEqual(metadata.resolution, (2.0, 2.0))
        self.assertEqual(metadata.attributes["title"], "example dataset")
        self.assertEqual(len(metadata.variables), 2)

        temperature, quality = metadata.variables
        self.assertEqual(temperature.name, "temperature")
        self.assertEqual(temperature.shape, (2, 2, 3))
        self.assertEqual(temperature.dimensions, ("time", "y", "x"))
        self.assertEqual(temperature.dtype, "float32")
        self.assertIsNone(temperature.source_index)
        self.assertEqual(temperature.nodata, -9999.0)
        self.assertEqual(temperature.scale, 0.1)
        self.assertEqual(temperature.offset, 273.15)
        self.assertEqual(temperature.unit, "K")
        self.assertEqual(temperature.attributes["units"], "K")
        self.assertEqual(temperature.storage.chunk_shape, (1, 2, 3))
        self.assertEqual(temperature.storage.compression, "zlib")
        self.assertEqual(temperature.storage.compression_level, 4)
        self.assertTrue(temperature.storage.shuffle)

        self.assertEqual(quality.name, "quality")
        self.assertEqual(quality.shape, (2, 3))
        self.assertEqual(quality.dimensions, ("y", "x"))
        self.assertEqual(quality.dtype, "uint8")
        self.assertIsNone(quality.nodata)

    def test_irregular_coordinates_do_not_imply_a_spatial_grid(self):
        path = Path(self.temporary_directory.name) / "irregular.nc"
        dataset = xr.Dataset(
            data_vars={
                "value": (
                    ("y", "x"),
                    np.ones((2, 3), dtype=np.float32),
                ),
            },
            coords={
                "x": ("x", [0.0, 1.0, 3.0]),
                "y": ("y", [1.0, 0.0]),
            },
        )
        dataset.to_netcdf(path, engine="netcdf4")
        dataset.close()

        with XarrayBackend().open(make_asset(str(path))) as session:
            metadata = session.metadata_reader.read_metadata()
            request = WindowReadRequest(
                window=PixelWindow(0, 0, 1, 1),
                variable_name="value",
            )
            with self.assertRaisesRegex(ValueError, "regular spatial grid"):
                session.window_reader.read_window(request)

        self.assertIsNone(metadata.transform)
        self.assertIsNone(metadata.bounds)
        self.assertIsNone(metadata.resolution)

    def test_reads_named_variable_with_dimension_selection(self):
        request = WindowReadRequest(
            window=PixelWindow(0, 1, 2, 2),
            variable_name="temperature",
            dimension_indices=(("time", 1),),
        )

        with XarrayBackend().open(self.asset) as session:
            result = session.window_reader.read_window(request)

        np.testing.assert_array_equal(
            result.data,
            np.array([[7, 8], [10, 11]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            result.valid_mask,
            np.ones((2, 2), dtype=np.bool_),
        )
        self.assertEqual(
            result.transform,
            (2.0, 0.0, 12.0, 0.0, -2.0, 20.0, 0.0, 0.0, 1.0),
        )
        self.assertIs(result.request, request)

    def test_preserves_unselected_nonspatial_dimensions(self):
        request = WindowReadRequest(
            window=PixelWindow(0, 0, 1, 2),
            variable_name="temperature",
        )

        with XarrayBackend().open(self.asset) as session:
            result = session.window_reader.read_window(request)

        np.testing.assert_array_equal(
            result.data,
            np.array(
                [
                    [[0, 1]],
                    [[6, 7]],
                ],
                dtype=np.float32,
            ),
        )
        self.assertEqual(result.data.shape, (2, 1, 2))

    def test_boundless_read_pads_and_marks_nodata_invalid(self):
        request = WindowReadRequest(
            window=PixelWindow(-1, -1, 3, 3),
            variable_name="temperature",
            dimension_indices=(("time", 0),),
            boundless=True,
            fill_value=-1,
        )

        with XarrayBackend().open(self.asset) as session:
            result = session.window_reader.read_window(request)

        np.testing.assert_array_equal(
            result.data,
            np.array(
                [
                    [-1, -1, -1],
                    [-1, 0, 1],
                    [-1, 3, -9999],
                ],
                dtype=np.float32,
            ),
        )
        np.testing.assert_array_equal(
            result.valid_mask,
            np.array(
                [
                    [False, False, False],
                    [False, True, True],
                    [False, True, False],
                ]
            ),
        )
        self.assertEqual(
            result.transform,
            (2.0, 0.0, 8.0, 0.0, -2.0, 22.0, 0.0, 0.0, 1.0),
        )

    def test_boundless_read_can_be_fully_outside_dataset(self):
        request = WindowReadRequest(
            window=PixelWindow(5, 5, 2, 2),
            variable_name="quality",
            boundless=True,
            fill_value=9,
        )

        with XarrayBackend().open(self.asset) as session:
            result = session.window_reader.read_window(request)

        np.testing.assert_array_equal(
            result.data,
            np.full((2, 2), 9, dtype=np.uint8),
        )
        np.testing.assert_array_equal(
            result.valid_mask,
            np.zeros((2, 2), dtype=np.bool_),
        )
        self.assertEqual(
            result.transform,
            (2.0, 0.0, 20.0, 0.0, -2.0, 10.0, 0.0, 0.0, 1.0),
        )

    def test_rejects_out_of_bounds_window_without_boundless_reading(self):
        request = WindowReadRequest(
            window=PixelWindow(-1, 0, 2, 2),
            variable_name="quality",
        )

        with XarrayBackend().open(self.asset) as session:
            with self.assertRaisesRegex(ValueError, "boundless=True"):
                session.window_reader.read_window(request)

    def test_requires_explicit_variable_name(self):
        request = WindowReadRequest(window=PixelWindow(0, 0, 1, 1))

        with XarrayBackend().open(self.asset) as session:
            with self.assertRaisesRegex(ValueError, "variable_name"):
                session.window_reader.read_window(request)

    def test_rejects_unknown_variable_and_dimension(self):
        unknown_variable = WindowReadRequest(
            window=PixelWindow(0, 0, 1, 1),
            variable_name="missing",
        )
        unknown_dimension = WindowReadRequest(
            window=PixelWindow(0, 0, 1, 1),
            variable_name="temperature",
            dimension_indices=(("level", 0),),
        )

        with XarrayBackend().open(self.asset) as session:
            with self.assertRaisesRegex(ValueError, "not present"):
                session.window_reader.read_window(unknown_variable)
            with self.assertRaisesRegex(ValueError, "Dimensions"):
                session.window_reader.read_window(unknown_dimension)

    def test_rejects_spatial_dimension_selector(self):
        request = WindowReadRequest(
            window=PixelWindow(0, 0, 1, 1),
            variable_name="temperature",
            dimension_indices=(("x", 0),),
        )

        with XarrayBackend().open(self.asset) as session:
            with self.assertRaisesRegex(ValueError, "PixelWindow"):
                session.window_reader.read_window(request)

    def test_rejects_raster_source_indices(self):
        request = WindowReadRequest(
            window=PixelWindow(0, 0, 1, 1),
            source_indices=(1,),
        )

        with XarrayBackend().open(self.asset) as session:
            with self.assertRaisesRegex(ValueError, "variable_name"):
                session.window_reader.read_window(request)

    def test_backend_applies_config_and_reuses_one_open_dataset(self):
        config = XarrayBackendConfig(engine="netcdf4")
        backend = XarrayBackend(config)

        with patch(
            "tiling_template.readers.xarray_reader.xr.open_dataset",
            wraps=xr.open_dataset,
        ) as open_mock:
            with backend.open(self.asset) as session:
                first = session.metadata_reader.read_metadata()
                second = session.metadata_reader.read_metadata()

        open_mock.assert_called_once_with(
            self.asset.path,
            engine="netcdf4",
            group=None,
            decode_cf=False,
            mask_and_scale=False,
            cache=False,
        )
        self.assertEqual(first, second)

    def test_backend_initializes_plugin_modules_before_opening_dataset(self):
        backend = XarrayBackend(
            XarrayBackendConfig(
                engine="netcdf4",
                plugin_modules=("first_plugin", "second_plugin"),
            )
        )

        with patch(
            "tiling_template.readers.xarray_reader.import_module"
        ) as import_mock:
            with backend.open(self.asset) as session:
                metadata = session.metadata_reader.read_metadata()

        self.assertEqual(
            [args[0] for args, _ in import_mock.call_args_list],
            ["first_plugin", "second_plugin"],
        )
        self.assertEqual(metadata.asset, self.asset)

    def test_backend_reports_plugin_initialization_failure(self):
        backend = XarrayBackend(
            XarrayBackendConfig(plugin_modules=("missing_plugin",))
        )

        with patch(
            "tiling_template.readers.xarray_reader.import_module",
            side_effect=ModuleNotFoundError("missing dependency"),
        ):
            with self.assertRaisesRegex(
                ImportError,
                "missing_plugin",
            ):
                with backend.open(self.asset):
                    pass


if __name__ == "__main__":
    unittest.main()
