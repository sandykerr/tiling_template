import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
import xarray as xr

from tiling_template.readers import (
    AssetReaderBackend,
    MetadataReader,
    ReaderRegistry,
    WindowReader,
    default_reader_registry,
)
from tiling_template.readers.rasterio_reader import (
    RasterioMetadataReader,
    RasterioWindowReader,
)
from tiling_template.readers.xarray_reader import (
    XarrayMetadataReader,
    XarrayWindowReader,
)
from tiling_template.records import AssetMetadata, AssetRef, VariableMetadata


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


class FakeMetadataReader(MetadataReader):
    def read_metadata(self, asset: AssetRef) -> AssetMetadata:
        return AssetMetadata(
            asset=asset,
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
    def read_window(self, asset: AssetRef) -> np.ndarray:
        return np.zeros((1, 1), dtype=np.uint8)


class AlternateMetadataReader(FakeMetadataReader):
    pass


def fake_backend() -> AssetReaderBackend:
    return AssetReaderBackend(
        metadata_reader=FakeMetadataReader(),
        window_reader=FakeWindowReader(),
    )


def alternate_backend() -> AssetReaderBackend:
    return AssetReaderBackend(
        metadata_reader=AlternateMetadataReader(),
        window_reader=FakeWindowReader(),
    )


class TestReaderInterfaces(unittest.TestCase):
    def test_abstract_metadata_reader_cannot_be_instantiated(self):
        with self.assertRaises(TypeError):
            MetadataReader()

    def test_abstract_window_reader_cannot_be_instantiated(self):
        with self.assertRaises(TypeError):
            WindowReader()

    def test_reader_returns_current_metadata_record_shape(self):
        asset = make_asset("scene.tif")

        metadata = FakeMetadataReader().read_metadata(asset)

        self.assertIs(metadata.asset, asset)
        self.assertEqual(metadata.variables[0].name, "band_1")
        self.assertEqual(metadata.variables[0].shape, (1, 1))


class TestReaderRegistry(unittest.TestCase):
    def test_constructs_reader_for_each_registered_extension(self):
        registry = ReaderRegistry()
        registry.register((".tif", ".tiff"), fake_backend)

        self.assertIsInstance(
            registry.backend_for(make_asset("scene.tif")).metadata_reader,
            FakeMetadataReader,
        )
        self.assertIsInstance(
            registry.backend_for(make_asset("scene.tiff")).window_reader,
            FakeWindowReader,
        )

    def test_normalizes_case_whitespace_and_leading_period(self):
        registry = ReaderRegistry()
        registry.register((" TIF ",), fake_backend)

        backend = registry.backend_for(make_asset("scene.TIF"))

        self.assertIsInstance(backend.metadata_reader, FakeMetadataReader)

    def test_creates_a_new_reader_for_each_lookup(self):
        registry = ReaderRegistry()
        registry.register((".tif",), fake_backend)
        asset = make_asset("scene.tif")

        first = registry.backend_for(asset)
        second = registry.backend_for(asset)

        self.assertIsNot(first, second)
        self.assertIsNot(first.metadata_reader, second.metadata_reader)
        self.assertIsNot(first.window_reader, second.window_reader)

    def test_rejects_empty_extension_collection(self):
        registry = ReaderRegistry()

        with self.assertRaisesRegex(ValueError, "At least one"):
            registry.register((), fake_backend)

    def test_rejects_blank_extension(self):
        registry = ReaderRegistry()

        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            registry.register(("  ",), fake_backend)

    def test_rejects_duplicate_normalized_extensions(self):
        registry = ReaderRegistry()

        with self.assertRaisesRegex(ValueError, "must be unique"):
            registry.register(("tif", ".TIF"), fake_backend)

    def test_rejects_an_extension_registered_by_another_reader(self):
        registry = ReaderRegistry()
        registry.register((".tif",), fake_backend)

        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register((".TIF",), alternate_backend)

    def test_conflicting_registration_is_atomic(self):
        registry = ReaderRegistry()
        registry.register((".tif",), fake_backend)

        with self.assertRaises(ValueError):
            registry.register((".nc", ".tif"), alternate_backend)

        self.assertIsInstance(
            registry.backend_for(make_asset("scene.tif")).metadata_reader,
            FakeMetadataReader,
        )
        with self.assertRaisesRegex(ValueError, "No reader is registered"):
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
            registry.backend_for(make_asset("scene.tif")).metadata_reader,
            RasterioMetadataReader,
        )
        self.assertIsInstance(
            registry.backend_for(make_asset("scene.tiff")).window_reader,
            RasterioWindowReader,
        )

    def test_registers_xarray_for_netcdf_extension(self):
        registry = default_reader_registry()

        self.assertIsInstance(
            registry.backend_for(make_asset("scene.nc")).metadata_reader,
            XarrayMetadataReader,
        )
        self.assertIsInstance(
            registry.backend_for(make_asset("scene.nc")).window_reader,
            XarrayWindowReader,
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
            dataset.write(
                np.arange(12, dtype=np.uint16).reshape(2, 2, 3)
            )
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
        metadata = RasterioMetadataReader().read_metadata(self.asset)

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

    def test_window_read_explicitly_reports_not_implemented(self):
        with self.assertRaisesRegex(NotImplementedError, "window reading"):
            RasterioWindowReader().read_window(self.asset)


class TestXarrayReaders(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary_directory.name) / "scene.nc"
        dataset = xr.Dataset(
            data_vars={
                "temperature": (
                    ("time", "y", "x"),
                    np.arange(12, dtype=np.float32).reshape(2, 2, 3),
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
        metadata = XarrayMetadataReader().read_metadata(self.asset)

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

        metadata = XarrayMetadataReader().read_metadata(make_asset(str(path)))

        self.assertIsNone(metadata.transform)
        self.assertIsNone(metadata.bounds)
        self.assertIsNone(metadata.resolution)

    def test_window_read_explicitly_reports_not_implemented(self):
        with self.assertRaisesRegex(NotImplementedError, "window reading"):
            XarrayWindowReader().read_window(self.asset)


if __name__ == "__main__":
    unittest.main()
