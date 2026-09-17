#!/usr/bin/env python3
"""Run serial, real-data checks against the Rasterio reader backend."""

import argparse
import logging
import sys
from pathlib import Path
from time import perf_counter

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from tiling_template.readers.rasterio_reader import RasterioBackend
from tiling_template.records import AssetRef, PixelWindow, WindowReadRequest


LOGGER = logging.getLogger("rasterio_reader_check")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        type=Path,
        help="GeoTIFF datasets to inspect.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=512,
        help="Maximum height and width of each sample window.",
    )
    parser.add_argument(
        "--windows-per-dataset",
        type=int,
        default=3,
        help="Number of deterministic diagonal windows to read.",
    )
    parser.add_argument(
        "--bands",
        nargs="+",
        type=int,
        default=[1],
        help="One-based band indices to read. Defaults to band 1.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    args = parser.parse_args()

    if args.window_size <= 0:
        parser.error("--window-size must be positive.")
    if args.windows_per_dataset <= 0:
        parser.error("--windows-per-dataset must be positive.")
    if any(index <= 0 for index in args.bands):
        parser.error("--bands must contain one-based positive integers.")
    if len(set(args.bands)) != len(args.bands):
        parser.error("--bands cannot contain duplicates.")

    return args


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )


def asset_ref(path: Path) -> AssetRef:
    resolved_path = path.expanduser().resolve(strict=True)
    if not resolved_path.is_file():
        raise ValueError(f"Dataset is not a file: {resolved_path}")

    file_stat = resolved_path.stat()
    return AssetRef(
        path=resolved_path,
        relative_path=Path(resolved_path.name),
        spec_id="rasterio-real-data-check",
        role="input",
        modality="raster",
        size_bytes=file_stat.st_size,
        modified_time_ns=file_stat.st_mtime_ns,
    )


def sample_windows(
    height: int,
    width: int,
    window_size: int,
    count: int,
) -> tuple[PixelWindow, ...]:
    sample_height = min(height, window_size)
    sample_width = min(width, window_size)
    maximum_row = height - sample_height
    maximum_column = width - sample_width

    if count == 1:
        fractions = (0.5,)
    else:
        fractions = tuple(index / (count - 1) for index in range(count))

    windows = {
        PixelWindow(
            row_offset=round(maximum_row * fraction),
            column_offset=round(maximum_column * fraction),
            height=sample_height,
            width=sample_width,
        )
        for fraction in fractions
    }
    return tuple(
        sorted(windows, key=lambda item: (item.row_offset, item.column_offset))
    )


def inspect_dataset(
    path: Path,
    *,
    window_size: int,
    windows_per_dataset: int,
    bands: tuple[int, ...],
) -> None:
    asset = asset_ref(path)
    started_at = perf_counter()

    with RasterioBackend().open(asset) as session:
        metadata = session.metadata_reader.read_metadata()
        if not metadata.variables:
            raise ValueError("Dataset contains no raster bands.")

        first_variable = metadata.variables[0]
        height, width = first_variable.shape
        LOGGER.info(
            "Opened %s | size=%d bytes | shape=%sx%s | bands=%d | "
            "dtype=%s | crs=%s | tiled=%s",
            asset.path,
            asset.size_bytes,
            height,
            width,
            len(metadata.variables),
            first_variable.dtype,
            metadata.crs,
            metadata.is_tiled,
        )

        windows = sample_windows(
            height,
            width,
            window_size,
            windows_per_dataset,
        )
        for window in tqdm(
            windows,
            desc=f"Windows: {asset.path.name}",
            unit="window",
            leave=False,
            file=sys.stdout,
        ):
            request = WindowReadRequest(
                window=window,
                source_indices=bands,
            )
            result = session.window_reader.read_window(request)
            valid_count = (
                int(result.valid_mask.sum())
                if result.valid_mask is not None
                else result.data.size
            )
            valid_fraction = valid_count / result.data.size
            LOGGER.info(
                "Read %s | offset=(%d, %d) | shape=%s | dtype=%s | "
                "valid=%.2f%%",
                asset.path.name,
                window.row_offset,
                window.column_offset,
                result.data.shape,
                result.data.dtype,
                valid_fraction * 100,
            )

    LOGGER.info(
        "Completed %s in %.3f seconds",
        asset.path,
        perf_counter() - started_at,
    )


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    failures = 0

    with logging_redirect_tqdm():
        for path in tqdm(
            args.datasets,
            desc="Rasterio datasets",
            unit="dataset",
            file=sys.stdout,
        ):
            try:
                inspect_dataset(
                    path,
                    window_size=args.window_size,
                    windows_per_dataset=args.windows_per_dataset,
                    bands=tuple(args.bands),
                )
            except Exception:
                failures += 1
                LOGGER.exception("Failed dataset: %s", path)

    if failures:
        LOGGER.error(
            "%d of %d Rasterio datasets failed.",
            failures,
            len(args.datasets),
        )
        return 1

    LOGGER.info("All %d Rasterio datasets passed.", len(args.datasets))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
