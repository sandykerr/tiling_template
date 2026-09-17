#!/usr/bin/env python3
"""Run serial, real-data checks against the Xarray reader backend."""

import argparse
import logging
import sys
from pathlib import Path
from time import perf_counter

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from tiling_template.readers.xarray_reader import XarrayBackend
from tiling_template.records import AssetRef


LOGGER = logging.getLogger("xarray_reader_check")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        type=Path,
        help="NetCDF datasets to inspect.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


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
        spec_id="xarray-real-data-check",
        role="input",
        modality="array",
        size_bytes=file_stat.st_size,
        modified_time_ns=file_stat.st_mtime_ns,
    )


def inspect_dataset(path: Path) -> None:
    asset = asset_ref(path)
    started_at = perf_counter()

    with XarrayBackend().open(asset) as session:
        metadata = session.metadata_reader.read_metadata()
        if not metadata.variables:
            raise ValueError("Dataset contains no data variables.")

        LOGGER.info(
            "Opened %s | size=%d bytes | variables=%d | crs=%s | "
            "resolution=%s",
            asset.path,
            asset.size_bytes,
            len(metadata.variables),
            metadata.crs,
            metadata.resolution,
        )
        for variable in metadata.variables:
            LOGGER.info(
                "Variable %s | dimensions=%s | shape=%s | dtype=%s | "
                "chunks=%s | compression=%s",
                variable.name,
                variable.dimensions,
                variable.shape,
                variable.dtype,
                variable.storage.chunk_shape,
                variable.storage.compression,
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
            desc="Xarray datasets",
            unit="dataset",
            file=sys.stdout,
        ):
            try:
                inspect_dataset(path)
            except Exception:
                failures += 1
                LOGGER.exception("Failed dataset: %s", path)

    if failures:
        LOGGER.error(
            "%d of %d Xarray datasets failed.",
            failures,
            len(args.datasets),
        )
        return 1

    LOGGER.info("All %d Xarray datasets passed.", len(args.datasets))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
