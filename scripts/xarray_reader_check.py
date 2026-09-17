#!/usr/bin/env python3
"""Run real-data checks against the Xarray reader backend."""

import argparse
import logging
import sys
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from multiprocessing import get_context
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
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of dataset worker processes. Defaults to 1.",
    )
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive.")
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
        spec_id="xarray-real-data-check",
        role="input",
        modality="array",
        size_bytes=file_stat.st_size,
        modified_time_ns=file_stat.st_mtime_ns,
    )


def inspect_dataset(path: Path) -> tuple[str, ...]:
    asset = asset_ref(path)
    started_at = perf_counter()
    messages: list[str] = []

    with XarrayBackend().open(asset) as session:
        metadata = session.metadata_reader.read_metadata()
        if not metadata.variables:
            raise ValueError("Dataset contains no data variables.")

        messages.append(
            f"Opened {asset.path} | size={asset.size_bytes} bytes | "
            f"variables={len(metadata.variables)} | crs={metadata.crs} | "
            f"resolution={metadata.resolution}"
        )
        for variable in metadata.variables:
            messages.append(
                f"Variable {variable.name} | "
                f"dimensions={variable.dimensions} | shape={variable.shape} | "
                f"dtype={variable.dtype} | "
                f"chunks={variable.storage.chunk_shape} | "
                f"compression={variable.storage.compression}"
            )

    messages.append(
        f"Completed {asset.path} in {perf_counter() - started_at:.3f} "
        "seconds"
    )
    return tuple(messages)


def log_messages(messages: tuple[str, ...]) -> None:
    for message in messages:
        LOGGER.info("%s", message)


def run_serial(paths: list[Path]) -> int:
    failures = 0
    for path in tqdm(
        paths,
        desc="Xarray datasets",
        unit="dataset",
        file=sys.stdout,
    ):
        try:
            log_messages(inspect_dataset(path))
        except Exception:
            failures += 1
            LOGGER.exception("Failed dataset: %s", path)
    return failures


def run_parallel(paths: list[Path], workers: int) -> int:
    failures = 0
    spawn_context = get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=spawn_context,
    ) as executor:
        futures: dict[Future[tuple[str, ...]], Path] = {
            executor.submit(inspect_dataset, path): path
            for path in paths
        }
        with tqdm(
            total=len(futures),
            desc="Xarray datasets",
            unit="dataset",
            file=sys.stdout,
        ) as progress:
            for future in as_completed(futures):
                path = futures[future]
                try:
                    log_messages(future.result())
                except Exception:
                    failures += 1
                    LOGGER.exception("Failed dataset: %s", path)
                finally:
                    progress.update()

    return failures


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    LOGGER.info(
        "Starting Xarray checks with %d worker process(es).",
        args.workers,
    )

    with logging_redirect_tqdm():
        if args.workers == 1:
            failures = run_serial(args.datasets)
        else:
            failures = run_parallel(args.datasets, args.workers)

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
