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

from tiling_template.configs.reader import XarrayBackendConfig
from tiling_template.readers.xarray import (
    XarrayBackend,
    XarrayMetadataReader,
)
from tiling_template.records import (
    AssetRef,
    PixelWindow,
    WindowReadRequest,
    XarrayVariableSelection,
)


LOGGER = logging.getLogger("xarray_reader_check")


def dimension_index(value: str) -> tuple[str, int]:
    name, separator, raw_index = value.partition("=")
    if not separator or not name.strip():
        raise argparse.ArgumentTypeError(
            "Dimension indices must use NAME=INDEX syntax."
        )
    try:
        index = int(raw_index)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Dimension index must be an integer: {value}"
        ) from error
    return name.strip(), index


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
    variable_selection = parser.add_mutually_exclusive_group()
    variable_selection.add_argument(
        "--variables",
        nargs="+",
        metavar="NAME",
        help="Named data variables to sample.",
    )
    variable_selection.add_argument(
        "--all-variables",
        action="store_true",
        help="Sample every data variable with the dataset's spatial grid.",
    )
    parser.add_argument(
        "--dimension-index",
        action="append",
        default=[],
        type=dimension_index,
        metavar="NAME=INDEX",
        help="Select a non-spatial dimension index; may be repeated.",
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
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive.")
    if args.window_size <= 0:
        parser.error("--window-size must be positive.")
    if args.windows_per_dataset <= 0:
        parser.error("--windows-per-dataset must be positive.")
    dimension_names = tuple(name for name, _ in args.dimension_index)
    if len(set(dimension_names)) != len(dimension_names):
        parser.error("--dimension-index names must be unique.")
    if args.variables and len(set(args.variables)) != len(args.variables):
        parser.error("--variables names must be unique.")
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
    fractions = (
        (0.5,)
        if count == 1
        else tuple(index / (count - 1) for index in range(count))
    )
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
    variable_names: tuple[str, ...] | None,
    all_variables: bool,
    dimension_indices: tuple[tuple[str, int], ...],
    window_size: int,
    windows_per_dataset: int,
) -> tuple[str, ...]:
    asset = asset_ref(path)
    started_at = perf_counter()
    messages: list[str] = []

    backend = XarrayBackend(
        XarrayBackendConfig(plugin_modules=("hdf5plugin",))
    )
    with backend.open(asset) as session:
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

        if variable_names is not None or all_variables:
            dataset = session.window_reader.dataset
            x_coordinate = XarrayMetadataReader._coordinate_for_axis(
                dataset,
                "X",
            )
            y_coordinate = XarrayMetadataReader._coordinate_for_axis(
                dataset,
                "Y",
            )
            if x_coordinate is None or y_coordinate is None:
                raise ValueError("Dataset has no one-dimensional spatial grid.")

            x_dimension = x_coordinate.dims[0]
            y_dimension = y_coordinate.dims[0]
            if all_variables:
                candidates = tuple(
                    variable.name for variable in metadata.variables
                )
            else:
                candidates = variable_names or ()

            missing_variables = set(candidates) - set(dataset.data_vars)
            if missing_variables:
                raise ValueError(
                    "Variables are not present in the dataset: "
                    f"{sorted(missing_variables)}"
                )

            selected_names: list[str] = []
            for variable_name in candidates:
                variable = dataset[variable_name]
                is_spatial = (
                    x_dimension in variable.dims
                    and y_dimension in variable.dims
                )
                if not is_spatial and all_variables:
                    messages.append(
                        f"Skipped non-spatial variable {variable_name}"
                    )
                    continue
                if not is_spatial:
                    raise ValueError(
                        f"Variable {variable_name!r} has no spatial grid."
                    )
                selected_names.append(variable_name)

            if all_variables and not selected_names:
                raise ValueError(
                    "Dataset contains no spatially window-readable variables."
                )

            messages.append(
                "Window variables: " + ", ".join(selected_names)
            )
            for variable_name in selected_names:
                variable = dataset[variable_name]
                height = variable.sizes[y_dimension]
                width = variable.sizes[x_dimension]
                for window in sample_windows(
                    height,
                    width,
                    window_size,
                    windows_per_dataset,
                ):
                    request = WindowReadRequest(
                        window=window,
                        selection=XarrayVariableSelection(
                            variable_name=variable_name,
                            dimension_indices=dimension_indices,
                        ),
                    )
                    result = session.window_reader.read_window(request)
                    valid_count = (
                        int(result.valid_mask.sum())
                        if result.valid_mask is not None
                        else result.data.size
                    )
                    valid_fraction = valid_count / result.data.size
                    messages.append(
                        f"Read {asset.path.name}:{variable_name} | "
                        f"offset=({window.row_offset}, "
                        f"{window.column_offset}) | "
                        f"dimensions={result.dimensions} | "
                        f"shape={result.data.shape} | "
                        f"dtype={result.data.dtype} | "
                        f"valid={valid_fraction * 100:.2f}%"
                    )

    messages.append(
        f"Completed {asset.path} in {perf_counter() - started_at:.3f} "
        "seconds"
    )
    return tuple(messages)


def log_messages(messages: tuple[str, ...]) -> None:
    for message in messages:
        LOGGER.info("%s", message)


def run_serial(
    paths: list[Path],
    *,
    variable_names: tuple[str, ...] | None,
    all_variables: bool,
    dimension_indices: tuple[tuple[str, int], ...],
    window_size: int,
    windows_per_dataset: int,
) -> int:
    failures = 0
    for path in tqdm(
        paths,
        desc="Xarray datasets",
        unit="dataset",
        file=sys.stdout,
    ):
        try:
            log_messages(
                inspect_dataset(
                    path,
                    variable_names=variable_names,
                    all_variables=all_variables,
                    dimension_indices=dimension_indices,
                    window_size=window_size,
                    windows_per_dataset=windows_per_dataset,
                )
            )
        except Exception:
            failures += 1
            LOGGER.exception("Failed dataset: %s", path)
    return failures


def run_parallel(
    paths: list[Path],
    *,
    workers: int,
    variable_names: tuple[str, ...] | None,
    all_variables: bool,
    dimension_indices: tuple[tuple[str, int], ...],
    window_size: int,
    windows_per_dataset: int,
) -> int:
    failures = 0
    spawn_context = get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=spawn_context,
    ) as executor:
        futures: dict[Future[tuple[str, ...]], Path] = {
            executor.submit(
                inspect_dataset,
                path,
                variable_names=variable_names,
                all_variables=all_variables,
                dimension_indices=dimension_indices,
                window_size=window_size,
                windows_per_dataset=windows_per_dataset,
            ): path
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
            failures = run_serial(
                args.datasets,
                variable_names=(
                    tuple(args.variables) if args.variables else None
                ),
                all_variables=args.all_variables,
                dimension_indices=tuple(args.dimension_index),
                window_size=args.window_size,
                windows_per_dataset=args.windows_per_dataset,
            )
        else:
            failures = run_parallel(
                args.datasets,
                workers=args.workers,
                variable_names=(
                    tuple(args.variables) if args.variables else None
                ),
                all_variables=args.all_variables,
                dimension_indices=tuple(args.dimension_index),
                window_size=args.window_size,
                windows_per_dataset=args.windows_per_dataset,
            )

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
