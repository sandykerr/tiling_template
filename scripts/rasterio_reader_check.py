#!/usr/bin/env python3
"""Run real-data checks against the Rasterio reader backend."""

import argparse
import logging
import sys
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    wait,
)
from dataclasses import dataclass
from multiprocessing import get_context
from multiprocessing.queues import Queue
from pathlib import Path
from queue import Empty
from time import perf_counter

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from tiling_template.readers.rasterio_reader import RasterioBackend
from tiling_template.records import AssetRef, PixelWindow, WindowReadRequest


LOGGER = logging.getLogger("rasterio_reader_check")
_PROGRESS_QUEUE: Queue | None = None


@dataclass(frozen=True, slots=True)
class WindowProgress:
    """Report one dataset's window-read progress to the parent process."""

    task_id: int
    dataset_name: str
    completed: int
    total: int


def initialize_progress_queue(progress_queue: Queue) -> None:
    """Bind a parent-owned progress queue inside a worker process."""

    global _PROGRESS_QUEUE
    _PROGRESS_QUEUE = progress_queue


def publish_window_progress(event: WindowProgress) -> None:
    if _PROGRESS_QUEUE is None:
        raise RuntimeError("Worker progress queue has not been initialized.")
    _PROGRESS_QUEUE.put(event)


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
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of dataset worker processes. Defaults to 1.",
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
    progress_task_id: int | None = None,
) -> tuple[str, ...]:
    asset = asset_ref(path)
    started_at = perf_counter()
    messages: list[str] = []

    with RasterioBackend().open(asset) as session:
        metadata = session.metadata_reader.read_metadata()
        if not metadata.variables:
            raise ValueError("Dataset contains no raster bands.")

        first_variable = metadata.variables[0]
        height, width = first_variable.shape
        messages.append(
            f"Opened {asset.path} | size={asset.size_bytes} bytes | "
            f"shape={height}x{width} | bands={len(metadata.variables)} | "
            f"dtype={first_variable.dtype} | crs={metadata.crs} | "
            f"tiled={metadata.is_tiled}"
        )

        windows = sample_windows(
            height,
            width,
            window_size,
            windows_per_dataset,
        )
        if progress_task_id is not None:
            publish_window_progress(
                WindowProgress(
                    task_id=progress_task_id,
                    dataset_name=asset.path.name,
                    completed=0,
                    total=len(windows),
                )
            )

        for completed, window in enumerate(windows, start=1):
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
            messages.append(
                f"Read {asset.path.name} | "
                f"offset=({window.row_offset}, {window.column_offset}) | "
                f"shape={result.data.shape} | dtype={result.data.dtype} | "
                f"valid={valid_fraction * 100:.2f}%"
            )
            if progress_task_id is not None:
                publish_window_progress(
                    WindowProgress(
                        task_id=progress_task_id,
                        dataset_name=asset.path.name,
                        completed=completed,
                        total=len(windows),
                    )
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
    window_size: int,
    windows_per_dataset: int,
    bands: tuple[int, ...],
) -> int:
    failures = 0
    for path in tqdm(
        paths,
        desc="Rasterio datasets",
        unit="dataset",
        file=sys.stdout,
    ):
        try:
            log_messages(
                inspect_dataset(
                    path,
                    window_size=window_size,
                    windows_per_dataset=windows_per_dataset,
                    bands=bands,
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
    window_size: int,
    windows_per_dataset: int,
    bands: tuple[int, ...],
) -> int:
    failures = 0
    spawn_context = get_context("spawn")
    progress_queue = spawn_context.Queue()
    window_bars: dict[int, tqdm] = {}
    bar_positions: dict[int, int] = {}
    available_positions = list(range(1, workers + 1))
    finished_tasks: set[int] = set()

    def drain_progress_events() -> None:
        while True:
            try:
                event = progress_queue.get_nowait()
            except Empty:
                return

            if event.task_id in finished_tasks:
                continue

            progress = window_bars.get(event.task_id)
            if progress is None:
                if available_positions:
                    position = available_positions.pop(0)
                else:
                    position = max(bar_positions.values(), default=0) + 1
                bar_positions[event.task_id] = position
                progress = tqdm(
                    total=event.total,
                    desc=f"Windows: {event.dataset_name}",
                    unit="window",
                    position=position,
                    leave=False,
                    file=sys.stdout,
                )
                window_bars[event.task_id] = progress

            increment = event.completed - int(progress.n)
            if increment > 0:
                progress.update(increment)

    def close_window_bar(task_id: int, *, completed: bool) -> None:
        progress = window_bars.pop(task_id, None)
        position = bar_positions.pop(task_id, None)
        if progress is not None:
            if completed and progress.total is not None:
                progress.update(max(0, progress.total - progress.n))
            progress.close()
        if position is not None:
            available_positions.append(position)
            available_positions.sort()

    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=spawn_context,
            initializer=initialize_progress_queue,
            initargs=(progress_queue,),
        ) as executor:
            futures: dict[Future[tuple[str, ...]], tuple[int, Path]] = {
                executor.submit(
                    inspect_dataset,
                    path,
                    window_size=window_size,
                    windows_per_dataset=windows_per_dataset,
                    bands=bands,
                    progress_task_id=task_id,
                ): (task_id, path)
                for task_id, path in enumerate(paths)
            }
            pending = set(futures)
            with tqdm(
                total=len(futures),
                desc="Rasterio datasets",
                unit="dataset",
                position=0,
                file=sys.stdout,
            ) as dataset_progress:
                while pending:
                    completed_futures, pending = wait(
                        pending,
                        timeout=0.1,
                        return_when=FIRST_COMPLETED,
                    )
                    drain_progress_events()

                    for future in completed_futures:
                        task_id, path = futures[future]
                        try:
                            messages = future.result()
                        except Exception:
                            failures += 1
                            finished_tasks.add(task_id)
                            close_window_bar(task_id, completed=False)
                            LOGGER.exception("Failed dataset: %s", path)
                        else:
                            finished_tasks.add(task_id)
                            close_window_bar(task_id, completed=True)
                            log_messages(messages)
                        finally:
                            dataset_progress.update()

                drain_progress_events()
    finally:
        for task_id in tuple(window_bars):
            close_window_bar(task_id, completed=False)
        progress_queue.close()
        progress_queue.join_thread()

    return failures


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    LOGGER.info(
        "Starting Rasterio checks with %d worker process(es).",
        args.workers,
    )

    with logging_redirect_tqdm():
        if args.workers == 1:
            failures = run_serial(
                args.datasets,
                window_size=args.window_size,
                windows_per_dataset=args.windows_per_dataset,
                bands=tuple(args.bands),
            )
        else:
            failures = run_parallel(
                args.datasets,
                workers=args.workers,
                window_size=args.window_size,
                windows_per_dataset=args.windows_per_dataset,
                bands=tuple(args.bands),
            )

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
