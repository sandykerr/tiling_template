# Reader Backend Checks

These scripts run checks against real datasets through the reader backend API.
Run the Slurm wrapper from the repository root so Slurm can open the configured
files under `scripts/logs/`.

The wrapper creates one Python worker process per Slurm CPU allocated through
`--cpus-per-task`. It defaults to one CPU and therefore serial processing. Each
worker opens and closes its own backend sessions; the parent process owns
logging and the progress bar.

## Rasterio

The Rasterio check reads normalized metadata and deterministic sample windows.
It reads band 1 by default and uses windows no larger than 512 by 512 pixels.

```bash
sbatch --cpus-per-task=10 scripts/sbatch_reader_backend_check.sh \
  --backend rasterio \
  --datasets /explore/nobackup/path/a.tif /explore/nobackup/path/b.tif \
  --bands 1 2 3 \
  --window-size 256 \
  --windows-per-dataset 5
```

## Xarray

The Xarray check exercises the currently supported metadata-reading path. It
does not load complete data variables or call the intentionally unimplemented
Xarray window reader.

```bash
sbatch --cpus-per-task=10 scripts/sbatch_reader_backend_check.sh \
  --backend xarray \
  --datasets /explore/nobackup/path/a.nc /explore/nobackup/path/b.nc
```

Both scripts use Python logging and write their `tqdm` progress bars to
standard output. They inspect every supplied dataset and return a nonzero exit
status if any dataset fails. The container must provide `tqdm` in addition to
the applicable backend dependencies.

The Python entry points also accept `--workers` when run directly. When using
the Slurm wrapper, worker count is intentionally controlled only by
`--cpus-per-task` so the process pool cannot exceed its CPU allocation.

The wrapper defaults to the Grace partition and this Apptainer sandbox:

```text
/explore/nobackup/people/ajkerr1/containers/pace-container-arm64
```

Set `CONTAINER_PATH`, `APPTAINER_BIN`, or `APPTAINER_BIND_PATHS` before
submission to override the corresponding defaults.
