# Reader Backend Checks

These scripts run serial checks against real datasets through the reader
backend API. Run the Slurm wrapper from the repository root so Slurm can open
the configured files under `scripts/logs/`.

## Rasterio

The Rasterio check reads normalized metadata and deterministic sample windows.
It reads band 1 by default and uses windows no larger than 512 by 512 pixels.

```bash
sbatch scripts/sbatch_reader_backend_check.sh \
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
sbatch scripts/sbatch_reader_backend_check.sh \
  --backend xarray \
  --datasets /explore/nobackup/path/a.nc /explore/nobackup/path/b.nc
```

Both scripts use Python logging and write their `tqdm` progress bars to
standard output. They inspect every supplied dataset and return a nonzero exit
status if any dataset fails. The container must provide `tqdm` in addition to
the applicable backend dependencies.

The wrapper defaults to the Grace partition and this Apptainer sandbox:

```text
/explore/nobackup/people/ajkerr1/containers/pace-container-arm64
```

Set `CONTAINER_PATH`, `APPTAINER_BIN`, or `APPTAINER_BIND_PATHS` before
submission to override the corresponding defaults.
