# Future Considerations

## Multimodal Workers with Shared Source Assets

The current real-data reader checks submit one task per asset path. A future
multimodal tiling pipeline should instead submit a `SourceRecord`, or a batch
of source records, as the unit of work.

For example, many MODIS input GeoTIFFs may be associated with the same global
label GeoTIFF:

```text
SourceRecord
├── MODIS input GeoTIFF
└── global label GeoTIFF
```

This arrangement is compatible with process-based parallelism. Each worker
should open its input and label assets locally, preferably using
`contextlib.ExitStack`, and keep those sessions open while processing all
windows for that source. Open Rasterio or GDAL handles must not be pickled,
passed between processes, or shared across process boundaries.

Multiple workers may safely open independent read-only handles to the same
global label. However, this can introduce performance costs:

- The label dataset is opened independently in multiple processes.
- Each process maintains its own GDAL block cache.
- Concurrent reads may compete for filesystem bandwidth.
- Memory use can approach the worker count multiplied by the per-process GDAL
  cache size.
- Spatially unrelated source tasks may perform random reads from different
  parts of the global label.

The initial implementation should favor correctness and a simple lifecycle:

1. Submit one `SourceRecord` per worker task.
2. Open every associated asset inside the worker process.
3. Keep the asset sessions open for all windows belonging to that source.
4. Treat shared label assets as read-only.
5. Close all sessions deterministically when the task finishes or fails.
6. Benchmark worker counts because the workload may become I/O-bound before
   all allocated CPUs are useful.

If repeated label opens or cache duplication become material, introduce
source-record batching:

```text
worker starts
    -> opens the shared global label once
    -> processes a batch of input sources
        -> opens one input asset
        -> reads aligned input and label windows
        -> closes the input asset
    -> closes the global label
```

Where practical, assign geographically adjacent inputs to the same batch.
This can improve reuse of label blocks already present in the worker-local
GDAL cache. The batching policy belongs in orchestration rather than the
reader backend; reader backends should remain responsible only for opening,
reading, and closing worker-local asset resources.

