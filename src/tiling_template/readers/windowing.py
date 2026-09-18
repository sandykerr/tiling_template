from dataclasses import dataclass

from ..records import PixelWindow


@dataclass(frozen=True, slots=True)
class WindowGeometry:
    """Describe a requested window's intersection with a pixel source."""

    requested: PixelWindow
    source: PixelWindow | None
    destination_row_offset: int
    destination_column_offset: int
    extends_beyond_source: bool


def resolve_window_geometry(
    requested: PixelWindow,
    source_height: int,
    source_width: int,
) -> WindowGeometry:
    """Resolve half-open source and destination bounds for a pixel window."""

    if source_height < 0 or source_width < 0:
        raise ValueError("Source height and width cannot be negative.")

    requested_row_end = requested.row_offset + requested.height
    requested_column_end = requested.column_offset + requested.width
    extends_beyond_source = (
        requested.row_offset < 0
        or requested.column_offset < 0
        or requested_row_end > source_height
        or requested_column_end > source_width
    )

    source_row_start = max(0, requested.row_offset)
    source_row_end = min(source_height, requested_row_end)
    source_column_start = max(0, requested.column_offset)
    source_column_end = min(source_width, requested_column_end)

    if (
        source_row_start >= source_row_end
        or source_column_start >= source_column_end
    ):
        source = None
        destination_row_offset = 0
        destination_column_offset = 0
    else:
        source = PixelWindow(
            row_offset=source_row_start,
            column_offset=source_column_start,
            height=source_row_end - source_row_start,
            width=source_column_end - source_column_start,
        )
        destination_row_offset = source_row_start - requested.row_offset
        destination_column_offset = (
            source_column_start - requested.column_offset
        )

    return WindowGeometry(
        requested=requested,
        source=source,
        destination_row_offset=destination_row_offset,
        destination_column_offset=destination_column_offset,
        extends_beyond_source=extends_beyond_source,
    )


def translate_transform(
    transform: tuple[float, ...],
    window: PixelWindow,
) -> tuple[float, ...]:
    """Translate a 3x3 affine transform to a pixel window's origin."""

    a, b, c, d, e, f, g, h, i = transform
    return (
        a,
        b,
        c + a * window.column_offset + b * window.row_offset,
        d,
        e,
        f + d * window.column_offset + e * window.row_offset,
        g,
        h,
        i,
    )
