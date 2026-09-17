from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias, Mapping
from numpy.typing import NDArray
import numpy as np

from .types import AssetRole


AssociationKey: TypeAlias = tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AssetRef:
    """Describe one local asset and its discovery context."""

    path: Path
    relative_path: Path
    spec_id: str
    role: AssetRole
    modality: str
    size_bytes: int
    modified_time_ns: int

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the asset."""

        return {
            "path": str(self.path),
            "relative_path": self.relative_path.as_posix(),
            "spec_id": self.spec_id,
            "role": self.role,
            "modality": self.modality,
            "size_bytes": self.size_bytes,
            "modified_time_ns": self.modified_time_ns,
        }


@dataclass(frozen=True, slots=True)
class SourceRecord:
    """Describe the physical assets associated with one logical source."""

    source_id: str
    assets: tuple[AssetRef, ...]
    association_key: AssociationKey


@dataclass(frozen=True, slots=True)
class VariableStorageMetadata:
    """Describe the physical storage layout of one variable."""

    chunk_shape: tuple[int, ...] | None = None
    compression: str | None = None
    compression_level: int | None = None
    shuffle: bool | None = None
    overview_factors: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class VariableMetadata:
    name: str
    shape: tuple[int, ...]  # dimensionality of data (MxNxO...)
    dimensions: tuple[str, ...]  # names of each shape idx
    dtype: str
    source_index: int | None = None  # 1-indexed band idx
    nodata: int | float | None = None
    scale: int | float | None = None
    offset: int | float | None = None
    unit: str | None = None
    attributes: Mapping[str, object] = field(default_factory=dict)
    storage: VariableStorageMetadata = field(
        default_factory=VariableStorageMetadata
    )


@dataclass(frozen=True, slots=True)
class AssetMetadata:
    """Represents the metadata extracted from a data asset."""
    asset: AssetRef
    variables: tuple[VariableMetadata, ...]
    crs: str | None = None
    transform: tuple[float, ...] | None = None
    bounds: tuple[float, ...] | None = None
    resolution: tuple[float, ...] | None = None
    is_tiled: bool | None = None
    attributes: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PixelWindow:
    """Describe a half-open window in integer pixel coordinates."""

    row_offset: int
    column_offset: int
    height: int
    width: int

    def __post_init__(self) -> None:
        if self.height <= 0 or self.width <= 0:
            raise ValueError(
                "Pixel window height and width must be positive."
            )


@dataclass(frozen=True, slots=True)
class WindowReadRequest:
    """Describe one pixel-window read from an open asset.

    Rasterio requests use one-based ``source_indices``. Xarray requests use
    one ``variable_name`` and may select integer positions from non-spatial
    dimensions through ``dimension_indices``.
    """

    window: PixelWindow
    source_indices: tuple[int, ...] | None = None  # 1-indexed
    variable_name: str | None = None
    dimension_indices: tuple[tuple[str, int], ...] = ()  # 1-indexed
    boundless: bool = False
    fill_value: int | float | None = None

    def __post_init__(self) -> None:
        if self.source_indices is not None:
            if not self.source_indices:
                raise ValueError("source_indices cannot be empty.")
            if any(index < 1 for index in self.source_indices):
                raise ValueError(
                    "Raster source indices must be one-based positive "
                    "integers."
                )
            if len(set(self.source_indices)) != len(self.source_indices):
                raise ValueError("source_indices must be unique.")

        if self.variable_name is not None and not self.variable_name.strip():
            raise ValueError("variable_name cannot be blank.")
        if self.source_indices is not None and self.variable_name is not None:
            raise ValueError(
                "source_indices and variable_name are mutually exclusive."
            )

        dimension_names = tuple(
            name for name, _ in self.dimension_indices
        )
        if any(not name.strip() for name in dimension_names):
            raise ValueError("Dimension names cannot be blank.")
        if len(set(dimension_names)) != len(dimension_names):
            raise ValueError("Dimension names must be unique.")


@dataclass(frozen=True, slots=True)
class WindowReadResult:
    """Contain one window read and an optional per-element validity mask."""

    data: NDArray[np.generic]
    valid_mask: NDArray[np.bool_] | None
    transform: tuple[float, ...]
    request: WindowReadRequest

    def __post_init__(self) -> None:
        if (
            self.valid_mask is not None
            and self.valid_mask.shape != self.data.shape
        ):
            raise ValueError(
                "valid_mask must have the same shape as data."
            )
