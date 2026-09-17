from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias, Mapping

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
