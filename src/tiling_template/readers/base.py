from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..records import AssetRef, AssetMetadata


class MetadataReader(ABC):
    """Read normalized metadata from an asset."""

    @abstractmethod
    def read_metadata(self, asset: AssetRef) -> AssetMetadata:
        """Read metadata from asset without loading data values."""
        ...


class WindowReader(ABC):
    """Read array windows from an asset."""

    @abstractmethod
    def read_window(self, asset: AssetRef) -> NDArray[np.generic]:
        """Read a spatial window for selected indices (dimensions)."""
        ...


@dataclass(frozen=True, slots=True)
class AssetReaderBackend:
    """Compose the metadata and window readers for one file backend."""

    metadata_reader: MetadataReader
    window_reader: WindowReader


BackendFactory = Callable[[], AssetReaderBackend]


class ReaderRegistry:
    """
    Register and construct asset-reader backends by file extension.

    This prevents us from having to check file extensions downstream,
    and registered backends automatically handle each file format.
    """
    def __init__(self) -> None:
        self._factories: dict[str, BackendFactory] = {}

    @staticmethod
    def _normalize_extension(extension: str) -> str:
        normalized = extension.strip().lower()

        if not normalized:
            raise ValueError("File extension cannot be empty.")

        return f".{normalized.lstrip('.')}"

    def register(
        self,
        extensions: tuple[str, ...],
        factory: BackendFactory,
    ) -> None:
        normalized_extensions = tuple(
            self._normalize_extension(extension)
            for extension in extensions
        )

        if not normalized_extensions:
            raise ValueError("At least one file extension is required.")

        if len(set(normalized_extensions)) != len(extensions):
            raise ValueError("File extensions must be unique.")

        conflicts = set(normalized_extensions) & self._factories.keys()
        if conflicts:
            raise ValueError(
                f"Backends already registered for {sorted(conflicts)}"
            )

        # Update all relevant keys with the passed factory
        self._factories.update(
            {
                extension: factory
                for extension in normalized_extensions
            }
        )

    def backend_for(self, asset: AssetRef) -> AssetReaderBackend:
        """Construct the registered backend for an asset's extension."""
        extension = self._normalize_extension(asset.path.suffix)

        try:
            factory = self._factories[extension]
        except KeyError:
            raise ValueError(
                f"No reader is registered for: {extension}"
            ) from None

        return factory()
