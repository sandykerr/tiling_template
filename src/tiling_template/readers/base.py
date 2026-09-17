from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass

from ..records import (
    AssetMetadata,
    AssetRef,
    WindowReadRequest,
    WindowReadResult,
)


class MetadataReader(ABC):
    """Read normalized metadata from an asset."""

    @abstractmethod
    def read_metadata(self) -> AssetMetadata:
        """Read metadata from asset without loading data values."""
        ...


class WindowReader(ABC):
    """Read array windows from an asset."""

    @abstractmethod
    def read_window(
        self,
        request: WindowReadRequest,
    ) -> WindowReadResult:
        """Read a spatial window for selected indices (dimensions)."""
        ...


@dataclass(frozen=True, slots=True)
class AssetReadSession:
    """Expose readers bound to one open asset resource."""

    metadata_reader: MetadataReader
    window_reader: WindowReader


class AssetReaderBackend(ABC):
    """Open worker-local read sessions for supported assets."""

    @abstractmethod
    def open(
        self,
        asset: AssetRef,
    ) -> AbstractContextManager[AssetReadSession]:
        """Open an asset and return a context-managed reader session."""
        ...


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
                f"No backend is registered for: {extension}"
            ) from None

        return factory()
