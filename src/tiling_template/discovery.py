import fnmatch
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .types import AssetRole


AssetSpecMatcher = Literal['glob', 'regex']
AssetSpecMatchScope = Literal['basename', 'stem', 'relative_path']


@dataclass(frozen=True, slots=True)
class AssetMatcher(ABC):
    """Match a string against an asset-discovery pattern."""

    pattern: str

    @abstractmethod
    def matches(self, value: str) -> bool:
        """Return whether value matches this pattern."""
        ...


@dataclass(frozen=True, slots=True)
class GlobMatcher(AssetMatcher):
    """Perform case-sensitive glob matching."""

    def matches(self, value: str) -> bool:
        return fnmatch.fnmatchcase(value, self.pattern)


@dataclass(frozen=True, slots=True)
class RegexMatcher(AssetMatcher):
    """Perform case-sensitive, full-string regular-expression matching."""

    pattern_compiled: re.Pattern[str] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        try:
            pattern_compiled = re.compile(self.pattern)
        except re.error as error:
            raise ValueError(
                f"Invalid regex pattern {self.pattern!r}: {error}"
            ) from error

        object.__setattr__(self, 'pattern_compiled', pattern_compiled)

    def matches(self, value: str) -> bool:
        return self.pattern_compiled.fullmatch(value) is not None


@dataclass(slots=True)
class AssetSpec:
    """Describe one category of local assets to discover."""

    spec_id: str
    root: Path
    role: AssetRole
    matcher_type: AssetSpecMatcher
    modality: str
    pattern: str
    match_scope: AssetSpecMatchScope
    recursive: bool = True
    exclude_patterns: tuple[str, ...] = ()
    max_depth: int = 10
    matcher: AssetMatcher = field(init=False, repr=False)
    exclude_matchers: tuple[AssetMatcher, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        matcher_class: type[AssetMatcher]
        if self.matcher_type == 'glob':
            matcher_class = GlobMatcher
        elif self.matcher_type == 'regex':
            matcher_class = RegexMatcher
        else:
            raise ValueError(
                f"Invalid AssetSpec matcher type: {self.matcher_type!r}"
            )

        if self.match_scope not in ('basename', 'stem', 'relative_path'):
            raise ValueError(
                f"Invalid AssetSpec match scope: {self.match_scope!r}"
            )
        if self.max_depth < 0:
            raise ValueError("AssetSpec max_depth cannot be negative.")

        self.matcher = matcher_class(self.pattern)
        self.exclude_matchers = tuple(
            matcher_class(pattern) for pattern in self.exclude_patterns
        )


@dataclass(frozen=True, slots=True)
class AssetRef:
    """Describe one local asset discovered from an AssetSpec."""

    path: Path
    relative_path: Path
    spec_id: str
    role: AssetRole
    modality: str
    size_bytes: int
    modified_time_ns: int

    def to_dict(self):
        return {
            "path": str(self.path),
            "relative_path": self.relative_path.as_posix(),
            "spec_id": self.spec_id,
            "role": self.role,
            "modality": self.modality,
            "size_bytes": self.size_bytes,
            "modified_time_ns": self.modified_time_ns,
        }


class AssetDiscoverer:
    """Discover matching local assets in deterministic path order."""

    def __init__(self, asset_specs: Sequence[AssetSpec]):
        self.asset_specs = tuple(asset_specs)
        if not self.asset_specs:
            raise ValueError("AssetDiscoverer requires at least one AssetSpec.")

        spec_ids = [asset_spec.spec_id for asset_spec in self.asset_specs]
        if len(set(spec_ids)) != len(spec_ids):
            raise ValueError("AssetSpec spec_id values must be unique.")

    @staticmethod
    def _check_asset_spec_root(asset_spec: AssetSpec) -> Path:
        """Return a resolved directory root or raise a discovery error."""

        root = asset_spec.root
        if not root.exists():
            raise ValueError(
                f"AssetSpec root does not exist for {asset_spec.spec_id!r}: "
                f"{root}"
            )
        if not root.is_dir():
            raise ValueError(
                f"AssetSpec root is not a directory for {asset_spec.spec_id!r}: "
                f"{root}"
            )
        return root.resolve()

    @staticmethod
    def _match_value(
        path: Path,
        relative_path: Path,
        match_scope: AssetSpecMatchScope,
    ) -> str:
        """Select the portion of a candidate path used for matching."""

        if match_scope == 'basename':
            return path.name
        if match_scope == 'stem':
            return path.stem
        return relative_path.as_posix()

    @classmethod
    def _is_excluded(
        cls,
        path: Path,
        relative_path: Path,
        asset_spec: AssetSpec,
    ) -> bool:
        """Return whether a path is hidden or matches an exclusion pattern."""

        if any(part.startswith('.') for part in relative_path.parts):
            return True

        value = cls._match_value(
            path,
            relative_path,
            asset_spec.match_scope,
        )
        return any(
            matcher.matches(value) for matcher in asset_spec.exclude_matchers
        )

    @classmethod
    def _iter_candidate_paths(
        cls,
        root: Path,
        asset_spec: AssetSpec,
    ) -> Iterator[Path]:
        """Walk a spec root while pruning excluded directories."""

        def raise_walk_error(error: OSError) -> None:
            raise error

        for current_root, directory_names, filenames in os.walk(
            root,
            topdown=True,
            onerror=raise_walk_error,
            followlinks=False,
        ):
            current_path = Path(current_root)
            relative_directory = current_path.relative_to(root)
            depth = len(relative_directory.parts)

            retained_directories: list[str] = []
            for directory_name in directory_names:
                directory_path = current_path / directory_name
                relative_path = directory_path.relative_to(root)
                if not cls._is_excluded(
                    directory_path,
                    relative_path,
                    asset_spec,
                ):
                    retained_directories.append(directory_name)
            directory_names[:] = retained_directories

            if not asset_spec.recursive or depth >= asset_spec.max_depth:
                directory_names.clear()

            for filename in filenames:
                path = current_path / filename
                relative_path = path.relative_to(root)
                if not cls._is_excluded(path, relative_path, asset_spec):
                    yield path

    @classmethod
    def _discover_spec(
        cls,
        asset_spec: AssetSpec,
        root: Path,
    ) -> Iterator[AssetRef]:
        """Yield references matching one specification."""

        for path in cls._iter_candidate_paths(root, asset_spec):
            relative_path = path.relative_to(root)
            value = cls._match_value(
                path,
                relative_path,
                asset_spec.match_scope,
            )
            if not asset_spec.matcher.matches(value):
                continue

            file_stat = path.stat()
            yield AssetRef(
                path=path.resolve(),
                relative_path=relative_path,
                spec_id=asset_spec.spec_id,
                role=asset_spec.role,
                modality=asset_spec.modality,
                size_bytes=file_stat.st_size,
                modified_time_ns=file_stat.st_mtime_ns,
            )

    def discover(self) -> list[AssetRef]:
        """Discover all files selected by the configured specifications."""

        results: list[AssetRef] = []
        matched_paths: dict[Path, str] = {}

        for asset_spec in self.asset_specs:
            root = self._check_asset_spec_root(asset_spec)
            for asset_ref in self._discover_spec(asset_spec, root):
                prior_spec_id = matched_paths.get(asset_ref.path)
                if prior_spec_id is not None:
                    raise ValueError(
                        f"Asset {asset_ref.path} matched multiple specs: "
                        f"{prior_spec_id!r} and {asset_ref.spec_id!r}."
                    )

                matched_paths[asset_ref.path] = asset_ref.spec_id
                results.append(asset_ref)

        return sorted(
            results,
            key=lambda asset_ref: (
                asset_ref.path.as_posix(),
                asset_ref.spec_id,
            ),
        )
