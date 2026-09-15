from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

from .configs import ModalityAssociationConfig
from .discovery import AssetRef


AssociationKey: TypeAlias = tuple[str, ...]
SourceIdFactory: TypeAlias = Callable[[AssociationKey], str]
FilenameMatchSpec = Literal['basename', 'stem']
DirectoryMatchSpec = Literal['immediate_parent', 'parent_indices']


class AssociationValidationError(ValueError):
    """Report all source groups that violate association requirements."""

    def __init__(self, issues: Sequence[str]):
        self.issues = tuple(issues)
        details = '\n'.join(f"- {issue}" for issue in self.issues)
        super().__init__(f"Asset association validation failed:\n{details}")


class AssociationStrategy(ABC):
    """Generate a shared key for assets belonging to one logical source."""

    @abstractmethod
    def key_for(self, asset: AssetRef) -> AssociationKey:
        """Generate an association key for one asset reference."""
        ...


@dataclass(frozen=True, slots=True)
class FilenameAssociationStrategy(AssociationStrategy):
    """Associate assets using a normalized filename."""

    match_type: FilenameMatchSpec = 'basename'
    prefix_to_remove: str | None = None
    suffix_to_remove: str | None = None

    def __post_init__(self) -> None:
        if self.match_type not in ('basename', 'stem'):
            raise ValueError(
                f"Invalid filename association match type: {self.match_type!r}"
            )

    def key_for(self, asset: AssetRef) -> AssociationKey:
        if self.match_type == 'basename':
            key_candidate = asset.path.name
        else:
            key_candidate = asset.path.stem

        if self.prefix_to_remove is not None:
            if not key_candidate.startswith(self.prefix_to_remove):
                raise ValueError(
                    f"Filename {key_candidate!r} does not start with configured "
                    f"prefix {self.prefix_to_remove!r}."
                )
            key_candidate = key_candidate.removeprefix(self.prefix_to_remove)

        if self.suffix_to_remove is not None:
            if not key_candidate.endswith(self.suffix_to_remove):
                raise ValueError(
                    f"Filename {key_candidate!r} does not end with configured "
                    f"suffix {self.suffix_to_remove!r}."
                )
            key_candidate = key_candidate.removesuffix(self.suffix_to_remove)

        if not key_candidate:
            raise ValueError("Filename association produced an empty key.")
        return (key_candidate,)


@dataclass(frozen=True, slots=True)
class DirectoryAssociationStrategy(AssociationStrategy):
    """Associate assets using components of their relative parent directory."""

    match_type: DirectoryMatchSpec
    parent_indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.match_type not in ('immediate_parent', 'parent_indices'):
            raise ValueError(
                f"Invalid directory association match type: {self.match_type!r}"
            )
        if self.match_type == 'immediate_parent':
            if self.parent_indices is not None:
                raise ValueError(
                    "parent_indices cannot be set for immediate_parent matching."
                )
        elif not self.parent_indices:
            raise ValueError(
                "parent_indices matching requires at least one index."
            )
        elif len(set(self.parent_indices)) != len(self.parent_indices):
            raise ValueError("Directory parent indices must be unique.")

    def key_for(self, asset: AssetRef) -> AssociationKey:
        parent_parts = asset.relative_path.parent.parts
        if not parent_parts:
            raise ValueError(
                f"Asset has no relative parent directory: {asset.relative_path}"
            )

        if self.match_type == 'immediate_parent':
            return (parent_parts[-1],)

        assert self.parent_indices is not None
        invalid_indices = [
            index
            for index in self.parent_indices
            if not -len(parent_parts) <= index < len(parent_parts)
        ]
        if invalid_indices:
            raise ValueError(
                f"Parent indices {tuple(invalid_indices)} are out of bounds for "
                f"{asset.relative_path}; parent components are {parent_parts}."
            )
        return tuple(parent_parts[index] for index in self.parent_indices)


@dataclass(frozen=True, slots=True)
class CompositeAssociationStrategy(AssociationStrategy):
    """Associate assets by composing multiple strategies in order."""

    strategies: tuple[AssociationStrategy, ...]

    def __post_init__(self) -> None:
        if not self.strategies:
            raise ValueError(
                "CompositeAssociationStrategy requires at least one strategy."
            )

    def key_for(self, asset: AssetRef) -> AssociationKey:
        return tuple(
            component
            for strategy in self.strategies
            for component in strategy.key_for(asset)
        )


@dataclass(frozen=True, slots=True)
class SourceRecord:
    """Describe the physical assets associated with one logical source."""

    source_id: str
    assets: tuple[AssetRef, ...]
    association_key: AssociationKey


def _default_source_id(association_key: AssociationKey) -> str:
    """Create a readable source ID from an association key."""

    return '__'.join(association_key)


class AssetAssociator:
    """Group discovered assets and validate per-source modality cardinality."""

    def __init__(
        self,
        found_assets: Sequence[AssetRef],
        strategy: AssociationStrategy,
        modality_configs: Sequence[ModalityAssociationConfig] = (),
        *,
        reject_unconfigured: bool = False,
        source_id_factory: SourceIdFactory = _default_source_id,
    ):
        self.found_assets = tuple(found_assets)
        self.strategy = strategy
        self.modality_configs = tuple(modality_configs)
        self.reject_unconfigured = reject_unconfigured
        self.source_id_factory = source_id_factory

        configured_modalities = [
            (config.role, config.modality)
            for config in self.modality_configs
        ]
        if len(set(configured_modalities)) != len(configured_modalities):
            raise ValueError(
                "Modality configurations must have unique role/modality pairs."
            )

        asset_paths = [asset.path for asset in self.found_assets]
        if len(set(asset_paths)) != len(asset_paths):
            raise ValueError("Found assets must not contain duplicate paths.")

    @staticmethod
    def _validate_key(key: object, asset: AssetRef) -> AssociationKey:
        if not isinstance(key, tuple) or not key:
            raise ValueError(
                f"Association strategy returned an invalid key for {asset.path}: "
                f"{key!r}"
            )
        if any(not isinstance(component, str) or not component for component in key):
            raise ValueError(
                f"Association key components must be nonempty strings for "
                f"{asset.path}: {key!r}"
            )
        return key

    def _group_assets_by_key(self) -> dict[AssociationKey, list[AssetRef]]:
        grouped_assets: defaultdict[AssociationKey, list[AssetRef]] = defaultdict(
            list
        )
        for asset in self.found_assets:
            key = self._validate_key(self.strategy.key_for(asset), asset)
            grouped_assets[key].append(asset)
        return dict(grouped_assets)

    def _validate_grouped_assets(
        self,
        grouped_assets: dict[AssociationKey, list[AssetRef]],
    ) -> None:
        configured_modalities = {
            (config.role, config.modality)
            for config in self.modality_configs
        }
        issues: list[str] = []

        for association_key in sorted(grouped_assets):
            assets = grouped_assets[association_key]
            counts = Counter(
                (asset.role, asset.modality)
                for asset in assets
            )

            for config in self.modality_configs:
                count = counts[(config.role, config.modality)]
                if count < config.minimum_count:
                    issues.append(
                        f"key {association_key!r} has {count} "
                        f"{config.role}/{config.modality} assets; expected at "
                        f"least {config.minimum_count}."
                    )
                if (
                    config.maximum_count is not None
                    and count > config.maximum_count
                ):
                    issues.append(
                        f"key {association_key!r} has {count} "
                        f"{config.role}/{config.modality} assets; expected at "
                        f"most {config.maximum_count}."
                    )

            if self.reject_unconfigured:
                for role, modality in sorted(set(counts) - configured_modalities):
                    issues.append(
                        f"key {association_key!r} contains unconfigured "
                        f"{role}/{modality} assets."
                    )

        if issues:
            raise AssociationValidationError(issues)

    def associate_assets(self) -> tuple[SourceRecord, ...]:
        grouped_assets = self._group_assets_by_key()
        self._validate_grouped_assets(grouped_assets)

        source_records: list[SourceRecord] = []
        source_ids: dict[str, AssociationKey] = {}
        for association_key in sorted(grouped_assets):
            source_id = self.source_id_factory(association_key)
            if not source_id:
                raise ValueError(
                    f"Source ID factory returned an empty ID for {association_key!r}."
                )

            prior_key = source_ids.get(source_id)
            if prior_key is not None and prior_key != association_key:
                raise ValueError(
                    f"Source ID {source_id!r} was generated for multiple keys: "
                    f"{prior_key!r} and {association_key!r}."
                )
            source_ids[source_id] = association_key

            assets = tuple(sorted(
                grouped_assets[association_key],
                key=lambda asset: (
                    asset.role,
                    asset.modality,
                    asset.relative_path.as_posix(),
                    asset.path.as_posix(),
                ),
            ))
            source_records.append(SourceRecord(
                source_id=source_id,
                assets=assets,
                association_key=association_key,
            ))

        return tuple(source_records)
