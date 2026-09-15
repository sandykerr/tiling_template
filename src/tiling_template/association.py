from glob import glob
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Literal, Tuple, Sequence
from abc import ABC, abstractmethod

from configs import ModalityAssocationConfig
from discovery import AssetRef

FilenameMatchSpec = Literal['basename', 'stem']
DirectoryMatchSpec = Literal['immediate_parents', 'parent_indices']

@dataclass(frozen=True, slots=True)
class AssociationStrategy(ABC):

    @abstractmethod
    def key_for(self, asset: AssetRef) -> Tuple[str, ...]:
        """Generate an association key for a single asset reference."""
        ...


@dataclass(frozen=True, slots=True)
class FilenameAssociationStrategy(AssociationStrategy):
    """Associates files by filename."""
    match_type: FilenameMatchSpec = 'basename'
    prefix_to_remove: str | None = None
    suffix_to_remove: str | None = None

    def key_for(self, asset: AssetRef) -> Tuple[str, ...]:
        if self.match_type == 'basename':
            key_candidate = asset.path.name
        elif self.match_type == 'stem':
            key_candidate = asset.path.stem
        else:
            raise ValueError(
                "FilenameAssociationStrategy must use basename/stem matching."
            )
        # Conditional prefix removal
        if self.prefix_to_remove is not None:
            key_candidate_old = key_candidate
            key_candidate = key_candidate.removeprefix(self.prefix_to_remove)
            if key_candidate_old == key_candidate:
                print(
                    f"[WARNING]: Prefix was not removed from filename. "
                    f"Prefix: {self.prefix_to_remove}, "
                    f"Filename: {key_candidate_old}, "
                )
        # Conditional suffix removal
        if self.suffix_to_remove is not None:
            key_candidate_old = key_candidate
            key_candidate = key_candidate.removesuffix(self.suffix_to_remove)
            if key_candidate_old == key_candidate:
                print(
                    f"[WARNING]: Suffix was not removed from filename. "
                    f"Suffix: {self.suffix_to_remove}, "
                    f"Filename: {key_candidate_old}, "
                )
        return key_candidate


@dataclass(frozen=True, slots=True)
class DirectoryAssociationStrategy(AssociationStrategy):
    """Associates files by directory."""
    match_type: DirectoryMatchSpec
    # Allow for extraction of variable-index directories; such as .../a/.../b
    parent_indices: List[int] | None = None

    def key_for(self, asset: AssetRef) -> Tuple[str, ...]:
        pass


@dataclass(frozen=True, slots=True)
class CompositeAssociationStrategy(AssociationStrategy):
    """Associates files by filename and directory."""
    strategies: Tuple[AssociationStrategy, ...]

    def key_for(self, asset: AssetRef) -> Tuple[str, ...]:
        pass


@dataclass(frozen=True, slots=True)
class SourceRecord:
    """Record of what source data is used in the tiling pipeline."""
    source_id: str
    assets: Tuple[AssetRef, ...]


class AssetAssociator:
    """
    Associates assets into SourceRecords. Pairs connected modalities,
    inputs and labels, etc.

    --------------------

    After calculating keys, the associator should:

    1. Group AssetRef objects by association key.
    2. Count assets by (role, modality) within each group.
    3. Compare counts to configured requirements.
    4. Detect missing inputs, labels, QA files, metadata, or sidecars.
    5. Detect unexpected duplicates.
    6. Identify orphan assets.
    7. Create one stable SourceRecord per valid group.
    8. Return records in deterministic key order.

    """
    def __init__(
        self,
        found_assets: Sequence[AssetRef],
        strategy: AssociationStrategy,
        modality_configs: Sequence[ModalityAssocationConfig]
    ):
        self.found_assets = tuple(found_assets)
        self.strategy = strategy
        self.modality_configs = tuple(modality_configs)

    def associate_assets(self) -> Tuple[SourceRecord]:
        pass
