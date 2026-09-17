from dataclasses import dataclass

from ..types import AssetRole


@dataclass(frozen=True, slots=True)
class SplitConfig:
    pass


@dataclass(frozen=True, slots=True)
class ModalityAssociationConfig:
    """Represent the cardinality required for one role and modality."""

    role: AssetRole
    modality: str
    minimum_count: int = 1
    maximum_count: int | None = None

    def __post_init__(self) -> None:
        if self.minimum_count < 0:
            raise ValueError(
                "Modality minimum_count cannot be negative."
            )

        if self.maximum_count is not None:
            if self.maximum_count < 0:
                raise ValueError(
                    "Modality maximum_count cannot be negative."
                )
            if self.maximum_count < self.minimum_count:
                raise ValueError(
                    "Modality maximum_count cannot be less than minimum_count."
                )


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Compose pipeline-wide data configuration."""

    supervised: bool = True
