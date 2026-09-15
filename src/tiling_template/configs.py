from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class SplitConfig:
    pass


@dataclass(frozen=True, slots=True)
class ModalityAssocationConfig:
    """Represents the configuration for a single modality association."""
    role: str
    modality: str
    minimum_count: int = 1
    maximum_count: int | None = None


@dataclass(frozen=True, slots=True)
class DataConfig:
    """
    Large pipeline-wide data configuration object.
    Composes multiple other configuration objects.
    """
    supervised: bool = True
