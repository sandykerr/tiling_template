from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RasterioBackendConfig:
    """Control worker-local Rasterio resource creation."""

    sharing: bool = False
    gdal_options: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        option_names = tuple(name for name, _ in self.gdal_options)
        if any(not name for name in option_names):
            raise ValueError("GDAL option names cannot be empty.")
        if len(set(option_names)) != len(option_names):
            raise ValueError("GDAL option names must be unique.")

    def options_dict(self) -> dict[str, object]:
        """Return options in the form expected by rasterio.Env."""

        return dict(self.gdal_options)


@dataclass(frozen=True, slots=True)
class XarrayBackendConfig:
    """Control worker-local xarray dataset creation."""

    engine: str | None = None
    group: str | None = None
    decode_cf: bool = False
    mask_and_scale: bool = False
    cache: bool = False
