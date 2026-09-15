from rasterio.windows import Window
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TileWindow:
    column_offset: int
    row_offset: int
    width: int
    height: int

# Class representing the generic tiling interface
class Tiler(ABC):
    pass