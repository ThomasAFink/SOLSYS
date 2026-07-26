"""2D/3D matplotlib axes adapter — always fed XYZ; drops Z in 2D."""

from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

Dimension = Literal['2d', '3d']


class DimensionPlotter:
    """Thin axes adapter: always fed XYZ; drops Z when drawing in 2D."""

    def __init__(self, axes: plt.Axes, dimension: Dimension):
        self.axes = axes
        self.is3d = dimension == '3d'

    def scatter(self, positionX, positionY, positionZ=None, **kwargs) -> None:
        if self.is3d:
            if positionZ is None:
                positionZ = np.zeros_like(np.asarray(positionX, dtype=float))
            self.axes.scatter(positionX, positionY, positionZ, **kwargs)
        else:
            self.axes.scatter(positionX, positionY, **kwargs)

    def plot(self, positionX, positionY, positionZ=None, *args, **kwargs) -> None:
        # Allow matplotlib format strings as *args, e.g. plot(x, y, z, '--', ...)
        if self.is3d:
            if positionZ is None:
                positionZ = np.zeros_like(np.asarray(positionX, dtype=float))
            self.axes.plot(positionX, positionY, positionZ, *args, **kwargs)
        else:
            self.axes.plot(positionX, positionY, *args, **kwargs)

    def text(self, positionX, positionY, positionZ, label: str, **kwargs) -> None:
        if self.is3d:
            self.axes.text(positionX, positionY, positionZ, label, **kwargs)
        else:
            self.axes.text(positionX, positionY, label, **kwargs)


