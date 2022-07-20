from __future__ import annotations

import uuid
from typing import Protocol, runtime_checkable

import numpy as np
from napari_broadcastable_points import BroadcastablePoints
from pymmcore_plus import CMMCorePlus
from useq import MDAEvent

__all__ = [
    "SnappableRamanAimingSource",
    "RamanAimingSource",
    "SimpleGridSource",
    "PointsLayerSource",
]


@runtime_checkable
class RamanAimingSource(Protocol):
    """
    A source of points to aim the laser
    """

    def get_mda_points(self, event: MDAEvent) -> np.ndarray:
        """
        Returns points in [0,1] space with shape (2, N)
        """
        ...

    name: str


@runtime_checkable
class SnappableRamanAimingSource(Protocol):
    name: str

    def get_current_points(self) -> np.ndarray:
        ...


class BaseSource:
    def __init__(self, name: str = None) -> None:
        if name is None:
            self._name = str(uuid.uuid1())

    @property
    def name(self) -> str:
        return self._name


class SimpleGridSource(BaseSource):
    """
    Make a grid to full extent of the Raman FOV
    """

    def __init__(self, N_x: int, N_y: int, name: str = None) -> None:
        self.N_x = N_x
        self.N_y = N_y
        X, Y = np.meshgrid(np.linspace(0, 1, N_x), np.linspace(0, 1, N_y))
        x = X.flatten()
        y = Y.flatten()
        self._grid = np.vstack([x, y])
        if name is None:
            name = f"grid-{uuid.uuid1()}"
        super().__init__(name)

    def get_current_points(self):
        return self._grid

    def get_mda_points(self, event: MDAEvent = None) -> np.ndarray:
        return self._grid


class PointsLayerSource(BaseSource):
    def __init__(
        self,
        points_layer: BroadcastablePoints,
        name: str = None,
        position_idx: int = 1,
        img_shape: tuple[int, int] = None,
    ) -> None:
        """
        Parameters
        ----------
        ...
        position_idx : int, default 1
            Which axis is position for the points layers. Can't assume this
            yet due to the brittleness of broadcastable points
        """
        self._pos_idx = position_idx
        self._points = points_layer
        if img_shape is None:
            core = CMMCorePlus.instance()
            self._img_shape = core.getImageWidth(), core.getImageHeight()
        else:
            self._img_shape = img_shape
        if name is None:
            name = f"points-{uuid.uuid1()}"
        super().__init__(name)

    def _get_pos_points(self, points: np.ndarray, pos: int):
        return points[points[:, self._pos_idx] == pos][:, -2:]

    def get_current_points(self) -> np.ndarray:
        points = self._points.last_displayed().T
        # put into [0, 1] for spectra collector
        points[0, :] /= self._img_shape[0]
        points[1, :] /= self._img_shape[1]
        return points

    def get_points_mda(self, event: MDAEvent) -> np.ndarray:
        p = event.index.get("p")
        points = self._get_pos_points(self._points.data, p)

        # put into [0, 1] for spectra collector
        points[0, :] /= self._img_shape[0]
        points[1, :] /= self._img_shape[1]
        return points
