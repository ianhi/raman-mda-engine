from __future__ import annotations

import numbers
import uuid
from abc import abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np
from napari_broadcastable_points import BroadcastablePoints
from napari.layers import Shapes
from pymmcore_plus import CMMCorePlus
from useq import MDAEvent
from .util import polygon_laser_focus

__all__ = [
    "SnappableRamanAimingSource",
    "RamanAimingSource",
    "SimpleGridSource",
    "PointsLayerSource",
    "ShapesLayerSource"
]


@runtime_checkable
class RamanAimingSource(Protocol):
    @abstractmethod
    def get_mda_points(self, event: MDAEvent) -> np.ndarray:
        """
        Generate points to aim the laser for a given MDA event

        Parameters
        ----------
        event : useq.MDAEvent

        Returns
        -------
        relative_coords : (N, 2) array
            Positions to aim the laser in relative coordinates [0, 1]

        """

    name: str


@runtime_checkable
class SnappableRamanAimingSource(RamanAimingSource, Protocol):
    @abstractmethod
    def get_current_points(self) -> np.ndarray:
        """
        Returns
        -------
        relative_coords : (N, 2) array
            Positions to aim the laser in relative coordinates [0, 1]
        """


class BaseSource:
    def __init__(self, name: str = None) -> None:
        if name is None:
            self._name = str(uuid.uuid1())
        else:
            self._name = name

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
        self._grid = np.hstack([x[:, None], y[:, None]])
        if name is None:
            name = f"grid-{N_x}_{N_y}-{uuid.uuid1()}"
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
        points = self._points.last_displayed()
        # put into [0, 1] for spectra collector
        points[:, 0] /= self._img_shape[0]
        points[:, 1] /= self._img_shape[1]
        return points

    def get_points_mda(self, event: MDAEvent) -> np.ndarray:
        p = event.index.get("p")
        points = self._get_pos_points(self._points.data, p)

        # put into [0, 1] for spectra collector
        points[:, 0] /= self._img_shape[0]
        points[:, 1] /= self._img_shape[1]
        return points

class ShapesLayerSource(BaseSource):
    def __init__(
        self,
        shapes_layer: Shapes,
        name: str = None,
        position_idx: int = 1,
        img_shape: tuple[int, int] = None,
        spacing: int = 15,
    ) -> None:
        """
        Parameters
        ----------
        ...
        position_idx : int, default 1
            Which axis is position for the points layers. Can't assume this
            yet due to the brittleness of broadcastable points
        spacing: int, default 15
            Number of pixels between points
        """

        self._pos_idx = position_idx
        self._shapes = shapes_layer
        self._spacing = spacing
        if img_shape is None:
            core = CMMCorePlus.instance()
            self._img_shape = core.getImageWidth(), core.getImageHeight()
        else:
            self._img_shape = img_shape
        if name is None:
            name = f"shapes-{uuid.uuid1()}"

        super().__init__(name)

    @property
    def spacing(self) -> int:
        return self._spacing

    @spacing.setter
    def spacing(self, val: int):
        if not isinstance(val, numbers.Integral):
            raise TypeError("spacing must be an integer")
        self._spacing = val

    def get_current_points(self, spacing: int = None) -> np.ndarray:
        """
        Parameters
        ----------
        spacing : int, optional
            THe spacing between points in pixels. If *None* default to self.spacing
        """
        spacing = spacing or self._spacing
        all_points = []
        for i in range(len(self._shapes.data)):
            points = polygon_laser_focus(
                shape_data=self._shapes.data[i],
                shape_type=self._shapes.shape_type[i],
                density=spacing,
                plot=False
            )
            points[:, 0] /= self._img_shape[0]
            points[:, 1] /= self._img_shape[1]
            all_points.append(points)

        return np.vstack(all_points)

    def get_mda_points(self):
        return self.get_current_points()