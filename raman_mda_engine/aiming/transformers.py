import numbers
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = [
    "Transformer",
    "Identity",
    "Crosshair",
    "Square",
    "Circle",
]


@runtime_checkable
class Transformer(Protocol):
    def __init__(self):
        pass

    def transform(self, xy: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Identity(Transformer):
    def transform(self, xy):
        return xy


class Crosshair(Transformer):
    def __init__(self, spacing):
        """
        Parameters
        ----------
        spacing : float
            Spacing between points in relative coordinates.
        """
        super().__init__()
        self._spacing = spacing

    @property
    def spacing(self) -> float:
        return self._spacing

    @spacing.setter
    def spacing(self, val: float):
        if not val > 0 and val <= 1:
            raise ValueError("Must be between 0 and 1")

    def transform(self, xy, spacing: float = None):
        """
        Parameters
        ----------
        xy : (N, 2) arraylike
        spacing : float, or None
            How far to space the points out.
            If *None* use self.spacing

        Returns
        -------
        xy (N*spacing, 2)
        """
        spacing = spacing or self.spacing
        mod = np.array([[-spacing, 0, 0, 0, spacing], [0, spacing, 0, -spacing, 0]]).T

        return (xy + mod[:, None, :]).reshape(-1, 2)


class Square(Transformer):
    def __init__(self, edge_length: float, N_points: int):
        """
        Parameters
        ----------
        edge_length : float
            Length of the edge.
        N_points : int
            Number of points to have per edge.
        """
        super().__init__()
        self._edge_length = edge_length
        self._N_points = N_points

    @property
    def edge_length(self) -> float:
        return self._edge_length

    @edge_length.setter
    def edge_length(self, val: float):
        self._edge_length = val

    @property
    def N_points(self) -> int:
        return self._N_points

    @N_points.setter
    def N_points(self, val: int):
        if not isinstance(val, numbers.Integral) or val <= 0:
            raise TypeError("N_points must be a positive integer")

    def transform(self, xy, edge_length=None, N_points=None):
        """
        Make a square centered on the point.

        Parameters
        ----------
        xy : (N, 2) arraylike
        edge_length : float, optional
            Length of the edge. If *None* use self.spacing
        N_points : int, optional
            Number of points to have per edge. If *None* use self.N_points

        Returns
        -------
        xy (N*N_points**2, 2)
        """
        edge_length = edge_length or self.edge_length
        N_points = N_points or self.N_points

        edge = np.linspace(-edge_length / 2, edge_length / 2, N_points)
        X, Y = np.meshgrid(edge, edge)
        stacked = np.vstack([X.ravel(), Y.ravel()])

        return (xy + stacked.T[:, None, :]).reshape(-1, 2)


class Circle(Transformer):
    def __init__(self, radius: float, N_on_radius: int):
        """
        Parameters
        ----------
        radius : float
            Radius of the circle in relative coordinates
        N_on_radius : int
            Number of points to have from the center point (maybe inclusive)
            to the edge (inclusive)
        """
        super().__init__()
        self._radius = radius
        self._N_on_radius = N_on_radius

    @property
    def radius(self) -> float:
        return self._radius

    @radius.setter
    def radius(self, val: float):
        self._radius = val

    @property
    def N_on_radius(self) -> int:
        return self._N_on_radius

    @N_on_radius.setter
    def N_on_radius(self, val: int):
        if not isinstance(val, numbers.Integral) or val <= 0:
            raise TypeError("N_on_radius must be a positive integer")

    def transform(self, xy, radius: float = None, N_on_radius: int = None):
        """
        Make a circle centered on the points.

        Parameters
        ----------
        xy : (N, 2) arraylike
        radius : float, optional
            Radius of the circle in relative coordinates. If *None*, use self.radius
        N_on_radius : int, optional
            Number of points to have from the center point
            to the edge (inclusive). If *None* use self.N_on_radius

        Returns
        -------
        xy (N_new, 2)
            In a circle shape.
        """
        radius = radius or self.radius
        N = N_on_radius or self.N_on_radius

        diam = np.linspace(-radius, radius, N)
        X, Y = np.meshgrid(diam, diam)
        idx = np.sqrt(X**2 + Y**2) <= radius
        stacked = np.vstack([X[idx].ravel(), Y[idx].ravel()])

        return (xy + stacked.T[:, None, :]).reshape(-1, 2)
