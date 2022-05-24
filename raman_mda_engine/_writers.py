from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from napari_broadcastable_points import BroadcastablePoints
from pymmcore_mda_writers import SimpleMultiFileTiffWriter

from ._engine import RamanEngine

if TYPE_CHECKING:
    pass

    from pymmcore_plus import CMMCorePlus
    from pymmcore_plus.mda import PMDAEngine
    from useq import MDASequence

__all__ = [
    "fakeAcquirer",
    "RamanTiffAndNumpyWriter",
]


class fakeAcquirer:
    """
    For development
    """

    def collect_spectra(self, points, exposure=20):
        assert points.shape[0] == 2
        arr = np.random.randn(points.shape[1], 1340) * exposure
        return np.cumsum(arr, axis=1)


class RamanTiffAndNumpyWriter(SimpleMultiFileTiffWriter):
    """
    Base acquirer. Implements the saving logic and interactions with MDA.
    Leaves actual laser control to a subclass to make this easier to develop.
    """

    def __init__(
        self,
        position_index: int,
        save_dir: str | Path,
        points_layers: BroadcastablePoints | list[BroadcastablePoints],
        core: CMMCorePlus = None,
        spectra_collector=None,
    ):
        self._pos_idx = position_index
        if not isinstance(points_layers, list):
            points_layers = [points_layers]
        if not all([isinstance(p, BroadcastablePoints) for p in points_layers]):
            raise TypeError(
                "All layers in points_layers argument must "
                "be BroadcastablePoints layers"
            )
        super().__init__(save_dir, core)

        # BaseWriter.get_unique_folder(self._save_dir, create=True)
        self._points_layers = points_layers
        self._spectra_collector = spectra_collector
        if self._spectra_collector is None:
            from raman_control import SpectraCollector

            self._spectra_collector = SpectraCollector.instance()

    def _on_mda_engine_registered(self, newEngine: PMDAEngine, oldEngine: PMDAEngine):
        super()._on_mda_engine_registered(newEngine, oldEngine)
        if isinstance(oldEngine, RamanEngine):
            oldEngine.raman_events.collectRamanSpectra.disconnect(self.record_raman)
        if isinstance(newEngine, RamanEngine):
            newEngine.raman_events.collectRamanSpectra.connect(self.record_raman)

    def _get_pos_points(self, points: np.ndarray, pos: int):
        return points[points[:, self._pos_idx] == pos][:, -2:]

    def _onMDAStarted(self, sequence: MDASequence):
        super()._onMDAStarted(sequence)
        self._raman_path = self._path / "raman"
        self._raman_path.mkdir()

    def record_raman(self, pos: int, t: int, exposure=500):
        """
        Record and save the raman spectra for the current position and time

        Parameters
        ----------
        pos, t : int
            The position and time indices.
        exposure : int, default 500
            Per point raman exposure in ms.

        Returns
        -------
        spec : Nx1360
        """
        points = []
        which = []
        for layer in self._points_layers:
            # inefficient to always query this
            # but also ensure that we always get the most up to date information
            new_points = self._get_pos_points(layer.data, pos)
            points.append(new_points)
            which.extend([layer.name] * len(new_points))
        points = np.concatenate(points).T

        spec = self._spectra_collector.collect_spectra(points, exposure)

        # TODO zarrify this
        save_name_base = (
            self._raman_path / f"raman_p{str(pos).zfill(3)}_t{str(t).zfill(3)}"
        )
        np.save(str(save_name_base) + "_data.npy", spec)
        np.save(str(save_name_base) + "_locations.npy", points)
        np.save(str(save_name_base) + "_designation.npy", which)
        return spec
