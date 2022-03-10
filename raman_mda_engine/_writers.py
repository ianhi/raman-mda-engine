from __future__ import annotations

from pathlib import Path
from pymmcore_mda_writers import SimpleMultiFileTiffWriter
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from napari.layers import Points
    from raman_mda_engine import RamanEngine
    from pymmcore_plus.mda import PMDAEngine
    from typing import Union, List
    from pymmcore_plus import CMMCorePlus
    from useq import MDASequence

__all__ = [
    "RamanTiffAndNumpyWriter",
]


class RamanTiffAndNumpyWriter(SimpleMultiFileTiffWriter):
    """
    Base acquirer. Implements the saving logic and interactions with MDA.
    Leaves actual laser control to a subclass to make this easier to develop.
    """

    def __init__(
        self,
        save_dir: Union[str, Path],
        points_layers: List[Points],
        core: CMMCorePlus = None,
    ):
        if not isinstance(points_layers, list):
            points_layers = list(points_layers)
        if not all([isinstance(p, Points) for p in points_layers]):
            raise TypeError(
                "All layers in points_layers argument must be Points layers"
            )
        super().__init__(save_dir, core)

        # BaseWriter.get_unique_folder(self._save_dir, create=True)
        self._points_layers = points_layers

    def _on_mda_engine_registered(self, newEngine: PMDAEngine, oldEngine: PMDAEngine):
        super()._on_mda_engine_registered(newEngine, oldEngine)
        if isinstance(oldEngine, RamanEngine):
            oldEngine.raman_events.collectRamanSpectra.disconnect(self.record_raman)
        if isinstance(newEngine, RamanEngine):
            newEngine.raman_events.collectRamanSpectra.connect(self.record_raman)

    @staticmethod
    def _get_pos_points(points: np.ndarray, pos: int):
        return points[points[:, 0] == pos][:, -2:]

    def acquire(self, points: np.ndarray, exposure: int, filter_wait: int = 3000):
        """
        Actually acquire the raman images by interacting with laser

        MUST OVERRIDE for actual usage.

        Parameters
        ----------
        points : Nx2 Array
        exposure : int
        filter_wait : int, default 3000
            How long to wait for the autofocus filter to move

        Returns
        -------
        spectra : Nx1360
        """
        points = np.asanyarray(points)
        # time.sleep(filter_wait)
        arr = np.random.randn(points.shape[0], 1360)
        arr = np.cumsum(arr, axis=1)
        # time.sleep(filter_wait)
        return arr

    def _onMDAStarted(self, sequence: MDASequence):
        super()._onMDAStarted(sequence)
        self._raman_path = self._path / "raman"
        self._raman_path.mkdir()

    def record_raman(self, pos: int, t: int):
        """
        Record and save the raman spectra for the current position and time

        Parameters
        ----------
        pos, t : int
            The position and time indices.

        Returns
        -------
        spec : Nx1360
        """
        print(f"here? {pos}, {t}")
        # arr = run_laser(points)
        points = []
        which = []
        for layer in self._points_layers:
            # inefficient to always query this
            # but also ensure that we always get the most up to date information
            new_points = self._get_pos_points(layer.data, pos)
            points.append(new_points)
            which.extend([layer.name] * len(new_points))
        points = np.concatenate(points)
        # print(points.shape)

        spec = self.acquire(points, 50)
        # print(which)

        # TODO zarrify this
        save_name_base = (
            self._raman_path / f"raman_p{str(pos).zfill(3)}_t{str(t).zfill(3)}"
        )
        np.save(str(save_name_base) + "_data.npy", spec)
        np.save(str(save_name_base) + "_locations.npy", points)
        np.save(str(save_name_base) + "_designation.npy", which)
        return spec
