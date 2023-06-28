from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pymmcore_mda_writers import SimpleMultiFileTiffWriter
from useq import MDAEvent

from ._engine import RamanEngine

if TYPE_CHECKING:
    pass

    from pymmcore_plus import CMMCorePlus
    from pymmcore_plus.mda import PMDAEngine
    from useq import MDASequence

__all__ = [
    "RamanTiffAndNumpyWriter",
]


class RamanTiffAndNumpyWriter(SimpleMultiFileTiffWriter):
    """Writer to save both images and Raman Spectra."""

    def __init__(
        self,
        save_dir: str | Path,
        core: CMMCorePlus = None,
    ):
        super().__init__(save_dir, core)
        if isinstance(self._core.mda.engine, RamanEngine):
            self._core.mda.engine.raman_events.ramanSpectraReady.connect(
                self._save_raman
            )

    def _on_mda_engine_registered(self, newEngine: PMDAEngine, oldEngine: PMDAEngine):
        # super()._on_mda_engine_registered(newEngine, oldEngine)
        if isinstance(oldEngine, RamanEngine):
            oldEngine.raman_events.ramanSpectraReady.disconnect(self._save_raman)
        if isinstance(newEngine, RamanEngine):
            newEngine.raman_events.ramanSpectraReady.connect(self._save_raman)

    def _save_raman(
        self, event: MDAEvent, spectra: np.ndarray, points: np.ndarray, which: list[str]
    ):
        # TODO zarrify this
        pos, t = event.index["p"], event.index.get("t", 0)
        save_name_base = (
            self._raman_path / f"raman_p{str(pos).zfill(3)}_t{str(t).zfill(3)}"
        )
        np.save(str(save_name_base) + "_data.npy", spectra)
        np.save(str(save_name_base) + "_locations.npy", points)
        np.save(str(save_name_base) + "_designation.npy", which)

    def _onMDAStarted(self, sequence: MDASequence):
        super()._onMDAStarted(sequence)
        self._raman_path = self._path / "raman"
        self._raman_path.mkdir()
