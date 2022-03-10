from __future__ import annotations
from typing import TYPE_CHECKING
from pymmcore_plus.mda import MDAEngine
from loguru import logger
from psygnal import Signal

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus
    from useq import MDASequence


class RamanEvents:
    collectRamanSpectra = Signal(int, int)  # pos, time


class RamanEngine(MDAEngine):
    def __init__(self, mmc: "CMMCorePlus" = None) -> None:
        super().__init__(mmc)
        self.raman_events = RamanEvents()

    def _sequence_axis_order(self, seq: MDASequence):
        event = next(seq.iter_events())
        event_axes = list(event.index.keys())
        return tuple(a for a in seq.axis_order if a in event_axes)

    def run(self, sequence: MDASequence) -> None:
        """
        Run the multi-dimensional acquistion defined by `sequence`.

        Most users should not use this directly as it will block further
        execution. Instead use ``run_mda`` on CMMCorePlus which will run on
        a thread.

        Parameters
        ----------
        sequence : MDASequence
            The sequence of events to run.
        """
        self._prepare_to_run(sequence)
        raman_meta = sequence.metadata.get("raman", None)
        if raman_meta:
            channel = raman_meta.get("channel", "BF")
            z = raman_meta.get("z", "center")
            z_index = self._sequence_axis_order(sequence).index("z")
            if z == "center":
                n_z = sequence.shape[z_index]
                if n_z % 2 == 0:
                    raise ValueError("for z=center n_z must be odd.")
                z = n_z // 2

        for event in sequence:
            cancelled = self._wait_until_event(event, sequence)

            # If cancelled break out of the loop
            if cancelled:
                break

            logger.info(event)
            self._prep_hardware(event)
            if raman_meta:
                if event.channel.config == channel and event.index["z"] == z:
                    p, t = event.index["p"], event.index.get("t", 0)
                    self.raman_events.collectRamanSpectra.emit(p, t)
                    logger.info(f"collecting raman: {p=}, {t=}, {z=}")

            self._mmc.snapImage()
            img = self._mmc.getImage()

            self._events.frameReady.emit(img, event)
        self._finish_run(sequence)
