from __future__ import annotations
from typing import TYPE_CHECKING
from pymmcore_plus.mda import MDAEngine
from loguru import logger

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus
    from useq import MDASequence

class RamanEngine(MDAEngine):
    def __init__(self, mmc: "CMMCorePlus" = None) -> None:
        super().__init__(mmc)
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
        raman_meta = sequence.metadata["raman"]
        channel = raman_meta.get("channel", "BF")
        z = raman_meta.get("z", "center")
        z_index = sequence.axis_order.index('z')
        if z == 'center':
            n_z = sequence.shape[z_index]
            if n_z % 2 == 0:
                raise ValueError("for z=center n_z must be odd.")
            z = n_z //2 


        for event in sequence:
            cancelled = self._wait_until_event(event, sequence)

            # If cancelled break out of the loop
            if cancelled:
                break

            logger.info(event)
            self._prep_hardware(event)
            if event.channel.config == channel and event.index['z'] == z:
                logger.debug("raman time!")

            self._mmc.snapImage()
            img = self._mmc.getImage()

            self._events.frameReady.emit(img, event)
        self._finish_run(sequence)
