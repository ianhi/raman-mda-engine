from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from psygnal import Signal
from pymmcore_plus.mda import MDAEngine
from skimage.draw import disk
from useq import MDAEvent

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus
    from useq import MDASequence


class RamanEvents:
    collectRamanSpectra = Signal(int, int)  # pos, time


class RamanEngine(MDAEngine):
    def __init__(
        self,
        mmc: CMMCorePlus = None,
        img_shape=(512, 512),
        step_size: float = 10,
        radius: float = 30,
        init_spread: float = 100,
    ) -> None:
        super().__init__(mmc)
        self.raman_events = RamanEvents()
        self._rng = np.random.default_rng()
        self._img_shape = np.asarray(img_shape)
        self._last_pos: dict[int, np.ndarray] = {}
        self._step_size = step_size
        self._radius = radius
        self._init_spread = init_spread

    def _sequence_axis_order(self, seq: MDASequence) -> tuple:
        event = next(seq.iter_events())
        event_axes = list(event.index.keys())
        return tuple(a for a in seq.axis_order if a in event_axes)

    def _prepare_to_run(self, sequence: MDASequence):
        self._axis_order = self._sequence_axis_order(sequence)
        P = sequence.shape[self._axis_order.index("p")]
        for p in range(P):
            if p not in self._last_pos:
                Np = self._rng.integers(low=3, high=10)
                self._last_pos[p] = self._rng.normal(
                    self._img_shape / 2, self._init_spread, (Np, 2)
                )

        # for convenience
        self.out = np.zeros((*sequence.shape, *self._img_shape))
        return super()._prepare_to_run(sequence)

    # cache to avoid adavancing the implicit time step when working through
    # the different channels
    @lru_cache
    def _gen_img(self, p, t=None):
        img = np.zeros(self._img_shape, dtype=np.uint8)
        pos = self._last_pos[p]
        pos += self._rng.normal(0, self._step_size, pos.shape)
        self._last_pos[p] = pos
        for p in pos:
            img[disk(p, self._radius, shape=self._img_shape)] = 1
        return img

    def _event_to_index(self, event: MDAEvent) -> tuple[int, ...]:
        return tuple(event.index[a] for a in self._axis_order)

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

            img = self._gen_img(event.index.get("p", None), event.index.get("t", None))
            self.out[self._event_to_index(event)] = img
            # self._mmc.snapImage()
            # img = self._mmc.getImage()

            self._events.frameReady.emit(img, event)
        self._finish_run(sequence)
