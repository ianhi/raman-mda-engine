from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from napari.layers import Points
from pymmcore_plus import CMMCorePlus
from pymmcore_plus.mda import MDAEngine
from useq import MDAEvent

from ._events import QRamanSignaler as RamanSignaler

if TYPE_CHECKING:
    from mda_simulator import ImageGenerator
    from useq import MDASequence


class fakeAcquirer:
    """
    For development
    """

    def collect_spectra_relative(self, points, exposure=20):
        if np.max(np.abs(points)) > 1 or np.min(points) < 0:
            raise ValueError("All points must be between 0 and 1 (inclusive).")
        points = (np.ascontiguousarray(points) - 0.5) * 1.2
        self.collect_spectra_volts(points, exposure)

    def collect_spectra_volts(self, points, exposure=20):
        points = np.ascontiguousarray(points)
        assert points.shape[0] == 2
        arr = np.random.randn(points.shape[1], 1340) * exposure
        return np.cumsum(arr, axis=1)


class RamanEngine(MDAEngine):
    def __init__(
        self,
        mmc: CMMCorePlus = None,
        default_rm_exp: float = 20,
        spectra_collector=None,
        position_idx: int = 1,
    ) -> None:
        """
        Parameters
        ----------
        ...
        position_idx : int, default 1
            Which axis is position for the points layers. Can't assume this
            yet due to the brittleness of broadcastable points
        """
        super().__init__(mmc)
        self.raman_events = RamanSignaler()
        self._rng = np.random.default_rng()
        self._img_gen: ImageGenerator | None = None
        self._default_rm_exp = default_rm_exp
        self.raman_events = RamanSignaler()
        self._spectra_collector = spectra_collector
        if self._spectra_collector is None:
            from raman_control import SpectraCollector

            self._spectra_collector = SpectraCollector.instance()
        self._points_layers: list[Points] = []
        self._pos_idx = position_idx

        # default engine doesn't do this in super to avoid import loops
        self._mmc = CMMCorePlus.instance()

        # need to know image size in order to scale raman aiming points
        if self._img_gen is not None:
            self._img_shape = self._img_gen.img_shape
        else:
            self._img_shape: tuple[int, int] = (
                self._mmc.getImageWidth(),
                self._mmc.getImageHeight(),
            )

    @property
    def points_layers(self) -> list[Points]:
        return self._points_layers

    @points_layers.setter
    def points_layers(self, val: list[Points]):
        self._points_layers = val

    @property
    def default_rm_exposure(self) -> float:
        return self._default_rm_exp

    @default_rm_exposure.setter
    def default_rm_exposure(self, val: float):
        if not isinstance(val, float):
            raise TypeError(
                f"default_rm_exposure must have type of float, got {type(val)}"
            )
        self._default_rm_exp = val

    def _sequence_axis_order(self, seq: MDASequence) -> tuple:
        event = next(seq.iter_events())
        event_axes = list(event.index.keys())
        return tuple(a for a in seq.axis_order if a in event_axes)

    def _event_to_index(self, event: MDAEvent) -> tuple[int, ...]:
        return tuple(event.index[a] for a in self._axis_order)

    def _get_pos_points(self, points: np.ndarray, pos: int):
        return points[points[:, self._pos_idx] == pos][:, -2:]

    def record_raman(self, event: MDAEvent):
        """
        Record and save the raman spectra for the current position and time

        Parameters
        ----------
        event : MDAEvent

        Returns
        -------
        spec : (N, 1340) array of float
        """
        p, t = event.index["p"], event.index.get("t", 0)
        points = []
        which = []
        for layer in self._points_layers:
            # inefficient to always query this
            # but also ensure that we always get the most up to date information
            new_points = self._get_pos_points(layer.data, self._pos_idx)
            points.append(new_points)
            which.extend([layer.name] * len(new_points))
        points = np.concatenate(points).T

        # put into [0, 1] for spectra collector
        points[0, :] /= self._img_shape[0]
        points[1, :] /= self._img_shape[1]

        logger.info(f"collecting raman: {p=}, {t=}")

        spec = self._spectra_collector.collect_spectra_relative(
            points, self._default_rm_exp
        )

        self.raman_events.ramanSpectraReady.emit(event, spec, points, which)

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
                    self.record_raman(event)

            self._mmc.snapImage()
            img = self._mmc.getImage()

            self._events.frameReady.emit(img, event)
        self._finish_run(sequence)
