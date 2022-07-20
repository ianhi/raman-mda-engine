from __future__ import annotations

from numbers import Real
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from pymmcore_plus import CMMCorePlus
from pymmcore_plus.mda import MDAEngine
from useq import MDAEvent

from ._events import QRamanSignaler as RamanSignaler
from .aiming import RamanAimingSource, SnappableRamanAimingSource

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
        return self.collect_spectra_volts(points, exposure)

    def collect_spectra_volts(self, points, exposure=20):
        points = np.ascontiguousarray(points)
        assert points.shape[0] == 2
        arr = np.random.randn(points.shape[1], 1340) * exposure
        return np.cumsum(arr, axis=1)


class RamanEngine(MDAEngine):
    def __init__(
        self,
        mmc: CMMCorePlus = None,
        default_rm_exp: Real = 20,
        spectra_collector=None,
        sources: list[RamanAimingSource] = None,
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
        self.aiming_sources = sources if sources is not None else []

        # default engine doesn't do this in super to avoid import loops
        self._mmc = CMMCorePlus.instance()

    @property
    def aiming_sources(self) -> list[RamanAimingSource]:
        return self._sources

    @aiming_sources.setter
    def aiming_sources(self, val: list[RamanAimingSource]):
        if val is None:
            self._sources = []
        elif all([isinstance(source, RamanAimingSource) for source in val]):
            self._sources = list(val)
        else:
            raise TypeError(
                "aiming_sources must be a list of objects"
                " conforming to the RamanAimingSource protocol."
            )

    @property
    def default_rm_exposure(self) -> Real:
        return self._default_rm_exp

    @default_rm_exposure.setter
    def default_rm_exposure(self, val: Real):
        if not isinstance(val, Real):
            raise TypeError(
                f"default_rm_exposure must be a real number, got {type(val)}"
            )
        self._default_rm_exp = val

    def _sequence_axis_order(self, seq: MDASequence) -> tuple:
        event = next(seq.iter_events())
        event_axes = list(event.index.keys())
        return tuple(a for a in seq.axis_order if a in event_axes)

    def _event_to_index(self, event: MDAEvent) -> tuple[int, ...]:
        return tuple(event.index[a] for a in self._axis_order)

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
        points = []
        which = []
        for source in self.aiming_sources:
            new_points = source.get_mda_points(event)
            points.append(new_points)
            which.extend([source.name] * len(new_points))
        points = np.hstack(points)

        p, t = event.index["p"], event.index.get("t", 0)
        logger.info(f"collecting raman: {p=}, {t=}")

        spec = self._spectra_collector.collect_spectra_relative(
            points, self._default_rm_exp
        )

        self.raman_events.ramanSpectraReady.emit(event, spec, points, which)

    def snap_raman(
        self,
        exposure: Real = None,
        aiming_sources: (
            SnappableRamanAimingSource | list[SnappableRamanAimingSource]
        ) = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Record raman

        Parameters
        ----------
        exposure : real, optional
            The exposure time to use, defaults to the *default_rm_exposure*
        aiming_sources : list[SnappableAimingSource]
            The aiming sources to use

        Returns
        -------
        spec : (N, 1340) np.ndarray
        points : (N, 2) absolute positions in image space where laser was aimed
        which : (N,) label (e.g. 'cell' or 'bkd') for each point
        """
        points = []
        which = []
        if aiming_sources is None:
            aiming_sources = [
                source
                for source in self.aiming_sources
                if isinstance(source, SnappableRamanAimingSource)
            ]
        for source in aiming_sources:
            if not isinstance(source, SnappableRamanAimingSource):
                raise TypeError(
                    "All aiming sources must be SnappableRamanAimingSources"
                )
            new_points = source.get_current_points()
            points.append(new_points)
            which.extend([source.name] * len(new_points))

        points = np.hstack(points)

        if exposure is None:
            exposure = self._default_rm_exp

        spec = self._spectra_collector.collect_spectra_relative(points, exposure)

        return spec, points, which

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
