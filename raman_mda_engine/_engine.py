from __future__ import annotations

from numbers import Real
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from loguru import logger
from pymmcore_plus import CMMCorePlus
from pymmcore_plus.mda import MDAEngine
from useq import MDAEvent

from ._error_handling import slack_notify
from ._events import QRamanSignaler as RamanSignaler
from .aiming import RamanAimingSource, SnappableRamanAimingSource

if TYPE_CHECKING:
    from mda_simulator import ImageGenerator
    from useq import MDASequence


class EventPayload(NamedTuple):
    image: np.ndarray


class fakeAcquirer:
    """For development."""

    def collect_spectra_relative(self, points, exposure=20):
        points = np.asarray(points)
        if points.min() < 0 or points.max() > 1:
            raise ValueError("Points must be in [0, 1]")
        if points.shape[1] != 2 or points.ndim != 2:
            raise ValueError(
                f"volts must have shape (N, 2) but has shape {points.shape}"
            )
        points = (np.ascontiguousarray(points) - 0.5) * 1.2
        return self.collect_spectra_volts(points, exposure)

    def collect_spectra_volts(self, points, exposure=20):
        points = np.ascontiguousarray(points)
        assert points.shape[1] == 2
        arr = np.random.randn(points.shape[0], 1340) * exposure
        return np.cumsum(arr, axis=1)


class RamanEngine(MDAEngine):
    def __init__(
        self,
        mmc: CMMCorePlus = None,
        default_rm_exp: float = 20.0,
        spectra_collector=None,
        sources: list[RamanAimingSource] = None,
    ) -> None:
        """
        A pymmcore-plus mda engine that also collects Raman data.

        Parameters
        ----------
        mmc : CMMCorePlus
            The core to use, or None to use the current instance
        default_rm_exp : float
            The default raman exposure in ms. Used if nothing else provided.
        spectra_collector : SpectraCollector instance
            If None use the default - or nothign if not importable
        sources : iterable
            Collection of aiming sources to aim the raman laser.
        """
        super().__init__(mmc)
        self.raman_events = RamanSignaler()
        self._rng = np.random.default_rng()
        self._img_gen: ImageGenerator | None = None
        self._default_rm_exp = default_rm_exp
        self.raman_events = RamanSignaler()
        self._spectra_collector = spectra_collector
        if self._spectra_collector is None:
            try:
                from raman_control import SpectraCollector

                self._spectra_collector = SpectraCollector.instance()
            except ImportError:
                self._spectra_collector = None
                logger.warning(
                    "Could not import SpectraCollector - No raman collection"
                )

        self._rm_meta = None
        self.aiming_sources = sources if sources is not None else []
        self._sources: list[RamanAimingSource]

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
        return self._default_rm_exp  # type: ignore

    @default_rm_exposure.setter
    def default_rm_exposure(self, val: Real):
        if not isinstance(val, Real):
            raise TypeError(
                f"default_rm_exposure must be a real number, got {type(val)}"
            )
        # ignore typing here because above is the best check
        # but mypy doesn't see float as part of Real so it's a mess
        self._default_rm_exp = val  # type: ignore

    def _sequence_axis_order(self, seq: MDASequence) -> tuple:
        event = next(seq.iter_events())
        event_axes = list(event.index.keys())
        return tuple(a for a in seq.axis_order if a in event_axes)

    def _event_to_index(self, event: MDAEvent) -> tuple[int, ...]:
        return tuple(event.index[a] for a in self._axis_order)

    def record_raman(self, event: MDAEvent):
        """
        Record and save the raman spectra for the current position and time.

        Parameters
        ----------
        event : MDAEvent
            From the mda sequence.

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
        points = np.vstack(points)

        p, t = event.index["p"], event.index.get("t", 0)
        logger.info(f"collecting raman: {p=}, {t=}")

        spec = self._spectra_collector.collect_spectra_relative(
            points, self._default_rm_exp
        )

        self.raman_events.ramanSpectraReady.emit(event, spec, points, which)

    @slack_notify
    def snap_raman(
        self,
        exposure: Real = None,
        aiming_sources: None
        | (SnappableRamanAimingSource | list[SnappableRamanAimingSource]) = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Record raman.

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
        elif not isinstance(aiming_sources, list):
            aiming_sources = [aiming_sources]
        for source in aiming_sources:
            if not isinstance(source, SnappableRamanAimingSource):
                raise TypeError(
                    "All aiming sources must be SnappableRamanAimingSources"
                )
            new_points = source.get_current_points()
            points.append(new_points)
            which.extend([source.name] * len(new_points))

        points = np.vstack(points)

        if exposure is None:
            exposure = self._default_rm_exp  # type: ignore

        spec = self._spectra_collector.collect_spectra_relative(points, exposure)

        return spec, points, which

    @slack_notify
    def setup_sequence(self, sequence: MDASequence) -> None:
        super().setup_sequence(sequence)
        raman_meta = sequence.metadata.get("raman", None)
        if raman_meta:
            if self._spectra_collector is None:
                raise ValueError("Spectra Collector not set - cannot collect Raman.")
            self._rm_channel = raman_meta.get("channel", "BF")

            z = raman_meta.get("z", "all")
            z_index = self._sequence_axis_order(sequence).index("z")
            if isinstance(z, str):
                if z.lower() == "center":
                    n_z = sequence.shape[z_index]
                    if n_z % 2 == 0:
                        raise ValueError("for z=center n_z must be odd.")
                    z = np.array(n_z // 2)
                elif z.lower() in ["all", "stack"]:
                    z = np.arange(sequence.shape[z_index])
            else:
                z = np.asanyarray(z)

            self._rm_z = z
            self._rm_meta = raman_meta

        self._z_rel = sequence.z_plan.positions()
        if "autofocus" in sequence.metadata:
            auto_meta = sequence.metadata["autofocus"]
            self._autofocus = True
            self._auto_device = auto_meta["autofocus_device"]
            self._rel_device = auto_meta["rel_focus_device"]
        else:
            self._autofocus = False

        self._last_pos = -1

    def _run_autofocus(self, event: MDAEvent, pos: int):
        # set to the last known good z for this position
        # to give ourselves the best shot of PFS working
        if pos in self._ref_z:
            self._mmc.setPosition(self._rel_device, self._ref_z[pos])
        self._mmc.waitForSystem()

        # compute new focus
        pfs_z = event.z_pos - self._z_rel
        self._mmc.setPosition(self._auto_device, pfs_z[0])
        self._mmc.fullFocus()
        self._ref_z[pos] = self._mmc.getPosition(self._rel_device)
        self._mmc.enableContinuousFocus(False)
        self._mmc.waitForSystem()

    @slack_notify
    def setup_event(self, event: MDAEvent) -> None:
        if event.x_pos is not None or event.y_pos is not None:
            x = event.x_pos if event.x_pos is not None else self._mmc.getXPosition()
            y = event.y_pos if event.y_pos is not None else self._mmc.getYPosition()
            self._mmc.setXYPosition(x, y)
        if event.channel is not None:
            self._mmc.setConfig(event.channel.group, event.channel.config)

        if event.z_pos is not None:
            if self._autofocus:
                pos = event.index["p"]
                if pos != self._last_pos:
                    self._last_pos = pos
                    # moved to a new position
                    # figure out what the PFS-Offset was
                    try:
                        self.autofocus(event, pos)
                    except RuntimeError:
                        self.autofocus(event, pos)
                z_pos = self._ref_z[pos] + self._z_rel[event.index["z"]]
                self._mmc.setPosition(self._rel_device, z_pos)
            else:
                self._mmc.setPosition(event.z_pos)

        if event.exposure is not None:
            self._mmc.setExposure(event.exposure)

        self._mmc.waitForSystem()

    @slack_notify
    def exec_event(self, event: MDAEvent) -> Any:
        if self._rm_meta:
            if (
                event.channel.config == self._rm_channel
                and event.index["z"] in self._rm_z
            ):
                self.record_raman(event)
        self._mmc.snapImage()
        # TODO: need a return object including the raman channel so that
        # napari-micro can interpret. Currently cannot make raman events
        # bc they mess with the shape of the acquisition for napari-micro
        # and it messes up display.
        return EventPayload(image=self._mmc.getImage())
