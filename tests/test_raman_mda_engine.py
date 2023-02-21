from unittest.mock import MagicMock

import pytest
from pymmcore_plus import CMMCorePlus
from useq import MDASequence

from raman_mda_engine import RamanEngine
from raman_mda_engine.aiming import SimpleGridSource, SnappableRamanAimingSource


def test_snappable():
    grid = SimpleGridSource(5, 5)
    assert isinstance(grid, SnappableRamanAimingSource)


def test_mda_no_autofocus(core: CMMCorePlus, engine: RamanEngine):
    seq = MDASequence(
        metadata={"raman": {"z": "center"}},
        channels=["BF"],
        time_plan={"interval": 1, "loops": 3},
        # z_plan={"range": 30, "step": 10},
        z_plan={"relative": [-15, 0, 15]},
        axis_order="tpcz",
        stage_positions=[(0, 1, 1), (512, 128, 0)],
    )

    rm_mock = MagicMock()
    engine.raman_events.ramanSpectraReady.connect(rm_mock)
    core.mda.run(seq)
    assert rm_mock.call_count == 6

    rm_mock.reset_mock()
    seq = seq.replace(metadata={"raman": {"z": "all"}})
    core.mda.run(seq)
    assert rm_mock.call_count == 18

    # remove aiming source and check that we raise an error
    engine.aiming_sources = []
    with pytest.raises(RuntimeError, match="No aiming sources - cannot collect Raman."):
        core.mda.run(seq)


# TODO: test with autofocus!!
