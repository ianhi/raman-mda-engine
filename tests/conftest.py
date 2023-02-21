from pathlib import Path

import pytest
from pymmcore_plus import CMMCorePlus

from raman_mda_engine import RamanEngine, fakeAcquirer
from raman_mda_engine.aiming import SimpleGridSource


@pytest.fixture
def core() -> CMMCorePlus:
    mmc = CMMCorePlus.instance()
    if len(mmc.getLoadedDevices()) < 2:
        mmc.loadSystemConfiguration(str(Path(__file__).parent / "test-config.cfg"))
    return mmc


@pytest.fixture
def engine(core: CMMCorePlus) -> RamanEngine:
    engine = RamanEngine(spectra_collector=fakeAcquirer())
    engine.aiming_sources.append(SimpleGridSource(5, 5))
    core.register_mda_engine(engine)
    return engine
