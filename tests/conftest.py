from pathlib import Path

import pytest
from pymmcore_plus import CMMCorePlus


@pytest.fixture
def core() -> CMMCorePlus:
    mmc = CMMCorePlus.instance()
    if len(mmc.getLoadedDevices()) < 2:
        mmc.loadSystemConfiguration(str(Path(__file__).parent / "test-config.cfg"))
    return mmc
