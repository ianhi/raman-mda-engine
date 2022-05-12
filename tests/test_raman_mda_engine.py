from pymmcore_plus import CMMCorePlus

from raman_mda_engine import RamanEngine


def test_test_conforms(core: CMMCorePlus):
    engine = RamanEngine()
    core.register_mda_engine(engine)
