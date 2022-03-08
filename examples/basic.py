from pymmcore_plus import CMMCorePlus
from raman_mda_engine import RamanEngine
from pathlib import Path
from useq import MDASequence

metadata={"raman": 
    {
        "z": "center",
        "channel": "BF",
    },
}
mda = MDASequence(
    metadata=metadata,
    stage_positions=[(100, 100, 30), (200, 150, 35)],
    channels=["BF", "DAPI"],
    time_plan={"interval": 1, "loops": 20},
    z_plan={"range": 4, "step": 0.5},
    axis_order="tpcz",
)
print(mda.axis_order.index('z'))
print(mda.shape)

core = CMMCorePlus.instance()

cfg = Path(__file__).parent.parent / "tests" / "test-config.cfg"
core.loadSystemConfiguration(cfg)

engine = RamanEngine()
core.register_mda_engine(engine)
core.run_mda(mda)
