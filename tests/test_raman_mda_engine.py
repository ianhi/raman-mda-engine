from raman_mda_engine.aiming import SimpleGridSource, SnappableRamanAimingSource

# def test_test_conforms(core: CMMCorePlus):
#     engine = RamanEngine()
#     core.register_mda_engine(engine)


def test_snappable():
    grid = SimpleGridSource(5, 5)
    assert isinstance(grid, SnappableRamanAimingSource)
