import numpy as np
from scipy.spatial.distance import cdist

from raman_mda_engine.aiming.transformers import Circle


def test_ordering():
    # points should come in groups, rather than interleaved in a crazy way
    radius = 1
    transformer = Circle(radius, 5)
    init_points = np.vstack([np.arange(5), np.arange(5)]).T * 10
    out = transformer.transform(init_points)
    # FIXME: add parameterize for other transformer types
    distances = cdist(out[: transformer.multiplier], out[: transformer.multiplier])
    assert distances.max() <= radius * 2
