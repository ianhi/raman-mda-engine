from __future__ import annotations

from typing import List

import numpy as np
from psygnal import Signal as Psygnal
from qtpy.QtCore import QObject, Signal
from useq import MDAEvent

__all__ = [
    "RamanSignaler",
    "QRamanSignaler",
]


class RamanSignaler:
    ramanSpectraReady = Psygnal(MDAEvent, np.ndarray, np.ndarray, List[str])


class QRamanSignaler(QObject):
    ramanSpectraReady = Signal(object, object, object, object)
