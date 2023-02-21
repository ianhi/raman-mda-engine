from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from napari_micromanager import MainWindow
    from useq import MDASequence

__all__ = ["get_seq_from_napari"]


def get_seq_from_napari(
    main_window: MainWindow,
    autofocus_device: str = "PFS-Offset",
    rel_focus_device: str = "Focus",
) -> MDASequence:
    """Get the MDA sequence from the MDA widget.

    Also adds the approriate autofocus metadata

    Parameters
    ----------
    main_window : napari_micromanager.MainWindow
        The plugin main widget.
    autofocus_device : str
        The device name for autofocusing. e.g. `PFS-Offset` for TE-2000
    rel_focus_device : str
        The Z-stage to use for relative steps.

    Returns
    -------
    MDASequence
    """
    mda_dock = main_window._dock_widgets["MDA"]
    seq = mda_dock.children()[4].get_state()
    new_metadata = dict(seq.metadata)
    new_metadata["autofocus"] = {
        "autofocus_device": autofocus_device,
        "rel_focus_device": rel_focus_device,
    }
    return seq.replace(metadata=new_metadata)
