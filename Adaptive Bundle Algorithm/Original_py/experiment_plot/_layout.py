"""Path bootstrap for the Aug-25 subfolder layout (see ../../CODE_MAP.md).

The sources were split into five subfolders on Aug 25, 2026 (user
reorganisation): "Core Engine", "baseline", "objective",
"experiment_plot", "sanity_check".  Modules keep importing each other
by bare name; importing this module first puts every subfolder on
sys.path so those imports keep working.  Identical copies of this file
live in all five subfolders.
"""
import sys
from pathlib import Path

_ORIGINAL_PY = Path(__file__).resolve().parent.parent
for _sub in ("Core Engine", "baseline", "objective",
             "experiment_plot", "sanity_check"):
    _p = str(_ORIGINAL_PY / _sub)
    if _p not in sys.path:
        sys.path.append(_p)
