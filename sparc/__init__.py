import sys

import sparc.src.utils.plotting as _plotting

sys.modules.setdefault("sparc.plotting", _plotting)

try:
    import sparc.src.utils.plotting.chemview as _chemview

    sys.modules.setdefault("sparc.plotting.chemview", _chemview)
except ImportError:
    pass

try:
    import sparc.src.utils.plotting.matplot as _matplot

    sys.modules.setdefault("sparc.plotting.matplot", _matplot)
except ImportError:
    pass

try:
    import sparc.src.utils.plotting.plot_utils as _plot_utils

    sys.modules.setdefault("sparc.plotting.plot_utils", _plot_utils)
except ImportError:
    pass

try:
    import sparc.src.utils.plotting.plotly as _plotly

    sys.modules.setdefault("sparc.plotting.plotly", _plotly)
except ImportError as e:
    import types

    _msg = (
        f"sparc.plotting.plotly requires 'plotly': pip install plotly\n(original: {e})"
    )
    _stub = types.ModuleType("sparc.plotting.plotly")
    _stub.__getattr__ = lambda name, _m=_msg: (_ for _ in ()).throw(ImportError(_m))
    sys.modules.setdefault("sparc.plotting.plotly", _stub)
