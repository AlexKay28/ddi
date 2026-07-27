"""Report renderers registered via side-effect imports.

Every module here contains one or more `@register(name)` calls from `ddi.report`.
Importing this package (done from `ddi/report.py` at module load) ensures every
renderer is available in `list_reports()` / `render()`.
"""

from ddi.reports import phase2_base_vs_instruct  # noqa: F401
from ddi.reports import phase2_bridge  # noqa: F401
from ddi.reports import phase2_pareto  # noqa: F401
from ddi.reports import phase2_quality  # noqa: F401
from ddi.reports import phase3_realization  # noqa: F401
from ddi.reports import phase4_throughput  # noqa: F401
from ddi.reports import phase5_composition  # noqa: F401
from ddi.reports import phase5_hop_split  # noqa: F401
