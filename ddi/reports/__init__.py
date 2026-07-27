"""Report renderers registered via side-effect imports.

Every module here contains one or more `@register(name)` calls from `ddi.report`.
Importing this package (done from `ddi/report.py` at module load) ensures every
renderer is available in `list_reports()` / `render()`.
"""

from ddi.reports import phase2_bridge  # noqa: F401
