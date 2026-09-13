PY3_LIBRARY()

PY_SRCS(
    __init__.py
    scripts.py
    pydl.py
    ddi_pipeline.py
    ddi_rows.py
    ddi_diagnose.py
)

PEERDIR(
    nirvana/vh3/src
    junk/alexkay28/ddi/nirvana/common
    junk/alexkay28/ddi/nirvana/operations
)

END()
