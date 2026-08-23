PY3_PROGRAM()

PY_MAIN(vh3.cli:main)

PEERDIR(
    junk/alexkay28/ddi/nirvana/operations
    junk/alexkay28/ddi/nirvana/graphs
)

INCLUDE(${ARCADIA_ROOT}/nirvana/vh3/add_conf.inc)

RESOURCE(
    junk/alexkay28/ddi/nirvana/bin/conf.yaml conf.yaml
)

END()
