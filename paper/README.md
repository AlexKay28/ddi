# paper/ — DDI preprint scaffold

This directory holds the LaTeX scaffold for the DDI preprint. The compact,
tracked `paper/evidence.json` preserves the audited aggregate values behind the
paper and the full SHA-256 and byte size of every raw input. Large historical
JSONL stores remain ignored; they are not needed to render from a clean checkout.
If one is present locally, its hash is verified before rendering. Required
tracked inputs must always be present and match.

From the repository root, the one paper command is:

```sh
./paper/build.sh
```

It validates `evidence.json`, verifies source hashes, regenerates every retained
artifact in `paper/generated/`, and then runs `pdflatex`, `bibtex`, `pdflatex`,
`pdflatex` when both TeX programs are installed. Without `pdflatex`, rendering
still succeeds and PDF compilation is explicitly skipped. This checkout has no
TeX installation, so VIBEQUEUE-50 does not claim a local PDF build result.

The renderer requires only Python 3.11 or newer and the standard library:

```sh
python3 paper/render.py --check
python3 paper/render.py --require-all-sources --check
```

The first command is the clean-checkout drift check. The second is the stronger
local audit and also requires every ignored historical store. A source that is
present but differs in size or SHA-256 always fails either command.

The build exports `TZ=UTC`, `FORCE_SOURCE_DATE=1`, and a default
`SOURCE_DATE_EPOCH=1788725843` (the evidence-audit commit time) before invoking
TeX. The repository does not pin a validated TeX distribution or container
digest, so byte-identical `preprint.pdf` output is not claimed across arbitrary
TeX installations. Required LaTeX packages are `geometry`, `parskip`,
`graphicx`, `xcolor`, `hyperref`, `booktabs`, `multirow`, `amsmath`, `amssymb`,
and `natbib`.

T3-flex was removed rather than preserved as numbers: its cited historical log
is absent, so no SHA-256 provenance can be established. F5 is deterministic TeX
instead of a renderer-dependent PDF binary.
