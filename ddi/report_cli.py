"""Tiny CLI over `ddi.report`.

    python -m ddi.report_cli --list
    python -m ddi.report_cli --name T2
    python -m ddi.report_cli --name F2 --output paper/generated/F2.pdf

Also invoked via the root Makefile as `make report NAME=T2` / `make report LIST=1`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ddi.report import list_reports, render


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ddi-report")
    parser.add_argument("--list", action="store_true", help="list registered reports and exit")
    parser.add_argument("--name", help="report name (T*, F*)")
    parser.add_argument("--output", type=Path, help="output path (required for figures)")
    args = parser.parse_args(argv)

    if args.list:
        for name in list_reports():
            print(name)
        return 0

    if not args.name:
        parser.error("--name is required unless --list is used")

    result = render(args.name, output=args.output)
    if result is not None:
        print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
