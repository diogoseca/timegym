"""Compatibility entry point for ``timegym replay REPORT [options]``."""

import sys

from timegym.cli import main as timegym_main


def main():
    return timegym_main(["replay", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
