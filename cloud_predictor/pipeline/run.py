"""Run the whole data pipeline: fetch -> build -> validate -> EDA.

    python -m pipeline.run [--refresh] [--as-of ISO8601]

Raw downloads are cached under data/raw/, so without --refresh this runs offline.
"""
from __future__ import annotations

import argparse

from . import build_events, build_specs, eda, fetch_discos, fetch_gunter, satcat, validate


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m pipeline.run", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--refresh", action="store_true", help="re-download DISCOS, SATCAT and Gunter pages")
    p.add_argument("--as-of", help="reference date for the catalog-lag cutoff (default: now)")
    a = p.parse_args(argv)
    refresh = ["--refresh"] if a.refresh else []
    as_of = ["--as-of", a.as_of] if a.as_of else []

    steps = [
        ("DISCOS", fetch_discos.main, refresh),
        ("SATCAT", satcat.main, refresh),
        ("events", build_events.main, []),
        ("Gunter", fetch_gunter.main, refresh),
        ("specs", build_specs.main, []),
        ("validate", validate.main, as_of),
        ("EDA", eda.main, as_of),
    ]
    for name, step, args in steps:
        print(f"== {name}")
        step(args)


if __name__ == "__main__":
    main()
