"""Space-Track SATCAT: launch years and debris counts by international designator.

    python -m pipeline.satcat [--refresh]

Reads data/raw/satcat.csv (a Space-Track `class/satcat` CSV export). --refresh re-downloads it, which
needs SPACETRACK_USER and SPACETRACK_PASS in the environment or the repo-root .env.
"""
from __future__ import annotations

import argparse

import pandas as pd

from . import config

LOGIN_URL = "https://www.space-track.org/ajaxauth/login"
QUERY_URL = "https://www.space-track.org/basicspacedata/query/class/satcat/orderby/NORAD_CAT_ID%20asc/format/csv"


def download() -> None:
    user, password = config.env("SPACETRACK_USER"), config.env("SPACETRACK_PASS")
    if not (user and password):
        raise SystemExit("SPACETRACK_USER / SPACETRACK_PASS are not set (put them in the repo-root .env)")
    s = config.session()
    s.post(LOGIN_URL, data={"identity": user, "password": password}, timeout=60).raise_for_status()
    r = config.get(s, QUERY_URL)
    config.SATCAT_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.SATCAT_PATH.write_bytes(r.content)


def load(refresh: bool = False) -> pd.DataFrame:
    if refresh or not config.SATCAT_PATH.exists():
        download()
    df = pd.read_csv(config.SATCAT_PATH, dtype={"INTLDES": str}, low_memory=False)
    df["prefix"] = df["INTLDES"].str[:8]
    df["launch_year"] = pd.to_datetime(df["LAUNCH"], errors="coerce").dt.year
    return df


def launch_years(satcat: pd.DataFrame) -> dict[int, int]:
    s = satcat.dropna(subset=["launch_year"])
    return dict(zip(s["NORAD_CAT_ID"].astype(int), s["launch_year"].astype(int)))


def debris_by_prefix(satcat: pd.DataFrame) -> dict[str, int]:
    """Cataloged DEBRIS pieces (decayed included) per launch designator, e.g. '1999-025'."""
    return satcat[satcat["OBJECT_TYPE"] == "DEBRIS"].groupby("prefix").size().to_dict()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m pipeline.satcat", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--refresh", action="store_true", help="re-download from Space-Track")
    a = p.parse_args(argv)
    df = load(a.refresh)
    print(f"{len(df)} SATCAT objects, {int((df['OBJECT_TYPE'] == 'DEBRIS').sum())} debris -> {config.SATCAT_PATH}")


if __name__ == "__main__":
    main()
