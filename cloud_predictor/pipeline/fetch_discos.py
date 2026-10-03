"""Download every ESA DISCOS fragmentation event with its involved objects.

    python -m pipeline.fetch_discos [--refresh]

Needs DISCOS_TOKEN in the environment or the repo-root .env. Writes data/raw/discos/fragmentations.json
as {"fragmentations": [...], "objects": {discos_object_id: attributes}}. Cached: reruns are offline
unless --refresh is given.
"""
from __future__ import annotations

import argparse

import pandas as pd

from . import config

API = "https://discosweb.esoc.esa.int"
PAGE_SIZE = 100
FIXES_PATH = config.MANUAL / "discos_object_fixes.csv"


def fetch() -> dict:
    token = config.env("DISCOS_TOKEN")
    if not token:
        raise SystemExit("DISCOS_TOKEN is not set (put it in the repo-root .env)")
    s = config.session({"Authorization": f"Bearer {token}", "DiscosWeb-Api-Version": "2"})

    fragmentations: list[dict] = []
    objects: dict[str, dict] = {}
    url, params = f"{API}/api/fragmentations", {"page[size]": PAGE_SIZE, "include": "objects", "sort": "epoch"}
    while url:
        body = config.get(s, url, params).json()
        for item in body["data"]:
            fragmentations.append({
                "id": item["id"],
                **item["attributes"],
                "object_ids": [o["id"] for o in item["relationships"]["objects"]["data"]],
            })
        for inc in body.get("included", []):
            if inc["type"] == "object":
                objects[inc["id"]] = inc["attributes"]
        nxt = body["links"].get("next")
        url, params = (f"{API}{nxt}" if nxt else None), None
        print(f"  {len(fragmentations)} fragmentations, {len(objects)} objects")
    return {"fragmentations": fragmentations, "objects": objects}


def apply_fixes(data: dict) -> dict:
    """Fill object attributes DISCOS lacks (e.g. a missing satno) from data/manual/discos_object_fixes.csv."""
    if not FIXES_PATH.exists():
        return data
    for r in pd.read_csv(FIXES_PATH, dtype=str).to_dict(orient="records"):
        obj = data["objects"].get(r["discos_object_id"])
        if obj is None:
            print(f"  WARNING {FIXES_PATH.name}: unknown DISCOS object {r['discos_object_id']}")
            continue
        if pd.notna(r.get("satno")):
            obj["satno"] = int(r["satno"])
        if pd.notna(r.get("cosparId")):
            obj["cosparId"] = r["cosparId"]
    return data


def load(refresh: bool = False) -> dict:
    if refresh or not config.DISCOS_PATH.exists():
        config.write_json(config.DISCOS_PATH, fetch())
    return apply_fixes(config.read_json(config.DISCOS_PATH))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m pipeline.fetch_discos", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--refresh", action="store_true", help="re-download even if cached")
    a = p.parse_args(argv)
    d = load(a.refresh)
    print(f"{len(d['fragmentations'])} fragmentations, {len(d['objects'])} objects -> {config.DISCOS_PATH}")


if __name__ == "__main__":
    main()
