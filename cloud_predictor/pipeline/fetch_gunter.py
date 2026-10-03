"""Scrape Gunter's Space Page for the spacecraft that broke up (payload parents only).

    python -m pipeline.fetch_gunter [--refresh]

Matching: the yearly launch chronology (doc_chr/lauYYYY.htm) maps each launch designator to the
spacecraft-type pages of its payloads; with several payloads on one launch, the link whose text best
matches the DISCOS name wins. data/manual/gunter_urls.csv (norad_id,url) overrides the match.
Writes data/raw/gunter/gunter_specs.json: {norad_id: {slug, url, match, fields...}}. Pages are cached
under data/raw/gunter/ and fetched at most once per second.
"""
from __future__ import annotations

import argparse
import difflib
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from bs4 import BeautifulSoup

from . import config, fetch_discos

BASE = "https://space.skyrocket.de"
MIN_NAME_SCORE = 0.5
SPECS_OUT = config.GUNTER_DIR / "gunter_specs.json"
URL_OVERRIDES = config.MANUAL / "gunter_urls.csv"

_last_request = 0.0


def _fetch(url: str, cache: Path, refresh: bool) -> Optional[bytes]:
    global _last_request
    if cache.exists() and not refresh:
        return cache.read_bytes()
    s = config.session()
    time.sleep(max(0.0, 1.0 - (time.monotonic() - _last_request)))
    _last_request = time.monotonic()
    r = s.get(url, timeout=60)
    if r.status_code == 404:
        print(f"  404 {url}")
        return None
    r.raise_for_status()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(r.content)
    return r.content


def _norm(name: str) -> str:
    n = name.lower().replace("cosmos", "kosmos")
    return re.sub(r"[^a-z0-9]", "", n)


def parse_chronology(html: bytes) -> dict[str, list[tuple[str, str]]]:
    """launch designator -> [(payload link text, doc_sdat slug)]."""
    out: dict[str, list[tuple[str, str]]] = {}
    for tr in BeautifulSoup(html, "html.parser").find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 3:
            continue
        launch = tds[0].get_text(strip=True)
        if not re.fullmatch(r"\d{4}-\d{3}", launch):
            continue
        links = []
        for a in tds[2].find_all("a", href=True):
            m = re.search(r"doc_sdat/([^/]+)\.htm", a["href"])
            if m:
                links.append((a.get_text(" ", strip=True), m.group(1)))
        out[launch] = links
    return out


def match_payload(name: str, candidates: list[tuple[str, str]]) -> tuple[Optional[str], str]:
    """Pick the doc_sdat slug for a DISCOS payload name. Returns (slug, how it matched)."""
    slugs = {slug for _, slug in candidates}
    if not slugs:
        return None, "no payload links"
    if len(slugs) == 1:
        return next(iter(slugs)), "only payload type on launch"
    target = _norm(name)
    score, slug = max((difflib.SequenceMatcher(None, target, _norm(text)).ratio(), slug) for text, slug in candidates)
    if score < MIN_NAME_SCORE:
        return None, f"ambiguous ({len(slugs)} payload types, best name score {score:.2f})"
    return slug, f"name match {score:.2f}"


def parse_spacecraft(html: bytes) -> dict[str, str]:
    """The 'satdata' field table of a doc_sdat page, e.g. {'Configuration': 'LM-700A', 'Mass': '689 kg'}."""
    table = BeautifulSoup(html, "html.parser").find(id="satdata")
    fields = {}
    if table is None:
        return fields
    for tr in table.find_all("tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
        if len(cells) >= 2 and cells[0].endswith(":"):
            value = " ".join(cells[1].split())
            if value and value != "?":
                fields[cells[0].rstrip(":").strip()] = value
    return fields


def _url_overrides() -> dict[int, str]:
    if not URL_OVERRIDES.exists():
        return {}
    df = pd.read_csv(URL_OVERRIDES, comment="#")
    return dict(zip(df["norad_id"].astype(int), df["url"]))


def fetch(payloads: list[dict], refresh: bool = False) -> dict[str, dict]:
    """payloads: [{norad_id, name, cospar_id}] -> {norad_id: record}."""
    overrides = _url_overrides()
    chron: dict[int, dict[str, list[tuple[str, str]]]] = {}
    out: dict[str, dict] = {}
    for p in payloads:
        norad, cospar = int(p["norad_id"]), p.get("cospar_id") or ""
        if norad in overrides:
            url = overrides[norad]
            slug, how = re.search(r"([^/]+)\.htm", url).group(1), "manual override"
        else:
            if not re.match(r"\d{4}-\d{3}", cospar):
                out[str(norad)] = {"slug": None, "match": "no COSPAR id"}
                continue
            year = int(cospar[:4])
            if year not in chron:
                html = _fetch(f"{BASE}/doc_chr/lau{year}.htm", config.GUNTER_DIR / "chr" / f"lau{year}.htm", refresh)
                chron[year] = parse_chronology(html) if html else {}
            slug, how = match_payload(p["name"], chron[year].get(cospar[:8], []))
            url = f"{BASE}/doc_sdat/{slug}.htm" if slug else None
        rec: dict = {"slug": slug, "url": url, "match": how}
        if slug:
            html = _fetch(url, config.GUNTER_DIR / "sdat" / f"{slug}.htm", refresh)
            rec.update(parse_spacecraft(html) if html else {"match": f"{how}; page missing"})
        out[str(norad)] = rec
    return out


def payload_parents() -> list[dict]:
    d = fetch_discos.load()
    return [
        {"norad_id": o["satno"], "name": o["name"], "cospar_id": o.get("cosparId")}
        for o in d["objects"].values()
        if o.get("satno") is not None and o.get("objectClass") == "Payload"
    ]


def load(refresh: bool = False) -> dict[str, dict]:
    if refresh or not SPECS_OUT.exists():
        config.write_json(SPECS_OUT, fetch(payload_parents(), refresh))
    return config.read_json(SPECS_OUT)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m pipeline.fetch_gunter", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--refresh", action="store_true", help="re-match and re-download pages")
    a = p.parse_args(argv)
    specs = load(a.refresh)
    matched = sum(1 for r in specs.values() if r.get("slug"))
    print(f"{matched}/{len(specs)} payload parents matched to a Gunter page -> {SPECS_OUT}")


if __name__ == "__main__":
    main()
