"""Shared paths, credentials and HTTP helpers for the data pipeline."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import requests
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]  # cloud_predictor/
REPO = ROOT.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
MANUAL = DATA / "manual"
REPORTS = DATA / "reports"

SPECS_PATH = DATA / "gunter_satellites.json"
EVENTS_PATH = DATA / "historical_breakups.csv"
SATCAT_PATH = RAW / "satcat.csv"
DISCOS_PATH = RAW / "discos" / "fragmentations.json"
GUNTER_DIR = RAW / "gunter"
PROVENANCE_PATH = REPORTS / "event_provenance.csv"

USER_AGENT = "Satellite-Collision-AI data pipeline (research; hackathon)"


def env(name: str) -> Optional[str]:
    """Process environment first, then the repo-root .env file."""
    value = os.environ.get(name) or dotenv_values(REPO / ".env").get(name)
    return value or None


def session(headers: Optional[dict[str, str]] = None) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, **(headers or {})})
    return s


def get(s: requests.Session, url: str, params: Optional[dict] = None, max_tries: int = 6) -> requests.Response:
    """GET with backoff on 429 (honours Retry-After) and 5xx; raises on other errors."""
    for attempt in range(max_tries):
        r = s.get(url, params=params, timeout=60)
        if r.status_code == 429 or r.status_code >= 500:
            wait = float(r.headers.get("Retry-After") or 2 ** attempt)
            print(f"  {r.status_code} from {url}; retrying in {wait:.0f}s")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r
    raise RuntimeError(f"gave up on {url} after {max_tries} tries")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
