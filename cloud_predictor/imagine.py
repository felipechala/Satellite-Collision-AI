"""Grok Imagine illustration of the parent satellite, built from the scenario parameters.

The image is an artist's impression for the demo page, never simulation output: the prompt is
derived from the BreakupEvent's target spacecraft (class, mass, construction) and the parent
orbit, and sent to xAI's image API (POST https://api.x.ai/v1/images/generations).

Key: XAI_TOKEN from the environment or the repo-root .env. Model: XAI_IMAGE_MODEL
(default grok-imagine-image). app.py serves it as POST /v1/satellite-image.
"""
from __future__ import annotations

import base64
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import httpx

from ml.schema import BreakupEvent

TOKEN_ENV = "XAI_TOKEN"
MODEL_ENV = "XAI_IMAGE_MODEL"
DEFAULT_MODEL = "grok-imagine-image"
API_URL = "https://api.x.ai/v1/images/generations"
ASPECT_RATIO = "4:3"
EARTH_RADIUS_KM = 6378.137
CACHE_SIZE = 32
REPO = Path(__file__).resolve().parents[1]


@dataclass
class GeneratedImage:
    data: bytes
    mime: str
    model: str

    def data_url(self) -> str:
        return f"data:{self.mime};base64,{base64.b64encode(self.data).decode('ascii')}"


def api_key() -> Optional[str]:
    """XAI_TOKEN from the environment, else from the repo-root .env (as pipeline.config.env)."""
    value = os.environ.get(TOKEN_ENV)
    if not value:
        from dotenv import dotenv_values

        value = dotenv_values(REPO / ".env").get(TOKEN_ENV)
    return value or None


def _size(object_class: str, mass_kg: float) -> str:
    if object_class == "rocket_body":
        if mass_kg < 1500:
            return "a small spent rocket upper stage, a cylinder about 2 m across with a single engine nozzle"
        return "a large spent rocket upper stage, a long cylinder several metres across with an engine nozzle"
    if object_class == "debris":
        return "a large tumbling piece of a broken satellite"
    if mass_kg <= 15:
        return "a CubeSat, a shoebox-sized satellite with small deployable solar panels"
    if mass_kg <= 200:
        return "a small satellite about one metre across with two solar panel wings"
    if mass_kg <= 2000:
        return "a car-sized satellite bus with two deployed solar array wings and antennas"
    return "a large satellite bus the size of a truck with long solar array wings, antennas and dishes"


def _orbit(alt_km: Optional[float], inc_deg: Optional[float]) -> str:
    if alt_km is None:
        return "in Earth orbit, Earth's curved horizon below"
    if alt_km < 2000:
        where = f"in low Earth orbit at about {alt_km:,.0f} km altitude, Earth's curved horizon filling the lower frame"
    elif alt_km < 30_000:
        where = f"in medium Earth orbit at about {alt_km:,.0f} km altitude, Earth a large globe in the background"
    else:
        where = "in a high orbit, Earth a small distant globe in the background"
    if inc_deg is not None and alt_km < 30_000 and 75 <= inc_deg <= 105:
        where += ", passing over the polar ice"
    return where


def satellite_prompt(event: BreakupEvent, alt_km: Optional[float] = None,
                     inc_deg: Optional[float] = None) -> str:
    """Image prompt for the event's target spacecraft before the breakup."""
    t = event.target
    mass = t.dry_mass_kg + t.propellant_mass_kg
    details = []
    if t.structure_material == "cfrp":
        details.append("dark carbon-fibre composite panels")
    elif t.structure_material == "aluminum":
        details.append("aluminium honeycomb panels")
    elif t.structure_material == "mixed":
        details.append("a mix of aluminium and carbon-fibre panels")
    if t.mli_fraction is not None and t.mli_fraction >= 0.3:
        details.append("wrapped in crinkled gold multi-layer insulation foil")
    if t.solar_array_area_m2:
        details.append(f"solar arrays totalling about {t.solar_array_area_m2:,.0f} square metres")
    if t.launch_year:
        details.append(f"design typical of {t.launch_year}")
    parts = [f"Photorealistic image of {_size(t.object_class, mass)} (about {mass:,.0f} kg)"]
    if details:
        parts.append(", ".join(details))
    parts.append(_orbit(alt_km, inc_deg))
    parts.append("black space, hard sunlight from one side, intact and undamaged. No text, no labels, no logos")
    return ". ".join(parts) + "."


def altitude_of(orbit: dict) -> tuple[Optional[float], Optional[float]]:
    """(alt_km, inc_deg) from request_schema.check_orbit output (circular or r/v state)."""
    if orbit.get("alt_km") is not None:
        return orbit["alt_km"], orbit.get("inc_deg")
    r = orbit.get("r0_km")
    if r:
        return sum(x * x for x in r) ** 0.5 - EARTH_RADIUS_KM, None
    return None, None


def _mime(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


class XaiImager:
    """Calls xAI's image API; caches by prompt so a page reload does not pay twice."""

    def __init__(self, key: str, model: str = DEFAULT_MODEL, http: Optional[httpx.Client] = None):
        self.key = key
        self.model = model
        self.http = http or httpx.Client(timeout=90.0)
        self.cache: OrderedDict[str, GeneratedImage] = OrderedDict()

    def __call__(self, prompt: str) -> GeneratedImage:
        if prompt in self.cache:
            self.cache.move_to_end(prompt)
            return self.cache[prompt]
        image = self._generate(prompt)
        self.cache[prompt] = image
        if len(self.cache) > CACHE_SIZE:
            self.cache.popitem(last=False)
        return image

    def _generate(self, prompt: str) -> GeneratedImage:
        body = {"model": self.model, "prompt": prompt, "n": 1, "response_format": "b64_json",
                "aspect_ratio": ASPECT_RATIO}
        try:
            r = self.http.post(API_URL, json=body, headers={"Authorization": f"Bearer {self.key}"})
        except httpx.HTTPError as e:
            raise RuntimeError(f"cannot reach xAI: {e}") from e
        if r.status_code >= 300:
            raise RuntimeError(f"xAI image API returned {r.status_code}: {r.text[:300]}")
        item = (r.json().get("data") or [{}])[0]
        if item.get("b64_json"):
            data = base64.b64decode(item["b64_json"])
        elif item.get("url"):
            try:
                img = self.http.get(item["url"])
                img.raise_for_status()
            except httpx.HTTPError as e:
                raise RuntimeError(f"cannot download the generated image: {e}") from e
            data = img.content
        else:
            raise RuntimeError("xAI image API returned no image")
        return GeneratedImage(data, _mime(data), self.model)


def imager_from_env() -> Optional[Callable[[str], GeneratedImage]]:
    key = api_key()
    return XaiImager(key, os.environ.get(MODEL_ENV, DEFAULT_MODEL)) if key else None
