"""Check the built training inputs with the ML loaders and report what training will skip.

    python -m pipeline.validate [--data DIR] [--as-of ISO8601]

Uses ml.data and ml.labels directly, so a clean report here means train_breakup_scaler accepts the files.
Writes data/reports/validation.md.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ml.data import build_training_rows, load_events, load_specs
from ml.labels import LEARNED_HEADS, compute_labels
from ml.schema import parse_epoch

from . import config

MIN_ROWS = 30  # train_breakup_scaler's default --min-rows


def validate(data_dir: Path, as_of: datetime) -> str:
    events = load_events(data_dir / config.EVENTS_PATH.name)
    specs = load_specs(data_dir / config.SPECS_PATH.name)
    rows, skipped = build_training_rows(events, specs)
    labels = compute_labels(rows, None, as_of)
    groups = {r.group for r in rows}

    lines = [
        "# Validation report",
        "",
        f"As of {as_of.date()}. {len(events)} events and {len(specs)} spec records read from `{data_dir}`.",
        "",
        f"- **{len(rows)}** events pass `ml.data.build_training_rows`; **{len(skipped)}** are skipped.",
        f"- {len(groups)} distinct CV groups (GroupKFold needs at least 2).",
        "",
        "## Labeled events per head",
        "",
        f"A head is trained only with at least {MIN_ROWS} labeled events. A/M and delta-v heads need "
        "fragments.csv (deferred to v2), so they fall back to neutral bands for now.",
        "",
        "| Head | Labeled | Enough to train |",
        "| --- | --- | --- |",
    ]
    for name in LEARNED_HEADS:
        n = int(np.isfinite(labels[name].to_numpy(float)).sum())
        lines.append(f"| `{name}` | {n} | {'yes' if n >= MIN_ROWS else 'no'} |")

    if skipped:
        lines += ["", "## Skipped events", "", "| Event | Reason |", "| --- | --- |"]
        for event_id, errs in skipped:
            lines.append(f"| {event_id} | {'; '.join(f'{e.loc}: {e.msg}' for e in errs)} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m pipeline.validate", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=config.DATA, help="directory holding the built files")
    p.add_argument("--as-of", help="reference date for the catalog-lag cutoff (default: now)")
    a = p.parse_args(argv)
    as_of = parse_epoch(a.as_of) if a.as_of else datetime.now(timezone.utc)
    report = validate(a.data, as_of)
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    out = config.REPORTS / "validation.md"
    out.write_text(report, encoding="utf-8")
    print(report.split("## Skipped events")[0].rstrip())
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
