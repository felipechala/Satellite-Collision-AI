"""Exploratory analysis of the training inputs, to read before training.

    python -m pipeline.eda [--data DIR] [--out DIR] [--as-of ISO8601]

Reads historical_breakups.csv and gunter_satellites.json from --data (default data/), plus the pipeline's
provenance files and raw DISCOS/SATCAT caches when they exist. Writes eda.md and figures/*.png to --out
(default data/reports/). Works on ml.synthetic output too, minus the provenance sections.
"""
from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ml import sbm  # noqa: E402
from ml.contract import BOUNDS  # noqa: E402
from ml.data import build_training_rows, load_events, load_specs  # noqa: E402
from ml.labels import CATALOG_LAG, count_label  # noqa: E402
from ml.schema import parse_epoch  # noqa: E402

from . import config  # noqa: E402

# Reference palette (dataviz skill, light mode): categorical slots in fixed order, chart chrome.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
EVENT_TYPE_ORDER = ["explosion", "collision"]
OBJECT_CLASS_ORDER = ["payload", "rocket_body", "debris"]
CAUSE_ORDER = ["propulsion", "battery", "deliberate", "unknown"]
SPEC_COLUMNS = ["dry_mass_kg", "propellant_mass_kg", "structure_material", "solar_array_area_m2",
                "bus_volume_m3", "mli_fraction", "launch_year", "bus_family"]
LN_BOUNDS = tuple(math.log(b) for b in BOUNDS["n_multiplier"])


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 12,
        "axes.titlelocation": "left", "axes.edgecolor": AXIS, "axes.linewidth": 1,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "legend.frameon": False, "legend.labelcolor": INK_2,
    })


def _save(fig, out: Path, name: str) -> str:
    path = out / "figures" / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return f"figures/{name}.png"


def _table(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> list[str]:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if v is None or (isinstance(v, float) and math.isnan(v)):
                cells.append("—")
            elif isinstance(v, float):
                cells.append(floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def event_frame(data_dir: Path, as_of: datetime) -> tuple[pd.DataFrame, list, pd.DataFrame]:
    """One row per valid training event with features of interest and the n_multiplier label."""
    events = load_events(data_dir / config.EVENTS_PATH.name)
    specs = load_specs(data_dir / config.SPECS_PATH.name)
    rows, skipped = build_training_rows(events, specs)
    recs = []
    for r in rows:
        e, t = r.event, r.event.target
        d = sbm.derive(e)
        n_sbm = float(sbm.n_cum_sbm(sbm.LC_ANCHOR_M, e))
        epoch = parse_epoch(e.epoch)
        recs.append({
            "event_id": r.event_id, "epoch": epoch, "year": epoch.year, "event_type": e.event_type,
            "explosion_cause": e.explosion_cause, "norad_id": t.norad_id, "object_class": t.object_class,
            "total_mass_kg": t.dry_mass_kg + t.propellant_mass_kg, "n_cataloged": r.n_cataloged, "n_sbm": n_sbm,
            "label": count_label(r, as_of), "in_catalog_lag": epoch > as_of - CATALOG_LAG,
            "emr_j_per_g": d.emr_j_per_g, "is_catastrophic": d.is_catastrophic, "group": r.group,
        })
    return pd.DataFrame(recs), skipped, events


def _decade_chart(df: pd.DataFrame, out: Path) -> str:
    decades = (df["year"] // 10 * 10).astype(int)
    counts = pd.crosstab(decades, df["event_type"]).reindex(columns=EVENT_TYPE_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    x = np.arange(len(counts))
    bottom = np.zeros(len(counts))
    for color, col in zip(SERIES, EVENT_TYPE_ORDER):
        ax.bar(x, counts[col], width=0.55, bottom=bottom, color=color, edgecolor=SURFACE, linewidth=1.5, label=col)
        bottom += counts[col].to_numpy()
    for xi, total in zip(x, bottom):
        ax.text(xi, total + 1.5, f"{int(total)}", ha="center", va="bottom", color=INK_2, fontsize=9)
    ax.set_xticks(x, [f"{d}s" for d in counts.index])
    ax.set_ylim(0, bottom.max() * 1.3)
    ax.set_ylabel("Breakup events")
    ax.grid(axis="x", visible=False)
    ax.set_title("Breakup events per decade")
    ax.legend(loc="upper right", ncols=2)
    return _save(fig, out, "events_per_decade")


def _label_hist(lab: pd.DataFrame, out: Path) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    bins = np.arange(math.floor(lab["label"].min()), math.ceil(lab["label"].max()) + 0.5, 0.5)
    data = [lab.loc[lab["event_type"] == t, "label"] for t in EVENT_TYPE_ORDER]
    ax.hist(data, bins=bins, stacked=True, color=SERIES[:2], edgecolor=SURFACE, linewidth=1.5,
            label=EVENT_TYPE_ORDER)
    ax.axvspan(*LN_BOUNDS, color=GRID, alpha=0.5, zorder=0, lw=0)
    ax.axvline(0, color=INK_2, lw=1)
    ymax = ax.get_ylim()[1]
    ax.text(0.05, ymax * 0.95, "SBM as-is", color=INK_2, fontsize=9, va="top")
    lo, hi = BOUNDS["n_multiplier"]
    ax.text(0.1, ymax * 0.7, f"allowed n_multiplier\nrange ({lo:g}–{hi:g}×)", color=MUTED, fontsize=8, va="top")
    ax.set_xlabel("ln(cataloged pieces ≥10 cm / SBM prediction)  —  the n_multiplier label")
    ax.set_ylabel("Events")
    ax.grid(axis="x", visible=False)
    ax.set_title("How far real breakups sit from the NASA SBM count")
    ax.legend(loc="upper center")
    return _save(fig, out, "n_multiplier_label_hist")


def _label_vs_mass(lab: pd.DataFrame, out: Path) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for color, cls in zip(SERIES, OBJECT_CLASS_ORDER):
        s = lab[lab["object_class"] == cls]
        if len(s):
            ax.scatter(np.log10(s["total_mass_kg"]), s["label"], s=36, color=color, edgecolor=SURFACE,
                       linewidth=1, alpha=0.85, label=f"{cls} ({len(s)})")
    ax.axhspan(*LN_BOUNDS, color=GRID, alpha=0.5, zorder=0, lw=0)
    ax.axhline(0, color=INK_2, lw=1)
    ax.set_xlabel("log10 parent mass (kg)")
    ax.set_ylabel("n_multiplier label (ln)")
    ax.set_title("Count label vs parent mass (the SBM explosion law has no mass term)")
    ax.legend(loc="upper left")
    return _save(fig, out, "n_multiplier_label_vs_mass")


def _label_by_cause(lab: pd.DataFrame, out: Path) -> str:
    groups = [(c, lab.loc[(lab["event_type"] == "explosion") & (lab["explosion_cause"] == c), "label"])
              for c in CAUSE_ORDER]
    groups.append(("collision", lab.loc[lab["event_type"] == "collision", "label"]))
    groups = [(n, s) for n, s in groups if len(s)]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    rng = np.random.default_rng(0)
    for i, (name, s) in enumerate(groups):
        color = SERIES[1] if name == "collision" else SERIES[0]
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(s)), s, s=30, color=color, edgecolor=SURFACE, linewidth=1, alpha=0.8)
        med = float(np.median(s))
        ax.plot([i - 0.3, i + 0.3], [med, med], color=INK, lw=2, solid_capstyle="round")
        ax.text(i + 0.33, med, f"{med:+.1f}", va="center", color=INK, fontsize=9)
    ax.axhspan(*LN_BOUNDS, color=GRID, alpha=0.5, zorder=0, lw=0)
    ax.axhline(0, color=INK_2, lw=1)
    ax.set_xticks(range(len(groups)), [f"{n}\n(n={len(s)})" for n, s in groups])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("n_multiplier label (ln)")
    ax.set_title("Count label by cause (black bar = median)")
    return _save(fig, out, "n_multiplier_label_by_cause")


def _optional_csv(path: Path) -> Optional[pd.DataFrame]:
    return pd.read_csv(path) if path.exists() else None


def _undiscovered_debris_launches(min_debris: int = 50) -> Optional[pd.DataFrame]:
    """Launches with many cataloged debris pieces but no DISCOS event on any of their objects."""
    if not (config.SATCAT_PATH.exists() and config.DISCOS_PATH.exists()):
        return None
    from . import fetch_discos, satcat
    sc = satcat.load()
    covered = {o["cosparId"][:8] for o in fetch_discos.load()["objects"].values() if o.get("cosparId")}
    deb = sc[sc["OBJECT_TYPE"] == "DEBRIS"].groupby("prefix").size()
    deb = deb[(deb >= min_debris) & ~deb.index.isin(covered)].sort_values(ascending=False)
    names = sc[sc["OBJECT_TYPE"] != "DEBRIS"].groupby("prefix")["SATNAME"].agg(lambda s: ", ".join(s.head(3)))
    return pd.DataFrame({"launch": deb.index, "debris_pieces": deb.to_numpy(),
                         "objects_on_launch": names.reindex(deb.index).to_numpy()})


def report(data_dir: Path, out: Path, as_of: datetime) -> str:
    _style()
    df, skipped, events = event_frame(data_dir, as_of)
    specs = pd.DataFrame(load_specs(data_dir / config.SPECS_PATH.name).values())
    lab = df[np.isfinite(df["label"])]
    L = [
        "# Training data EDA",
        "",
        f"As of {as_of.date()}, data from `{data_dir}`. Generated by `python -m pipeline.eda`.",
        "",
        "## Summary",
        "",
        f"- {len(events)} events in the CSV; **{len(df)} valid** after joining specs, {len(skipped)} skipped "
        "(see validation.md).",
        f"- **{len(lab)} events carry an `n_multiplier` label** (the fragment-count correction). The other "
        f"{len(df) - len(lab)} have no label: "
        f"{int(df['n_cataloged'].isna().sum())} have an ambiguous count (parent broke up more than once), "
        f"{int((df['n_cataloged'] == 0).sum())} have zero cataloged pieces, and "
        f"{int((df['in_catalog_lag'] & df['n_cataloged'].gt(0)).sum())} are under a year old (catalog still filling).",
        f"- Explosions: {int((df['event_type'] == 'explosion').sum())}, collisions: "
        f"{int((df['event_type'] == 'collision').sum())}. {df['group'].nunique()} CV groups.",
    ]
    if len(lab):
        inside = lab["label"].between(*LN_BOUNDS).mean()
        L.append(f"- Median label **{lab['label'].median():+.2f}** (×{math.exp(lab['label'].median()):.2f} of the "
                 f"SBM count); {inside:.0%} of labels fall inside the allowed n_multiplier range "
                 f"{BOUNDS['n_multiplier'][0]:g}–{BOUNDS['n_multiplier'][1]:g}×.")

    L += ["", "## Events", "", f"![Events per decade]({_decade_chart(df, out)})", ""]
    by_cause = (df.assign(explosion_cause=df["explosion_cause"].fillna("—"))
                .groupby(["event_type", "explosion_cause", "object_class"]).size().unstack(fill_value=0).reset_index())
    L += _table(by_cause) + [""]

    L += ["## Spec field coverage", "",
          "Share of valid events whose parent record has the field. Missing values are imputed by the model "
          "(the neural net heads carry missing indicators and a missing bucket per categorical); a parent with "
          "no construction info at all (material, panels, volume, MLI, bus family) gets the neutral fallback band.", ""]
    targets = specs.set_index("norad_id").reindex(df["norad_id"].unique())
    cov = pd.DataFrame({"field": SPEC_COLUMNS,
                        "coverage": [targets[c].notna().mean() if c in targets else 0.0 for c in SPEC_COLUMNS]})
    if "structure_material" in targets:
        cov.loc[cov["field"] == "structure_material", "coverage"] = (
            targets["structure_material"].fillna("unknown") != "unknown").mean()
    cov["coverage"] = cov["coverage"].map(lambda v: f"{v:.0%}")
    L += _table(cov) + [""]

    if len(lab):
        L += ["## Fragment-count label (`n_multiplier`)", "",
              "Label = ln(cataloged pieces ≥10 cm / NASA SBM N(≥10 cm)). 0 means the SBM is right; the grey band is "
              "the range the estimator clips predictions to (`ml.contract.BOUNDS`).", "",
              f"![Label histogram]({_label_hist(lab, out)})", "",
              f"![Label by cause]({_label_by_cause(lab, out)})", "",
              f"![Label vs mass]({_label_vs_mass(lab, out)})", ""]
        stats = (lab.assign(cause=lab["explosion_cause"].fillna("—"))
                 .groupby(["event_type", "cause"])["label"]
                 .agg(n="size", median="median", p10=lambda s: s.quantile(0.1), p90=lambda s: s.quantile(0.9))
                 .reset_index())
        L += _table(stats) + [""]
        corr = np.corrcoef(np.log10(lab["total_mass_kg"]), lab["label"])[0, 1] if len(lab) > 2 else float("nan")
        L += [f"Correlation of the label with log10 parent mass: **{corr:+.2f}**. The SBM explosion law "
              "(N = 6·S·Lc^-1.6, S = 1 for every event here) ignores mass, so any trend here is signal the "
              "model can learn.", ""]
        ext = lab.reindex(lab["label"].abs().sort_values(ascending=False).index).head(10)
        ext = ext[["event_id", "year", "event_type", "explosion_cause", "object_class", "total_mass_kg",
                   "n_cataloged", "n_sbm", "label"]]
        L += ["Most extreme labels:", ""] + _table(ext, "{:.1f}") + [""]

    coll = df[df["event_type"] == "collision"]
    if len(coll):
        L += ["## Collisions", "", "EMR = impactor kinetic energy / target mass; ≥ 40 J/g is catastrophic.", ""]
        L += _table(coll[["event_id", "year", "total_mass_kg", "emr_j_per_g", "is_catastrophic", "n_cataloged",
                          "n_sbm", "label"]], "{:.1f}") + [""]

    prov = _optional_csv(config.PROVENANCE_PATH) if data_dir.resolve() == config.DATA.resolve() else None
    if prov is not None:
        L += ["## DISCOS event selection", "",
              f"{len(prov)} DISCOS fragmentation events; {int(prov['included'].sum())} kept as structural breakups.", ""]
        kept = prov.groupby(["discos_event_type", "included"]).size().unstack(fill_value=0)
        kept.columns = ["excluded" if not c else "kept" for c in kept.columns]
        L += _table(kept.sort_values(list(kept.columns)[-1], ascending=False).reset_index()) + [""]
        single = prov[prov["included"] & prov["count_method"].eq("DISCOS cataloguedFragments")
                      & prov["n_debris_satcat_prefix"].notna()]
        if len(single):
            diff = (single["n_cataloged"] - single["n_debris_satcat_prefix"]).abs()
            agree = (diff <= np.maximum(2, 0.1 * single["n_cataloged"])).mean()
            L += [f"Count cross-check: for {len(single)} single-event parents, the DISCOS fragment count agrees with "
                  f"the SATCAT debris count on the same launch designator (within 10% or 2 pieces) for "
                  f"**{agree:.0%}**. Disagreements usually mean another object on the same launch also shed debris.",
                  ""]
            worst = single.assign(diff=diff).sort_values("diff", ascending=False).head(8)
            L += _table(worst[["event_id", "parent_name", "n_cataloged", "n_debris_satcat_prefix"]], "{:.0f}") + [""]
    miss = _undiscovered_debris_launches() if prov is not None else None
    if miss is not None and len(miss):
        L += ["Launches with ≥ 50 cataloged debris pieces but no DISCOS event (likely mission-related objects or "
              "breakups DISCOS files under another launch):", ""] + _table(miss.head(12)) + [""]

    sprov = _optional_csv(config.REPORTS / "spec_provenance.csv") if prov is not None else None
    if sprov is not None:
        L += ["## Spec sources", ""]
        for col in ("mass_source", "bus_family_source", "volume_source"):
            counts = sprov[col].fillna("none").value_counts()
            L.append(f"- `{col}`: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
        g = sprov[sprov["discos_object_class"] == "Payload"]["gunter_match"].fillna("not attempted")
        L.append(f"- Gunter page found for {g.str.match(r'(only|name|manual)').sum()} of {len(g)} payload parents.")
        dropped = sprov[sprov["dropped"].notna()]
        if len(dropped):
            L.append(f"- Dropped parents: {len(dropped)} (" + "; ".join(
                f"{r.norad_id} {r.name}: {r.dropped}" for r in dropped.itertuples()) + ")")
        L.append("")

    L += ["## Assumptions to keep in mind", "",
          "- DISCOS `mass` is used as dry mass and propellant as 0 unless overridden; most breakups happen after "
          "propellant is spent, but propulsion explosions of stages with residual propellant are understated.",
          "- `n_cataloged` counts every piece ever cataloged (decayed included). Pieces that decayed before "
          "they could be cataloged are missing, so short-lived low-altitude clouds read low.",
          "- Events DISCOS classes as anomalous shedding, re-entry break-up, coolant release, or unconfirmed are "
          "excluded (see `pipeline/build_events.py: EVENT_TYPE_MAP`).",
          "- Collision rows need impactor mass and v_rel from `data/manual/collisions.csv`; rows without them are "
          "skipped in training.", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m pipeline.eda", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=config.DATA, help="directory holding the built files")
    p.add_argument("--out", type=Path, default=config.REPORTS, help="where to write eda.md and figures/")
    p.add_argument("--as-of", help="reference date for the catalog-lag cutoff (default: now)")
    a = p.parse_args(argv)
    as_of = parse_epoch(a.as_of) if a.as_of else datetime.now(timezone.utc)
    a.out.mkdir(parents=True, exist_ok=True)
    text = report(a.data, a.out, as_of)
    (a.out / "eda.md").write_text(text, encoding="utf-8")
    print(f"wrote {a.out / 'eda.md'}")


if __name__ == "__main__":
    main()
