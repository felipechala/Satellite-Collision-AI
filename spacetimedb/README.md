# SpacetimeDB module (debris-tracker)

Real-time sync layer: stores precomputed voxel density frames, advances a shared
time-compressed playback clock in a scheduled reducer (10 Hz), and streams every change
to all subscribed clients. Physics stays in Python (`cloud_predictor/`) — this module
never re-runs it.

**Why Rust, not C++:** SpacetimeDB server modules can only be written in Rust or C#.
The gameplan's C++ module snippets (`spacetime init --lang cpp`) describe a toolchain
that does not exist; its own later section uses Rust, which this module follows.

## Build & run

```bash
# one-time setup
# Windows: iwr https://windows.spacetimedb.com -UseBasicParsing | iex
# rustup:  winget install Rustlang.Rustup, then:
rustup target add wasm32-unknown-unknown

spacetime start                                   # local instance on :3000
spacetime publish --project-path spacetimedb debris-tracker
```

## Feed it from the pipeline

```bash
cd cloud_predictor
python publish_breakup.py --event-json examples/event.json --event-id 1 \
    --sat-name "Demo Sat" --alt-km 780 --inc-deg 86.4 --replace \
    --model-dir models/estimator-v1   # optional; omit for uncorrected SBM
```

## Tables (all public)

| table | what clients get |
| --- | --- |
| `breakup_event` | event metadata card (mass, EMR, catastrophic, model version) |
| `cloud_frame` | one row per snapshot: `t_sim_s`, voxel count, peak density |
| `debris_voxel` | voxel centers (ECI km) + density (fragments/km³) per frame |
| `playback` | shared timeline: `frame` is what to render *now*; `speed` is time compression |

## Reducers

Pipeline: `start_breakup`, `publish_frame`, `finalize_breakup`, `delete_breakup`.
Viewers: `set_playback(event_id, playing, speed)`, `seek(event_id, t_sim_s)`.

## Frontend subscription sketch (TypeScript SDK)

```ts
conn.subscriptionBuilder().subscribe([
  "SELECT * FROM breakup_event",
  "SELECT * FROM playback",
  "SELECT * FROM debris_voxel WHERE event_id = 1",
]);
// render voxels where voxel.frame === playback.frame; playback updates ~10 Hz
```

The playback row updating at 10 Hz is the "live" signal; voxel rows are static per
frame, so the initial subscription downloads them once (≤1000 voxels × ~20 frames)
and animation is just switching which frame you draw.
