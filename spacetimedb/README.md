# SpacetimeDB module (debris-tracker)

Real-time sync layer: stores precomputed voxel density frames, advances a shared
time-compressed playback clock in a scheduled reducer (10 Hz), and streams every change
to all subscribed clients. Physics stays in Python (`cloud_predictor/`) — this module
never re-runs it. It is also the temporary store behind the API's `/v1/simulations`:
requests are queued here and a Python worker writes the results back.

Written in Rust (SpacetimeDB also supports C#, TypeScript and C++ modules). No auth: this
is a single-user local demo, so any client may call any reducer.

## Build & run

```bash
# one-time setup
# Windows: iwr https://windows.spacetimedb.com -UseBasicParsing | iex
# rustup:  winget install Rustlang.Rustup, then:
rustup target add wasm32-unknown-unknown

spacetime start                                   # local instance on :3000
# from the repo root (the CLI builds ./spacetimedb by default; -p/--module-path to override)
spacetime publish --server local debris-tracker
# after a schema change (e.g. new columns), wipe the local data:
spacetime publish --server local debris-tracker --delete-data=always --yes
```

## Simulation requests (the API path)

```
POST /v1/simulations --> request_simulation(key, params_json)   row: queued
worker.py            --> claim_simulation(id)                   row: running
                         report_progress(id, f)  (per Monte Carlo run)
                         start_breakup / publish_frame / finalize_breakup  (event_id = id)
                         complete_simulation(id)                row: done
                         (any error: fail_simulation(id, msg)   row: failed, frames dropped)
GET /v1/simulations/{id}/frame?t_s=, /frames  <-- SQL reads of cloud_frame + debris_voxel
```

The worker keeps only the 10 newest finished simulations (`delete_simulation`).

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
| `cloud_frame` | one row per snapshot: `t_sim_s`, voxel count, peak density, fragments in orbit |
| `debris_voxel` | voxel centers (ECI km) + density (ensemble mean) and `density_p10/p50/p90` across Monte Carlo runs (fragments/km³). In SQL these columns are named `density_p_10`, `density_p_50`, `density_p_90` (SpacetimeDB adds an underscore before digits). |
| `cloud_particle` | one representative run's fragment orbits (elements + drift rates + re-entry time) so viewers can animate the cloud continuously; see the struct docs in `src/lib.rs` |
| `simulation_request` | API requests: `request_key`, `params_json`, `status` (queued/running/done/failed), `progress`, `error` |
| `playback` | shared timeline: `frame` is what to render *now*; `speed` is time compression |

## Reducers

Pipeline: `start_breakup`, `publish_particles(event_id, a_km[], e[], inc[], raan[], argp[], mean_anomaly[],
raan_dot[], argp_dot[], m_dot[], a_dot[], m_ddot[], t_decay_s[], lc_m[], weight[])` (≤ 5000 per call), `publish_frame(event_id, frame, t_sim_s, fragments_in_orbit,
x[], y[], z[], density[], density_p10[], density_p50[], density_p90[])`, `finalize_breakup`,
`delete_breakup`.
Requests: `request_simulation(request_key, params_json)`; worker: `claim_simulation`,
`report_progress`, `complete_simulation`, `fail_simulation`; cleanup: `delete_simulation`.
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
