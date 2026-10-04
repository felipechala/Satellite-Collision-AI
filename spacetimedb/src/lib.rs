//! SpacetimeDB module for the debris cloud demo: real-time sync + shared playback.
//!
//! The physics (NN-corrected NASA breakup model, J2 + drag propagation, voxelization)
//! runs in Python (cloud_predictor/), which publishes precomputed voxel density frames
//! here via reducers. This module is the real-time layer the gameplan asks SpacetimeDB
//! to be: it stores the frames, advances a shared time-compressed playback clock in a
//! scheduled reducer, and streams every change to all subscribed clients (web globe,
//! other viewers) over WebSockets. It deliberately does NOT re-run physics per tick:
//! stepping voxels linearly inside the DB would be a second, wrong physics engine.
//!
//! Requests: API callers (and the demo page) queue a simulation with request_simulation;
//! the Python worker (cloud_predictor/worker.py) claims it, runs the ensemble, publishes
//! the frames under event_id = request id, and marks it done. The DB is a temporary
//! store: the worker keeps only the most recent simulations.
//!
//! The module is Rust (SpacetimeDB also supports C#, TypeScript and C++ modules).
//! No auth: this is a single-user local demo; anyone connected may call any reducer.

use spacetimedb::{reducer, table, ReducerContext, ScheduleAt, Table, TimeDuration, Timestamp};

const TICK_MS: i64 = 100;
const MAX_VOXELS_PER_FRAME: usize = 2048; // keep WebGL clients in the gameplan's budget
const DEFAULT_SPEED: f32 = 3600.0; // 1 wall second = 1 simulated hour
const MAX_PARAMS_BYTES: usize = 16 * 1024;
const MAX_ERROR_BYTES: usize = 1000;
const MAX_PARTICLES_PER_CALL: usize = 5000;

// ---------- tables (public: all clients can subscribe) ----------

/// One row per breakup event: the metadata the frontend shows on a card.
#[table(accessor = breakup_event, public)]
pub struct BreakupEvent {
    #[primary_key]
    pub event_id: u64,
    pub sat_name: String,
    pub mass_kg: f32,
    pub emr_j_per_g: f32, // 0 for explosions
    pub is_catastrophic: bool,
    pub model_version: String, // which estimator produced the corrections
    pub voxel_km: f32,
}

/// One row per published snapshot of an event's debris cloud.
#[table(accessor = cloud_frame, public)]
pub struct CloudFrame {
    #[primary_key]
    #[auto_inc]
    pub id: u64,
    #[index(btree)]
    pub event_id: u64,
    pub frame: u32,
    pub t_sim_s: f64, // simulated seconds after breakup
    pub n_voxels: u32,
    pub peak_density: f32,       // fragments / km^3 (ensemble mean)
    pub fragments_in_orbit: f64, // weighted real fragments still in orbit (ensemble mean)
}

/// Voxel centers + densities for one frame. Clients subscribe per event and render
/// the frame the playback row points at.
#[table(accessor = debris_voxel, public)]
pub struct DebrisVoxel {
    #[primary_key]
    #[auto_inc]
    pub id: u64,
    #[index(btree)]
    pub event_id: u64,
    pub frame: u32,
    pub x: f32, // ECI km, voxel center
    pub y: f32,
    pub z: f32,
    pub density: f32, // fragments / km^3, ensemble mean (weighted real fragments, not particles)
    pub density_p10: f32, // percentiles across Monte Carlo runs (all equal for a single run)
    pub density_p50: f32,
    pub density_p90: f32,
}

/// One representative fragment's orbit (cloud_predictor engine.propagator.particle_orbits), so a
/// viewer can animate the cloud continuously: at time t after breakup the fragment is at the
/// Kepler position of a = a_km + a_dot t, raan + raan_dot t, argp + argp_dot t,
/// M = mean_anomaly + m_dot t + m_ddot t^2 / 2 (e, inc fixed). Units: km, rad, s.
#[table(accessor = cloud_particle, public)]
pub struct CloudParticle {
    #[primary_key]
    #[auto_inc]
    pub id: u64,
    #[index(btree)]
    pub event_id: u64,
    pub a_km: f32,
    pub e: f32,
    pub inc: f32,
    pub raan: f32,
    pub argp: f32,
    pub mean_anomaly: f32,
    pub raan_dot: f32,
    pub argp_dot: f32,
    pub m_dot: f32,
    pub a_dot: f32,
    pub m_ddot: f32,
    pub t_decay_s: f32, // re-entry time; -1 = still in orbit at the horizon, 0 = never bound
    pub lc_m: f32,      // characteristic length (size) [m]
    pub weight: f32,    // real fragments this particle represents
}

/// One row per simulation asked for through the API. status: queued -> running -> done | failed.
/// Its results are published as breakup_event / cloud_frame / debris_voxel with event_id = id.
#[table(accessor = simulation_request, public)]
pub struct SimulationRequest {
    #[primary_key]
    #[auto_inc]
    pub id: u64,
    #[unique]
    pub request_key: String, // client-chosen UUID, so the caller can find its row
    pub params_json: String, // the /v1/simulations request body; validated by the worker
    pub status: String,
    pub progress: f32, // 0..1
    pub error: String,
    pub created_at: Timestamp,
}

/// Shared playback state: every connected client sees the same timeline position.
#[table(accessor = playback, public)]
pub struct Playback {
    #[primary_key]
    pub event_id: u64,
    pub ready: bool,
    pub playing: bool,
    pub speed: f32, // simulated seconds per wall-clock second (time compression)
    pub t_sim_s: f64,
    pub frame: u32, // what clients should render right now
    pub n_frames: u32,
    pub duration_s: f64,
}

#[table(accessor = tick_timer, scheduled(tick))]
pub struct TickTimer {
    #[primary_key]
    #[auto_inc]
    pub scheduled_id: u64,
    pub scheduled_at: ScheduleAt,
}

// ---------- lifecycle ----------

#[reducer(init)]
pub fn init(ctx: &ReducerContext) {
    ctx.db.tick_timer().insert(TickTimer {
        scheduled_id: 0,
        // A TimeDuration converts into ScheduleAt::Interval: the reducer runs in a loop.
        scheduled_at: TimeDuration::from_micros(TICK_MS * 1000).into(),
    });
}

// ---------- reducers called by the Python pipeline ----------

/// Register a breakup before streaming its frames.
#[reducer]
pub fn start_breakup(
    ctx: &ReducerContext,
    event_id: u64,
    sat_name: String,
    mass_kg: f32,
    emr_j_per_g: f32,
    is_catastrophic: bool,
    model_version: String,
    voxel_km: f32,
) -> Result<(), String> {
    if ctx.db.breakup_event().event_id().find(event_id).is_some() {
        return Err(format!("event {event_id} already exists; delete_breakup it first"));
    }
    ctx.db.breakup_event().insert(BreakupEvent {
        event_id,
        sat_name,
        mass_kg,
        emr_j_per_g,
        is_catastrophic,
        model_version,
        voxel_km,
    });
    ctx.db.playback().insert(Playback {
        event_id,
        ready: false,
        playing: false,
        speed: DEFAULT_SPEED,
        t_sim_s: 0.0,
        frame: 0,
        n_frames: 0,
        duration_s: 0.0,
    });
    Ok(())
}

/// Upload one voxel frame (parallel arrays, one entry per voxel). density is the ensemble
/// mean; density_p10/p50/p90 are percentiles across runs (equal to density for one run).
#[reducer]
pub fn publish_frame(
    ctx: &ReducerContext,
    event_id: u64,
    frame: u32,
    t_sim_s: f64,
    fragments_in_orbit: f64,
    x: Vec<f32>,
    y: Vec<f32>,
    z: Vec<f32>,
    density: Vec<f32>,
    density_p10: Vec<f32>,
    density_p50: Vec<f32>,
    density_p90: Vec<f32>,
) -> Result<(), String> {
    if ctx.db.breakup_event().event_id().find(event_id).is_none() {
        return Err(format!("unknown event {event_id}; call start_breakup first"));
    }
    let n = x.len();
    if [y.len(), z.len(), density.len(), density_p10.len(), density_p50.len(), density_p90.len()]
        .iter()
        .any(|&len| len != n)
    {
        return Err("x, y, z, density and density_p10/p50/p90 must have equal lengths".into());
    }
    if n == 0 || n > MAX_VOXELS_PER_FRAME {
        return Err(format!("frame must hold 1..={MAX_VOXELS_PER_FRAME} voxels, got {n}"));
    }
    let mut peak = 0.0f32;
    for i in 0..n {
        peak = peak.max(density[i]);
        ctx.db.debris_voxel().insert(DebrisVoxel {
            id: 0,
            event_id,
            frame,
            x: x[i],
            y: y[i],
            z: z[i],
            density: density[i],
            density_p10: density_p10[i],
            density_p50: density_p50[i],
            density_p90: density_p90[i],
        });
    }
    ctx.db.cloud_frame().insert(CloudFrame {
        id: 0,
        event_id,
        frame,
        t_sim_s,
        n_voxels: n as u32,
        peak_density: peak,
        fragments_in_orbit,
    });
    Ok(())
}

/// Upload a batch of particle orbits (parallel arrays, one entry per particle; call repeatedly).
#[reducer]
pub fn publish_particles(
    ctx: &ReducerContext,
    event_id: u64,
    a_km: Vec<f32>,
    e: Vec<f32>,
    inc: Vec<f32>,
    raan: Vec<f32>,
    argp: Vec<f32>,
    mean_anomaly: Vec<f32>,
    raan_dot: Vec<f32>,
    argp_dot: Vec<f32>,
    m_dot: Vec<f32>,
    a_dot: Vec<f32>,
    m_ddot: Vec<f32>,
    t_decay_s: Vec<f32>,
    lc_m: Vec<f32>,
    weight: Vec<f32>,
) -> Result<(), String> {
    if ctx.db.breakup_event().event_id().find(event_id).is_none() {
        return Err(format!("unknown event {event_id}; call start_breakup first"));
    }
    let n = a_km.len();
    let lens = [
        e.len(), inc.len(), raan.len(), argp.len(), mean_anomaly.len(), raan_dot.len(), argp_dot.len(),
        m_dot.len(), a_dot.len(), m_ddot.len(), t_decay_s.len(), lc_m.len(), weight.len(),
    ];
    if lens.iter().any(|&len| len != n) {
        return Err("all particle arrays must have equal lengths".into());
    }
    if n > MAX_PARTICLES_PER_CALL {
        return Err(format!("at most {MAX_PARTICLES_PER_CALL} particles per call, got {n}"));
    }
    for i in 0..n {
        ctx.db.cloud_particle().insert(CloudParticle {
            id: 0,
            event_id,
            a_km: a_km[i],
            e: e[i],
            inc: inc[i],
            raan: raan[i],
            argp: argp[i],
            mean_anomaly: mean_anomaly[i],
            raan_dot: raan_dot[i],
            argp_dot: argp_dot[i],
            m_dot: m_dot[i],
            a_dot: a_dot[i],
            m_ddot: m_ddot[i],
            t_decay_s: t_decay_s[i],
            lc_m: lc_m[i],
            weight: weight[i],
        });
    }
    Ok(())
}

/// All frames are in: compute the timeline and (optionally) start playing.
#[reducer]
pub fn finalize_breakup(
    ctx: &ReducerContext,
    event_id: u64,
    speed: f32,
    autoplay: bool,
) -> Result<(), String> {
    let mut pb = ctx
        .db
        .playback()
        .event_id()
        .find(event_id)
        .ok_or(format!("unknown event {event_id}"))?;
    let mut n_frames = 0u32;
    let mut duration = 0.0f64;
    for f in ctx.db.cloud_frame().event_id().filter(event_id) {
        n_frames += 1;
        duration = duration.max(f.t_sim_s);
    }
    if n_frames == 0 {
        return Err(format!("event {event_id} has no frames"));
    }
    pb.ready = true;
    pb.playing = autoplay;
    if speed > 0.0 {
        pb.speed = speed;
    }
    pb.t_sim_s = 0.0;
    pb.frame = frame_at(ctx, event_id, 0.0);
    pb.n_frames = n_frames;
    pb.duration_s = duration;
    ctx.db.playback().event_id().update(pb);
    Ok(())
}

/// Remove an event and everything attached to it.
#[reducer]
pub fn delete_breakup(ctx: &ReducerContext, event_id: u64) -> Result<(), String> {
    if remove_event_data(ctx, event_id) {
        Ok(())
    } else {
        Err(format!("unknown event {event_id}"))
    }
}

/// Delete an event's voxels, frames, playback row and metadata. Returns whether the event existed.
fn remove_event_data(ctx: &ReducerContext, event_id: u64) -> bool {
    ctx.db.debris_voxel().event_id().delete(event_id);
    ctx.db.cloud_frame().event_id().delete(event_id);
    ctx.db.cloud_particle().event_id().delete(event_id);
    if let Some(pb) = ctx.db.playback().event_id().find(event_id) {
        ctx.db.playback().delete(pb);
    }
    match ctx.db.breakup_event().event_id().find(event_id) {
        Some(ev) => {
            ctx.db.breakup_event().delete(ev);
            true
        }
        None => false,
    }
}

// ---------- simulation requests (API -> worker) ----------

/// Queue a simulation. Anyone may call it; the worker validates params_json.
#[reducer]
pub fn request_simulation(ctx: &ReducerContext, request_key: String, params_json: String) -> Result<(), String> {
    if request_key.is_empty() || request_key.len() > 64 {
        return Err("request_key must be 1-64 characters".into());
    }
    if params_json.len() > MAX_PARAMS_BYTES {
        return Err(format!("params_json is over {MAX_PARAMS_BYTES} bytes"));
    }
    if ctx.db.simulation_request().request_key().find(&request_key).is_some() {
        return Err(format!("request_key {request_key} already used"));
    }
    ctx.db.simulation_request().insert(SimulationRequest {
        id: 0,
        request_key,
        params_json,
        status: "queued".into(),
        progress: 0.0,
        error: String::new(),
        created_at: ctx.timestamp,
    });
    Ok(())
}

fn find_request(ctx: &ReducerContext, id: u64) -> Result<SimulationRequest, String> {
    ctx.db
        .simulation_request()
        .id()
        .find(id)
        .ok_or(format!("unknown simulation {id}"))
}

/// Worker: take a queued simulation.
#[reducer]
pub fn claim_simulation(ctx: &ReducerContext, id: u64) -> Result<(), String> {
    let mut req = find_request(ctx, id)?;
    if req.status != "queued" {
        return Err(format!("simulation {id} is {}, not queued", req.status));
    }
    req.status = "running".into();
    ctx.db.simulation_request().id().update(req);
    Ok(())
}

/// Worker: fraction of the Monte Carlo runs finished (0..1).
#[reducer]
pub fn report_progress(ctx: &ReducerContext, id: u64, progress: f32) -> Result<(), String> {
    let mut req = find_request(ctx, id)?;
    req.progress = progress.clamp(0.0, 1.0);
    ctx.db.simulation_request().id().update(req);
    Ok(())
}

/// Worker: all frames are published under event_id = id.
#[reducer]
pub fn complete_simulation(ctx: &ReducerContext, id: u64) -> Result<(), String> {
    let mut req = find_request(ctx, id)?;
    if ctx.db.breakup_event().event_id().find(id).is_none() {
        return Err(format!("simulation {id} has no published event"));
    }
    req.status = "done".into();
    req.progress = 1.0;
    ctx.db.simulation_request().id().update(req);
    Ok(())
}

/// Worker: the run failed; drop any partially published frames.
#[reducer]
pub fn fail_simulation(ctx: &ReducerContext, id: u64, error: String) -> Result<(), String> {
    let mut req = find_request(ctx, id)?;
    remove_event_data(ctx, id);
    req.status = "failed".into();
    req.error = error.chars().take(MAX_ERROR_BYTES).collect();
    ctx.db.simulation_request().id().update(req);
    Ok(())
}

/// Delete a simulation request and its published results.
#[reducer]
pub fn delete_simulation(ctx: &ReducerContext, id: u64) -> Result<(), String> {
    let req = find_request(ctx, id)?;
    remove_event_data(ctx, id);
    ctx.db.simulation_request().delete(req);
    Ok(())
}

// ---------- reducers called by viewers (multiplayer controls) ----------

/// Play/pause and set time compression. speed <= 0 keeps the current speed.
#[reducer]
pub fn set_playback(ctx: &ReducerContext, event_id: u64, playing: bool, speed: f32) -> Result<(), String> {
    let mut pb = ctx
        .db
        .playback()
        .event_id()
        .find(event_id)
        .ok_or(format!("unknown event {event_id}"))?;
    pb.playing = playing && pb.ready;
    if speed > 0.0 {
        pb.speed = speed;
    }
    ctx.db.playback().event_id().update(pb);
    Ok(())
}

/// Jump the shared timeline to a simulated time (the slider).
#[reducer]
pub fn seek(ctx: &ReducerContext, event_id: u64, t_sim_s: f64) -> Result<(), String> {
    let mut pb = ctx
        .db
        .playback()
        .event_id()
        .find(event_id)
        .ok_or(format!("unknown event {event_id}"))?;
    pb.t_sim_s = t_sim_s.clamp(0.0, pb.duration_s);
    pb.frame = frame_at(ctx, event_id, pb.t_sim_s);
    ctx.db.playback().event_id().update(pb);
    Ok(())
}

// ---------- the scheduled playback clock ----------

/// Runs every TICK_MS inside the database: advances playing timelines and loops them.
/// Updating the playback row is what pushes the new frame pointer to every client.
#[reducer]
pub fn tick(ctx: &ReducerContext, _timer: TickTimer) {
    let rows: Vec<Playback> = ctx.db.playback().iter().collect();
    for mut pb in rows {
        if !(pb.ready && pb.playing && pb.duration_s > 0.0) {
            continue;
        }
        pb.t_sim_s += pb.speed as f64 * (TICK_MS as f64 / 1000.0);
        if pb.t_sim_s > pb.duration_s {
            pb.t_sim_s %= pb.duration_s; // loop the demo
        }
        let frame = frame_at(ctx, pb.event_id, pb.t_sim_s);
        if frame != pb.frame || pb.playing {
            pb.frame = frame;
            ctx.db.playback().event_id().update(pb);
        }
    }
}

/// Latest frame whose snapshot time is at or before t (frames per event are few).
fn frame_at(ctx: &ReducerContext, event_id: u64, t: f64) -> u32 {
    let mut best_frame = 0u32;
    let mut best_t = f64::NEG_INFINITY;
    for f in ctx.db.cloud_frame().event_id().filter(event_id) {
        if f.t_sim_s <= t && f.t_sim_s > best_t {
            best_t = f.t_sim_s;
            best_frame = f.frame;
        }
    }
    best_frame
}
