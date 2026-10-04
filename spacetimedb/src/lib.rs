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
//! Note: SpacetimeDB server modules are Rust or C# only (the gameplan's C++ module
//! snippets describe a toolchain that does not exist; its own later section is Rust).

use spacetimedb::{reducer, table, ReducerContext, ScheduleAt, Table, TimeDuration};

const TICK_MS: i64 = 100;
const MAX_VOXELS_PER_FRAME: usize = 2048; // keep WebGL clients in the gameplan's budget
const DEFAULT_SPEED: f32 = 3600.0; // 1 wall second = 1 simulated hour

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
    pub peak_density: f32, // fragments / km^3
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
    pub density: f32, // fragments / km^3 (weighted real fragments, not particles)
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

/// Upload one voxel frame (parallel arrays, one entry per voxel).
#[reducer]
pub fn publish_frame(
    ctx: &ReducerContext,
    event_id: u64,
    frame: u32,
    t_sim_s: f64,
    x: Vec<f32>,
    y: Vec<f32>,
    z: Vec<f32>,
    density: Vec<f32>,
) -> Result<(), String> {
    if ctx.db.breakup_event().event_id().find(event_id).is_none() {
        return Err(format!("unknown event {event_id}; call start_breakup first"));
    }
    let n = x.len();
    if y.len() != n || z.len() != n || density.len() != n {
        return Err("x, y, z, density must have equal lengths".into());
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
        });
    }
    ctx.db.cloud_frame().insert(CloudFrame {
        id: 0,
        event_id,
        frame,
        t_sim_s,
        n_voxels: n as u32,
        peak_density: peak,
    });
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
    ctx.db.debris_voxel().event_id().delete(event_id);
    ctx.db.cloud_frame().event_id().delete(event_id);
    if let Some(pb) = ctx.db.playback().event_id().find(event_id) {
        ctx.db.playback().delete(pb);
    }
    match ctx.db.breakup_event().event_id().find(event_id) {
        Some(ev) => {
            ctx.db.breakup_event().delete(ev);
            Ok(())
        }
        None => Err(format!("unknown event {event_id}")),
    }
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
