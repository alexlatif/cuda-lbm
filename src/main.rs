use anyhow::{Context, Result, bail};
use cudarc::{
    driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc,
};
use nvml_wrapper::Nvml;
use serde_json::json;
use std::{
    env, fs,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

const Q: usize = 19;
const ALGORITHMIC_FLOPS_PER_LUP: u64 = 467;
const CX: [i32; Q] = [0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0];
const CY: [i32; Q] = [0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1];
const CZ: [i32; Q] = [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1];

#[derive(Debug)]
struct Args {
    n: usize,
    warmup: usize,
    steps: usize,
    device: usize,
    block: u32,
    omega: f32,
    u0: f32,
    result: PathBuf,
    markers: PathBuf,
    power: PathBuf,
    idle_seconds: u64,
    cpu_parity: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            n: 128,
            warmup: 20,
            steps: 1000,
            device: 0,
            block: 256,
            omega: 1.0,
            u0: 0.03,
            result: "cuda-lbm.json".into(),
            markers: "cuda-lbm.markers.json".into(),
            power: "cuda-lbm.power.csv".into(),
            idle_seconds: 0,
            cpu_parity: false,
        }
    }
}

fn parse_args() -> Result<Args> {
    let mut out = Args::default();
    if let Ok(v) = env::var("LBM_N") {
        out.n = v.parse()?;
    }
    if let Ok(v) = env::var("LBM_WARMUP") {
        out.warmup = v.parse()?;
    }
    if let Ok(v) = env::var("LBM_STEPS") {
        out.steps = v.parse()?;
    }
    if let Ok(v) = env::var("LBM_DEVICE") {
        out.device = v.parse()?;
    }
    if let Ok(v) = env::var("LBM_BLOCK") {
        out.block = v.parse()?;
    }
    if let Ok(v) = env::var("LBM_RESULT") {
        out.result = v.into();
    }
    if let Ok(v) = env::var("LBM_MARKERS") {
        out.markers = v.into();
    }
    if let Ok(v) = env::var("LBM_POWER") {
        out.power = v.into();
    }

    let mut it = env::args().skip(1);
    while let Some(key) = it.next() {
        if key == "--cpu-parity" {
            out.cpu_parity = true;
            continue;
        }
        let value = it
            .next()
            .with_context(|| format!("missing value for {key}"))?;
        match key.as_str() {
            "--n" => out.n = value.parse()?,
            "--warmup" => out.warmup = value.parse()?,
            "--steps" => out.steps = value.parse()?,
            "--device" => out.device = value.parse()?,
            "--block" => out.block = value.parse()?,
            "--omega" => out.omega = value.parse()?,
            "--u0" => out.u0 = value.parse()?,
            "--result" => out.result = value.into(),
            "--markers" => out.markers = value.into(),
            "--power" => out.power = value.into(),
            "--idle-seconds" => out.idle_seconds = value.parse()?,
            _ => bail!("unknown argument: {key}"),
        }
    }

    if out.n == 0 || (!out.n.is_power_of_two()) {
        bail!("--n must be a positive power of two");
    }
    if out.steps == 0 && out.idle_seconds == 0 {
        bail!("--steps must be positive");
    }
    if !matches!(out.block, 128 | 256 | 512) {
        bail!("--block must be one of 128, 256, or 512");
    }
    if out.cpu_parity && out.n > 32 {
        bail!("--cpu-parity is bounded to n <= 32");
    }
    Ok(out)
}

fn weight(q: usize) -> f32 {
    if q == 0 {
        1.0 / 3.0
    } else if q <= 6 {
        1.0 / 18.0
    } else {
        1.0 / 36.0
    }
}

fn initial_state(n: usize, u0: f32) -> Vec<f32> {
    let cells = n * n * n;
    let mut state = vec![0.0f32; Q * cells];
    let two_pi = std::f32::consts::TAU;
    for x in 0..n {
        for y in 0..n {
            for z in 0..n {
                let xf = two_pi * x as f32 / n as f32;
                let yf = two_pi * y as f32 / n as f32;
                let zf = two_pi * z as f32 / n as f32;
                let ux = u0 * xf.sin() * yf.cos() * zf.cos();
                let uy = -u0 * xf.cos() * yf.sin() * zf.cos();
                let uz = 0.0f32;
                let u2 = ux * ux + uy * uy + uz * uz;
                let cell = (x * n + y) * n + z;
                for q in 0..Q {
                    let cu = 3.0 * (CX[q] as f32 * ux + CY[q] as f32 * uy + CZ[q] as f32 * uz);
                    state[q * cells + cell] = weight(q) * (1.0 + cu + 0.5 * cu * cu - 1.5 * u2);
                }
            }
        }
    }
    state
}

fn cpu_step(input: &[f32], n: usize, omega: f32) -> Vec<f32> {
    let cells = n * n * n;
    let mask = n - 1;
    let mut output = vec![0.0f32; input.len()];
    for x in 0..n {
        for y in 0..n {
            for z in 0..n {
                let cell = (x * n + y) * n + z;
                let mut f = [0.0f32; Q];
                let mut rho = 0.0f32;
                let mut mx = 0.0f32;
                let mut my = 0.0f32;
                let mut mz = 0.0f32;
                for q in 0..Q {
                    let sx = x.wrapping_sub_signed(CX[q] as isize) & mask;
                    let sy = y.wrapping_sub_signed(CY[q] as isize) & mask;
                    let sz = z.wrapping_sub_signed(CZ[q] as isize) & mask;
                    let source = (sx * n + sy) * n + sz;
                    let fq = input[q * cells + source];
                    f[q] = fq;
                    rho += fq;
                    mx += CX[q] as f32 * fq;
                    my += CY[q] as f32 * fq;
                    mz += CZ[q] as f32 * fq;
                }
                let inv_rho = 1.0f32 / rho;
                let ux = mx * inv_rho;
                let uy = my * inv_rho;
                let uz = mz * inv_rho;
                let u2 = ux * ux + uy * uy + uz * uz;
                let mut moving_sum = 0.0f32;
                for q in 1..Q {
                    let cu = 3.0f32 * (CX[q] as f32 * ux + CY[q] as f32 * uy + CZ[q] as f32 * uz);
                    let feq = weight(q) * rho * (1.0f32 + cu + 0.5f32 * cu * cu - 1.5f32 * u2);
                    let value = f[q] - omega * (f[q] - feq);
                    output[q * cells + cell] = value;
                    moving_sum += value;
                }
                output[cell] = rho - moving_sum;
            }
        }
    }
    output
}

#[derive(Debug)]
struct Diagnostics {
    mass: f64,
    kinetic: f64,
    rho_min: f32,
    rho_max: f32,
    velocity_l2: f64,
    divergence_rms: f64,
    enstrophy: f64,
}

fn diagnostics(state: &[f32], n: usize) -> Diagnostics {
    let cells = n * n * n;
    let mut mass = 0.0f64;
    let mut kinetic_sum = 0.0f64;
    let mut velocity_sq_sum = 0.0f64;
    let mut rho_min = f32::INFINITY;
    let mut rho_max = f32::NEG_INFINITY;
    let mut ux = vec![0.0f32; cells];
    let mut uy = vec![0.0f32; cells];
    let mut uz = vec![0.0f32; cells];

    for cell in 0..cells {
        let mut rho = 0.0f32;
        let mut mx = 0.0f32;
        let mut my = 0.0f32;
        let mut mz = 0.0f32;
        for q in 0..Q {
            let f = state[q * cells + cell];
            rho += f;
            mx += CX[q] as f32 * f;
            my += CY[q] as f32 * f;
            mz += CZ[q] as f32 * f;
        }
        let vx = mx / rho;
        let vy = my / rho;
        let vz = mz / rho;
        ux[cell] = vx;
        uy[cell] = vy;
        uz[cell] = vz;
        let speed_sq = (vx * vx + vy * vy + vz * vz) as f64;
        mass += rho as f64;
        kinetic_sum += 0.5 * rho as f64 * speed_sq;
        velocity_sq_sum += speed_sq;
        rho_min = rho_min.min(rho);
        rho_max = rho_max.max(rho);
    }

    let mask = n - 1;
    let mut divergence_sq_sum = 0.0f64;
    let mut curl_sq_sum = 0.0f64;
    for x in 0..n {
        let xm = x.wrapping_sub(1) & mask;
        let xp = (x + 1) & mask;
        for y in 0..n {
            let ym = y.wrapping_sub(1) & mask;
            let yp = (y + 1) & mask;
            for z in 0..n {
                let zm = z.wrapping_sub(1) & mask;
                let zp = (z + 1) & mask;
                let at = |xx: usize, yy: usize, zz: usize| (xx * n + yy) * n + zz;
                let dux_dx = 0.5 * (ux[at(xp, y, z)] - ux[at(xm, y, z)]);
                let duy_dy = 0.5 * (uy[at(x, yp, z)] - uy[at(x, ym, z)]);
                let duz_dz = 0.5 * (uz[at(x, y, zp)] - uz[at(x, y, zm)]);
                let divergence = dux_dx + duy_dy + duz_dz;
                divergence_sq_sum += (divergence * divergence) as f64;

                let curl_x = 0.5 * (uz[at(x, yp, z)] - uz[at(x, ym, z)])
                    - 0.5 * (uy[at(x, y, zp)] - uy[at(x, y, zm)]);
                let curl_y = 0.5 * (ux[at(x, y, zp)] - ux[at(x, y, zm)])
                    - 0.5 * (uz[at(xp, y, z)] - uz[at(xm, y, z)]);
                let curl_z = 0.5 * (uy[at(xp, y, z)] - uy[at(xm, y, z)])
                    - 0.5 * (ux[at(x, yp, z)] - ux[at(x, ym, z)]);
                curl_sq_sum += (curl_x * curl_x + curl_y * curl_y + curl_z * curl_z) as f64;
            }
        }
    }

    Diagnostics {
        mass,
        kinetic: kinetic_sum / cells as f64,
        rho_min,
        rho_max,
        velocity_l2: (velocity_sq_sum / (3 * cells) as f64).sqrt(),
        divergence_rms: (divergence_sq_sum / cells as f64).sqrt(),
        enstrophy: 0.5 * curl_sq_sum / cells as f64,
    }
}

fn parity_error(reference: &[f32], observed: &[f32]) -> (f64, f32) {
    let mut diff_sq = 0.0f64;
    let mut ref_sq = 0.0f64;
    let mut max_abs = 0.0f32;
    for (&expected, &actual) in reference.iter().zip(observed) {
        let diff = actual - expected;
        diff_sq += (diff as f64) * (diff as f64);
        ref_sq += (expected as f64) * (expected as f64);
        max_abs = max_abs.max(diff.abs());
    }
    let rel_l2 = if ref_sq > 0.0 {
        (diff_sq / ref_sq).sqrt()
    } else {
        0.0
    };
    (rel_l2, max_abs)
}

fn wall_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_nanos() as u64
}

fn mono_ns() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}

fn specialized_kernel(n: usize, block: u32) -> Result<String> {
    if !n.is_power_of_two() {
        bail!("kernel specialization requires a power-of-two lattice");
    }
    let log2_n = n.trailing_zeros();
    let cells = n.checked_pow(3).context("lattice size overflow")?;
    if cells > u32::MAX as usize {
        bail!("specialized kernel currently requires <= u32::MAX cells");
    }
    Ok(include_str!("../kernels/lbm_step.cu")
        .replace("{{N}}", &n.to_string())
        .replace("{{MASK}}", &(n - 1).to_string())
        .replace("{{LOG2_N}}", &log2_n.to_string())
        .replace("{{LOG2_PLANE}}", &(2 * log2_n).to_string())
        .replace("{{CELLS}}", &cells.to_string())
        .replace("{{BLOCK}}", &block.to_string()))
}

fn launch_step(
    stream: &CudaStream,
    func: &CudaFunction,
    input: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    cells: u64,
    block: u32,
    omega: f32,
) -> Result<()> {
    let grid = ((cells + block as u64 - 1) / block as u64) as u32;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launcher = stream.launch_builder(func);
    launcher.arg(input);
    launcher.arg(output);
    launcher.arg(&omega);
    unsafe { launcher.launch(cfg) }.context("launch cuda_lbm_step")?;
    Ok(())
}

type PowerSample = (u64, f64);

fn start_power_sampler(
    device_index: u32,
) -> Result<(Arc<AtomicBool>, thread::JoinHandle<Vec<PowerSample>>)> {
    let stop = Arc::new(AtomicBool::new(false));
    let thread_stop = Arc::clone(&stop);
    let (ready_tx, ready_rx) = mpsc::sync_channel::<std::result::Result<(), String>>(1);
    let handle = thread::spawn(move || {
        let nvml = match Nvml::init() {
            Ok(v) => v,
            Err(e) => {
                let _ = ready_tx.send(Err(format!("NVML init: {e}")));
                return Vec::new();
            }
        };
        let device = match nvml.device_by_index(device_index) {
            Ok(v) => v,
            Err(e) => {
                let _ = ready_tx.send(Err(format!("NVML device {device_index}: {e}")));
                return Vec::new();
            }
        };
        let sample = || {
            device
                .power_usage()
                .map(|mw| (wall_ns(), mw as f64 / 1000.0))
        };
        let first = match sample() {
            Ok(value) => value,
            Err(e) => {
                let _ = ready_tx.send(Err(format!("NVML power_usage: {e}")));
                return Vec::new();
            }
        };
        let mut samples = vec![first];
        let _ = ready_tx.send(Ok(()));
        while !thread_stop.load(Ordering::Acquire) {
            thread::sleep(Duration::from_millis(50));
            if let Ok(value) = sample() {
                samples.push(value);
            }
        }
        // This final sample brackets the end marker for exact-boundary interpolation.
        if let Ok(value) = sample() {
            samples.push(value);
        }
        samples
    });
    ready_rx
        .recv()
        .context("NVML sampler startup channel")?
        .map_err(anyhow::Error::msg)?;
    Ok((stop, handle))
}

#[derive(Debug)]
struct PowerIntegration {
    energy_j: f64,
    mean_power_w: f64,
    sample_count: usize,
    max_gap_s: f64,
    boundary_start_w: f64,
    boundary_end_w: f64,
}

fn interpolate_power(samples: &[PowerSample], target: u64) -> Result<f64> {
    for &(t, p) in samples {
        if t == target {
            return Ok(p);
        }
    }
    for pair in samples.windows(2) {
        let (t0, p0) = pair[0];
        let (t1, p1) = pair[1];
        if t0 < target && target < t1 {
            let alpha = (target - t0) as f64 / (t1 - t0) as f64;
            return Ok(p0 + alpha * (p1 - p0));
        }
    }
    bail!("power samples do not bracket marker {target}")
}

fn integrate_power(samples: &[PowerSample], start: u64, end: u64) -> Result<PowerIntegration> {
    if start >= end {
        bail!("invalid power integration interval");
    }
    if samples.len() < 2 {
        bail!("need at least two power samples");
    }
    let boundary_start_w = interpolate_power(samples, start)?;
    let boundary_end_w = interpolate_power(samples, end)?;
    let mut selected = Vec::with_capacity(samples.len() + 2);
    selected.push((start, boundary_start_w));
    selected.extend(
        samples
            .iter()
            .copied()
            .filter(|(t, _)| start < *t && *t < end),
    );
    selected.push((end, boundary_end_w));

    let mut energy_j = 0.0f64;
    let mut max_gap_s = 0.0f64;
    for pair in selected.windows(2) {
        let (t0, p0) = pair[0];
        let (t1, p1) = pair[1];
        let dt = (t1 - t0) as f64 / 1e9;
        max_gap_s = max_gap_s.max(dt);
        energy_j += 0.5 * (p0 + p1) * dt;
    }
    let elapsed_s = (end - start) as f64 / 1e9;
    Ok(PowerIntegration {
        energy_j,
        mean_power_w: energy_j / elapsed_s,
        sample_count: selected.len(),
        max_gap_s,
        boundary_start_w,
        boundary_end_w,
    })
}

fn power_csv(samples: &[PowerSample], device_index: usize) -> String {
    let mut out = String::from("wall_ns,backend,device,power_w\n");
    for (t, watts) in samples {
        out.push_str(&format!("{t},nvidia,gpu{device_index},{watts:.6}\n"));
    }
    out
}

fn ensure_parents(args: &Args) -> Result<()> {
    for path in [&args.result, &args.markers, &args.power] {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}

fn run_idle(args: &Args) -> Result<()> {
    ensure_parents(args)?;
    let (power_stop, power_handle) = start_power_sampler(args.device as u32)?;
    let start_wall = wall_ns();
    let start_mono = mono_ns();
    thread::sleep(Duration::from_secs(args.idle_seconds));
    let end_mono = mono_ns();
    let end_wall = wall_ns();
    power_stop.store(true, Ordering::Release);
    let power_samples = power_handle
        .join()
        .map_err(|_| anyhow::anyhow!("power sampler panicked"))?;
    let integrated = integrate_power(&power_samples, start_wall, end_wall)?;
    let result = json!({
        "schema_version": 2,
        "mode": "idle",
        "backend": "cuda",
        "device": args.device,
        "duration_s": (end_mono - start_mono) as f64 / 1e9,
        "power_scope": "nvidia_nvml_board_power",
        "energy_j": integrated.energy_j,
        "mean_power_w": integrated.mean_power_w,
        "power_integration": {
            "method": "trapezoidal-linear-interpolated-exact-marker-boundaries",
            "selected_samples": integrated.sample_count,
            "max_sample_gap_s": integrated.max_gap_s,
            "boundary_start_w": integrated.boundary_start_w,
            "boundary_end_w": integrated.boundary_end_w
        },
        "markers": {
            "start_wall_ns": start_wall,
            "end_wall_ns": end_wall,
            "start_monotonic_ns": start_mono,
            "end_monotonic_ns": end_mono
        }
    });
    let markers = result["markers"].clone();
    fs::write(&args.result, serde_json::to_string_pretty(&result)? + "\n")?;
    fs::write(
        &args.markers,
        serde_json::to_string_pretty(&markers)? + "\n",
    )?;
    fs::write(&args.power, power_csv(&power_samples, args.device))?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

fn main() -> Result<()> {
    let args = parse_args()?;
    if args.idle_seconds > 0 {
        return run_idle(&args);
    }
    ensure_parents(&args)?;

    let cells = args.n.checked_pow(3).context("lattice size overflow")?;
    let host = initial_state(args.n, args.u0);
    let initial = diagnostics(&host, args.n);

    let source = specialized_kernel(args.n, args.block)?;
    let ctx = CudaContext::new(args.device).context("create CUDA context")?;
    let ptx = nvrtc::compile_ptx(source).context("NVRTC compile specialized fused D3Q19")?;
    let module = ctx.load_module(ptx).context("load D3Q19 module")?;
    let func = module
        .load_function("cuda_lbm_step")
        .context("load cuda_lbm_step")?;
    let stream = ctx.default_stream();
    let mut a = stream
        .clone_htod(&host)
        .context("copy initial state to device")?;
    let mut b = stream
        .alloc_zeros::<f32>(host.len())
        .context("allocate output state")?;
    let cells_u64 = cells as u64;

    for _ in 0..args.warmup {
        launch_step(
            &stream, &func, &a, &mut b, cells_u64, args.block, args.omega,
        )?;
        std::mem::swap(&mut a, &mut b);
    }
    stream.synchronize().context("warmup synchronize")?;

    let (power_stop, power_handle) = start_power_sampler(args.device as u32)?;
    let start_wall = wall_ns();
    let start_mono = mono_ns();
    for _ in 0..args.steps {
        launch_step(
            &stream, &func, &a, &mut b, cells_u64, args.block, args.omega,
        )?;
        std::mem::swap(&mut a, &mut b);
    }
    stream.synchronize().context("measurement synchronize")?;
    let end_mono = mono_ns();
    let end_wall = wall_ns();
    power_stop.store(true, Ordering::Release);
    let power_samples = power_handle
        .join()
        .map_err(|_| anyhow::anyhow!("power sampler panicked"))?;
    let integrated = integrate_power(&power_samples, start_wall, end_wall)?;

    let output = stream.clone_dtoh(&a).context("copy result to host")?;
    let final_diag = diagnostics(&output, args.n);
    let elapsed_s = (end_mono - start_mono) as f64 / 1e9;
    let relative_mass_error = (final_diag.mass - initial.mass).abs() / initial.mass.abs();

    let parity = if args.cpu_parity {
        let mut reference_a = host.clone();
        for _ in 0..(args.warmup + args.steps) {
            reference_a = cpu_step(&reference_a, args.n, args.omega);
        }
        let (rel_l2, max_abs) = parity_error(&reference_a, &output);
        Some((rel_l2, max_abs))
    } else {
        None
    };

    let physical_gate = relative_mass_error < 1e-4
        && final_diag.rho_min > 0.0
        && final_diag.rho_max.is_finite()
        && final_diag.velocity_l2.is_finite();
    let parity_gate = parity
        .map(|(rel_l2, max_abs)| rel_l2 < 2e-6 && max_abs < 2e-5)
        .unwrap_or(true);
    let correctness_pass = physical_gate && parity_gate;
    let lattice_updates = cells as u64 * args.steps as u64;

    let result = json!({
        "schema_version": 2,
        "workload": "d3q19-taylor-green",
        "backend": "cuda",
        "implementation": "cudarc-nvrtc-specialized-fused-soa-fp32",
        "execution_mode": "single-device",
        "device": args.device,
        "n": args.n,
        "cells": cells,
        "steps": args.steps,
        "warmup": args.warmup,
        "block_threads": args.block,
        "omega": args.omega,
        "u0": args.u0,
        "elapsed_s": elapsed_s,
        "work_units": lattice_updates,
        "mlups": lattice_updates as f64 / elapsed_s / 1e6,
        "algorithmic_flops_per_lattice_update": ALGORITHMIC_FLOPS_PER_LUP,
        "algorithmic_gflop_s": lattice_updates as f64 * ALGORITHMIC_FLOPS_PER_LUP as f64 / elapsed_s / 1e9,
        "power_scope": "nvidia_nvml_board_power",
        "energy_j": integrated.energy_j,
        "mean_power_w": integrated.mean_power_w,
        "lattice_updates_per_j": lattice_updates as f64 / integrated.energy_j,
        "algorithmic_gflop_per_j": lattice_updates as f64 * ALGORITHMIC_FLOPS_PER_LUP as f64 / integrated.energy_j / 1e9,
        "power_integration": {
            "method": "trapezoidal-linear-interpolated-exact-marker-boundaries",
            "selected_samples": integrated.sample_count,
            "max_sample_gap_s": integrated.max_gap_s,
            "boundary_start_w": integrated.boundary_start_w,
            "boundary_end_w": integrated.boundary_end_w
        },
        "kernel_specialization": {
            "compile_time_n": args.n,
            "compile_time_cells": cells,
            "power_of_two_addressing": true,
            "block_threads": args.block,
            "layout": "soa"
        },
        "initial": {
            "mass": initial.mass,
            "kinetic_energy": initial.kinetic,
            "rho_min": initial.rho_min,
            "rho_max": initial.rho_max,
            "velocity_l2": initial.velocity_l2,
            "divergence_rms": initial.divergence_rms,
            "enstrophy": initial.enstrophy
        },
        "final": {
            "mass": final_diag.mass,
            "kinetic_energy": final_diag.kinetic,
            "rho_min": final_diag.rho_min,
            "rho_max": final_diag.rho_max,
            "velocity_l2": final_diag.velocity_l2,
            "divergence_rms": final_diag.divergence_rms,
            "enstrophy": final_diag.enstrophy
        },
        "relative_mass_error": relative_mass_error,
        "parity": parity.map(|(rel_l2, max_abs)| json!({
            "reference": "independent-host-fp32-pull-bgk",
            "relative_l2": rel_l2,
            "max_abs": max_abs,
            "relative_l2_threshold": 2e-6,
            "max_abs_threshold": 2e-5
        })),
        "correctness_pass": correctness_pass,
        "markers": {
            "start_wall_ns": start_wall,
            "end_wall_ns": end_wall,
            "start_monotonic_ns": start_mono,
            "end_monotonic_ns": end_mono
        }
    });
    let markers = result["markers"].clone();
    fs::write(&args.result, serde_json::to_string_pretty(&result)? + "\n")?;
    fs::write(
        &args.markers,
        serde_json::to_string_pretty(&markers)? + "\n",
    )?;
    fs::write(&args.power, power_csv(&power_samples, args.device))?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    if !correctness_pass {
        bail!("correctness gate failed");
    }
    Ok(())
}
