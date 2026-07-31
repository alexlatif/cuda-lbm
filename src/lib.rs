//! Independently owned handwritten CUDA D3Q19 adapter.
//!
//! This crate is the external comparison oracle for compiler-generated kernels.
//! It owns CUDA source compilation and execution so consumers do not embed or
//! load handwritten CUDA themselves.

use anyhow::{Context, Result, bail};
use cudarc::{
    driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg, sys},
    nvrtc,
};
use std::{path::Path, process::Command, sync::Arc};

pub const Q: usize = 19;

/// Static compiler evidence for the independently owned handwritten kernel.
#[derive(Debug, Clone)]
pub struct HandwrittenAssemblyEvidence {
    cubin: Vec<u8>,
    ptxas_stdout: String,
    ptxas_stderr: String,
    sass: String,
}

impl HandwrittenAssemblyEvidence {
    pub fn cubin(&self) -> &[u8] {
        &self.cubin
    }

    pub fn ptxas_stdout(&self) -> &str {
        &self.ptxas_stdout
    }

    pub fn ptxas_stderr(&self) -> &str {
        &self.ptxas_stderr
    }

    pub fn sass(&self) -> &str {
        &self.sass
    }
}

/// A compile-time-specialized handwritten D3Q19 step running on one CUDA stream.
pub struct HandwrittenD3q19 {
    _context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    function: CudaFunction,
    ptx: String,
    input: CudaSlice<f32>,
    output: CudaSlice<f32>,
    cells: u64,
    block: u32,
    omega: f32,
}

impl HandwrittenD3q19 {
    /// Compile the external CUDA kernel and initialize its device state.
    pub fn new(device: usize, n: usize, block: u32, omega: f32, initial: &[f32]) -> Result<Self> {
        if block == 0 || block > 1024 {
            bail!("CUDA block size must be in 1..=1024");
        }
        let cells = n.checked_pow(3).context("lattice size overflow")?;
        if initial.len() != Q.checked_mul(cells).context("state size overflow")? {
            bail!("initial state length does not match D3Q19 lattice");
        }
        let source = specialized_kernel(n, block)?;
        let context = CudaContext::new(device).context("create CUDA context")?;
        let ptx = nvrtc::compile_ptx(source).context("NVRTC compile handwritten D3Q19")?;
        let ptx_source = ptx.to_src();
        let module = context
            .load_module(ptx)
            .context("load handwritten D3Q19 module")?;
        let function = module
            .load_function("cuda_lbm_step")
            .context("load cuda_lbm_step")?;
        let stream = context.default_stream();
        let input = stream
            .clone_htod(initial)
            .context("copy handwritten initial state")?;
        let output = stream
            .alloc_zeros::<f32>(initial.len())
            .context("allocate handwritten output state")?;
        Ok(Self {
            _context: context,
            stream,
            function,
            ptx: ptx_source,
            input,
            output,
            cells: cells as u64,
            block,
            omega,
        })
    }

    /// Return the exact NVRTC-produced PTX image loaded for this adapter.
    ///
    /// The comparison harness may persist this as diagnostic evidence; JXRS
    /// never receives ownership of or executes this image.
    pub fn ptx(&self) -> &str {
        &self.ptx
    }

    /// Assemble and disassemble the exact PTX as diagnostic evidence.
    ///
    /// The resulting cubin is never loaded by JXRS or used for paired timing.
    pub fn assembly_evidence(
        &self,
        ptxas: impl AsRef<Path>,
        cuobjdump: impl AsRef<Path>,
        architecture: &str,
    ) -> Result<HandwrittenAssemblyEvidence> {
        if architecture.is_empty()
            || !architecture
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        {
            bail!("invalid CUDA architecture label");
        }
        let scratch = tempfile::tempdir().context("create handwritten diagnostic workspace")?;
        let ptx_path = scratch.path().join("handwritten.ptx");
        let cubin_path = scratch.path().join("handwritten.cubin");
        std::fs::write(&ptx_path, &self.ptx).context("write handwritten diagnostic PTX")?;
        let assembly = Command::new(ptxas.as_ref())
            .arg(format!("--gpu-name={architecture}"))
            .arg("--verbose")
            .arg("--output-file")
            .arg(&cubin_path)
            .arg(&ptx_path)
            .output()
            .context("invoke ptxas for handwritten diagnostic evidence")?;
        if !assembly.status.success() {
            bail!(
                "ptxas rejected handwritten PTX: {}",
                String::from_utf8_lossy(&assembly.stderr).trim()
            );
        }
        let disassembly = Command::new(cuobjdump.as_ref())
            .arg("-sass")
            .arg(&cubin_path)
            .output()
            .context("invoke cuobjdump for handwritten diagnostic evidence")?;
        if !disassembly.status.success() {
            bail!(
                "cuobjdump rejected handwritten cubin: {}",
                String::from_utf8_lossy(&disassembly.stderr).trim()
            );
        }
        Ok(HandwrittenAssemblyEvidence {
            cubin: std::fs::read(&cubin_path).context("read handwritten diagnostic cubin")?,
            ptxas_stdout: String::from_utf8_lossy(&assembly.stdout).into_owned(),
            ptxas_stderr: String::from_utf8_lossy(&assembly.stderr).into_owned(),
            sass: String::from_utf8_lossy(&disassembly.stdout).into_owned(),
        })
    }

    /// Replace state with an identical host fixture before a paired trial.
    pub fn reset(&mut self, initial: &[f32]) -> Result<()> {
        if initial.len() != self.input.len() {
            bail!("reset state length mismatch");
        }
        self.input = self
            .stream
            .clone_htod(initial)
            .context("reset handwritten input")?;
        self.output = self
            .stream
            .alloc_zeros::<f32>(initial.len())
            .context("reset handwritten output")?;
        Ok(())
    }

    /// Execute steps without host synchronization.
    pub fn launch_steps(&mut self, steps: usize) -> Result<()> {
        for _ in 0..steps {
            launch_step(
                &self.stream,
                &self.function,
                &self.input,
                &mut self.output,
                self.cells,
                self.block,
                self.omega,
            )?;
            std::mem::swap(&mut self.input, &mut self.output);
        }
        Ok(())
    }

    /// Measure a batch using CUDA events and return nanoseconds per step.
    pub fn measure_steps_ns(&mut self, steps: usize) -> Result<u64> {
        if steps == 0 {
            bail!("measurement requires at least one step");
        }
        let start = self
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .context("record start event")?;
        self.launch_steps(steps)?;
        let end = self
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .context("record end event")?;
        end.synchronize().context("synchronize end event")?;
        self._context
            .bind_to_thread()
            .context("bind handwritten CUDA context for event elapsed time")?;
        let elapsed_ms = start.elapsed_ms(&end).context("elapsed CUDA events")?;
        Ok(((f64::from(elapsed_ms) * 1_000_000.0) / steps as f64).round() as u64)
    }

    /// Synchronize and copy the current state to the host.
    pub fn to_host(&self) -> Result<Vec<f32>> {
        self.stream
            .clone_dtoh(&self.input)
            .context("copy handwritten state to host")
    }
}

fn specialized_kernel(n: usize, block: u32) -> Result<String> {
    if !n.is_power_of_two() {
        bail!("kernel specialization requires a power-of-two lattice");
    }
    let log2_n = n.trailing_zeros();
    let cells = n.checked_pow(3).context("lattice size overflow")?;
    if cells > u32::MAX as usize {
        bail!("specialized kernel requires <= u32::MAX cells");
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
    function: &CudaFunction,
    input: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    cells: u64,
    block: u32,
    omega: f32,
) -> Result<()> {
    let grid = ((cells + u64::from(block) - 1) / u64::from(block)) as u32;
    let config = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launcher = stream.launch_builder(function);
    launcher.arg(input);
    launcher.arg(output);
    launcher.arg(&omega);
    unsafe { launcher.launch(config) }.context("launch cuda_lbm_step")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specialization_is_complete_and_deterministic() {
        let first = specialized_kernel(64, 256).unwrap();
        let second = specialized_kernel(64, 256).unwrap();
        assert_eq!(first, second);
        for token in [
            "{{N}}",
            "{{MASK}}",
            "{{LOG2_N}}",
            "{{LOG2_PLANE}}",
            "{{CELLS}}",
            "{{BLOCK}}",
        ] {
            assert!(!first.contains(token), "unresolved token {token}");
        }
        assert!(first.contains("#define LBM_CELLS 262144"));
    }

    #[test]
    fn specialization_rejects_non_power_of_two() {
        assert!(specialized_kernel(63, 256).is_err());
    }
}
