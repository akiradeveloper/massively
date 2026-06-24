use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::{Executor, scatter_if};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let values = exec.to_device(&[10.0_f32, -20.0, 30.0])?;
    let indices = exec.to_device(&[2_u32, 0, 1])?;
    let stencil = exec.to_device(&[1_u32, 0, 1])?;

    let (output,) = scatter_if(
        &exec,
        massively::SoA1(values.slice(..)),
        indices.slice(..),
        3,
        (0.0_f32,),
        stencil.slice(..),
    )?;

    assert_eq!(exec.to_host(&output)?, vec![0.0, 30.0, 10.0]);
    Ok(())
}
