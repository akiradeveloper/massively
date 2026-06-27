use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::{Executor, gather};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let values = exec.to_device(&[10.0_f32, 20.0, 30.0])?;
    let indices = exec.to_device(&[2_u32, 0, 1])?;

    let output = exec.to_device(&[0.0_f32; 3])?;
    gather(
        &exec,
        massively::SoA1(values.slice(..)),
        indices.slice(..),
        massively::SoA1(output.slice_mut(..)),
    )?;

    assert_eq!(exec.to_host(&output)?, vec![30.0, 10.0, 20.0]);
    Ok(())
}
