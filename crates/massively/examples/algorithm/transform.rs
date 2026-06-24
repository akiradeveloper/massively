use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::{Executor, transform};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0])?;

    let (output,) = transform(&exec, massively::SoA1(values.slice(..)), common::AddOne)?;

    assert_eq!(exec.to_host(&output)?, vec![2.0, 3.0, 4.0]);
    Ok(())
}
