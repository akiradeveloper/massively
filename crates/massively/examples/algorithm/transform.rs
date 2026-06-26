use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::{Executor, transform};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0])?;

    let mut output = exec.to_device(&[0.0_f32; 3])?;
    transform(
        &exec,
        massively::SoA1(values.slice(..)),
        common::AddOne,
        massively::SoA1(output.slice_mut(..)),
    )?;

    assert_eq!(exec.to_host(&output)?, vec![2.0, 3.0, 4.0]);
    Ok(())
}
