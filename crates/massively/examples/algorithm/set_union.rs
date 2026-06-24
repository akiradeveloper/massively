use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::{Executor, set_union};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let left = exec.to_device(&[1.0_f32, 2.0, 4.0])?;
    let right = exec.to_device(&[2.0_f32, 3.0])?;

    let (output,) = set_union(
        &exec,
        massively::SoA1(left.slice(..)),
        massively::SoA1(right.slice(..)),
        common::LessF32,
    )?;

    println!("set union: {:?}", exec.to_host(&output)?);
    Ok(())
}
