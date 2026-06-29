use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::prelude::*;
use massively::{Executor, set_difference};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let left = exec.to_device(&[1.0_f32, 2.0, 4.0])?;
    let right = exec.to_device(&[2.0_f32, 3.0])?;

    let SoA1(output) = set_difference(
        &exec,
        SoA1(left.slice(..)),
        SoA1(right.slice(..)),
        common::LessF32,
    )?;

    println!("set difference: {:?}", exec.to_host(&output)?);
    Ok(())
}
