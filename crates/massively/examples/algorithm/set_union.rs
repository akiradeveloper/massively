use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::prelude::*;
use massively::{Executor, set_union};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let left = exec.to_device(&[1.0_f32, 2.0, 4.0])?;
    let right = exec.to_device(&[2.0_f32, 3.0])?;

    let output = exec.to_device(&[0.0_f32; 5])?;
    let len = set_union(
        &exec,
        Zip1(left.slice(..)),
        Zip1(right.slice(..)),
        common::LessF32,
        Zip1(output.slice_mut(..)),
    )?;

    println!("set union: {:?}", exec.to_host(&output.slice(..len))?);
    Ok(())
}
