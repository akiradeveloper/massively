use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::prelude::*;
use massively::{Executor, sort, transform};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let values = exec.to_device(&[3.0_f32, 1.0, 2.0])?;

    let sorted = exec.to_device(&[0.0_f32; 3])?;
    sort(
        &exec,
        Zip1(values.slice(..)),
        common::LessF32,
        Zip1(sorted.slice_mut(..)),
    )?;
    let output = exec.to_device(&[0.0_f32; 3])?;
    transform(
        &exec,
        Zip1(sorted.slice(..)),
        common::AddOne,
        Zip1(output.slice_mut(..)),
    )?;

    assert_eq!(exec.to_host(&output)?, vec![2.0, 3.0, 4.0]);
    Ok(())
}
