use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::prelude::*;
use massively::{Executor, equal};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let left = exec.to_device(&[1.0_f32, 2.0, 3.0])?;
    let right = exec.to_device(&[1.0_f32, 2.0, 3.0])?;

    let result = equal(
        &exec,
        Zip1(left.slice(..)),
        Zip1(right.slice(..)),
        common::EqualF32,
    )?;

    assert!(result);
    Ok(())
}
