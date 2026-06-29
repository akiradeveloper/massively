use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::prelude::*;
use massively::{Executor, mismatch};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let left = exec.to_device(&[1.0_f32, 2.0, 3.0])?;
    let right = exec.to_device(&[1.0_f32, 9.0, 3.0])?;

    let index = mismatch(
        &exec,
        SoA1(left.slice(..)),
        SoA1(right.slice(..)),
        common::EqualF32,
    )?;

    assert_eq!(index, Some(1));
    Ok(())
}
