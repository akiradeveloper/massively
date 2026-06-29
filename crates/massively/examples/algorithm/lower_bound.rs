use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::prelude::*;
use massively::{Executor, lower_bound};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let sorted = exec.to_device(&[1.0_f32, 2.0, 2.0, 4.0])?;
    let values = exec.to_device(&[2.0_f32, 3.0])?;

    let indices = lower_bound(
        &exec,
        SoA1(sorted.slice(..)),
        SoA1(values.slice(..)),
        common::LessF32,
    )?;

    assert_eq!(exec.to_host(&indices)?, vec![1, 3]);
    Ok(())
}
