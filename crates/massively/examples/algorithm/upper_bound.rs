use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::prelude::*;
use massively::{Executor, upper_bound};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let sorted = exec.to_device(&[1.0_f32, 2.0, 2.0, 4.0])?;
    let values = exec.to_device(&[2.0_f32, 3.0])?;

    let indices = exec.to_device(&[0_u32; 2])?;
    upper_bound(
        &exec,
        SoA1(sorted.slice(..)),
        SoA1(values.slice(..)),
        common::LessF32,
        indices.slice_mut(..),
    )?;

    assert_eq!(exec.to_host(&indices)?, vec![3, 3]);
    Ok(())
}
