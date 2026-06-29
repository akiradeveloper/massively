use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::prelude::*;
use massively::{Executor, stable_sort_by_key};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let keys = exec.to_device(&[2_u32, 0, 1])?;
    let values = exec.to_device(&[20.0_f32, 0.0, 10.0])?;

    let (SoA1(keys), SoA1(values)) = stable_sort_by_key(
        &exec,
        SoA1(keys.slice(..)),
        SoA1(values.slice(..)),
        common::LessU32,
    )?;

    assert_eq!(exec.to_host(&keys)?, vec![0, 1, 2]);
    assert_eq!(exec.to_host(&values)?, vec![0.0, 10.0, 20.0]);
    Ok(())
}
