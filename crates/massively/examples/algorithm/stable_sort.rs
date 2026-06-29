use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::prelude::*;
use massively::{Executor, stable_sort};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let values = exec.to_device(&[3.0_f32, 1.0, 2.0])?;

    let SoA1(output) = stable_sort(&exec, SoA1(values.slice(..)), common::LessF32)?;

    assert_eq!(exec.to_host(&output)?, vec![1.0, 2.0, 3.0]);
    Ok(())
}
