use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::prelude::*;
use massively::{Executor, equal_range};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let sorted = exec.to_device(&[1.0_f32, 2.0, 2.0, 4.0])?;

    let range = equal_range(&exec, SoA1(sorted.slice(..)), (2.0,), common::LessF32)?;

    assert_eq!(range, (1, 3));
    Ok(())
}
