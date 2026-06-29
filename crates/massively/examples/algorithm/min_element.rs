use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::prelude::*;
use massively::{Executor, min_element};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let values = exec.to_device(&[3.0_f32, 1.0, 2.0])?;

    let index = min_element(&exec, SoA1(values.slice(..)), common::LessF32)?;

    assert_eq!(index, Some(1));
    Ok(())
}
