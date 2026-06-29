use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::prelude::*;
use massively::{Executor, adjacent_find};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let values = exec.to_device(&[1.0_f32, 2.0, 2.0, 3.0])?;

    let index = adjacent_find(&exec, SoA1(values.slice(..)), common::EqualF32)?;

    assert_eq!(index, Some(1));
    Ok(())
}
