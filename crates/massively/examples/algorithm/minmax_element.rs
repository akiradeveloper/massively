use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::prelude::*;
use massively::{Executor, minmax_element};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let values = exec.to_device(&[3.0_f32, 1.0, 2.0])?;

    let indices = minmax_element(&exec, Zip1(values.slice(..)), common::LessF32)?;

    assert_eq!(indices, Some((1, 0)));
    Ok(())
}
