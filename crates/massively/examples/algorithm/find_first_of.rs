use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::prelude::*;
use massively::{Executor, find_first_of};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let input = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0])?;
    let needles = exec.to_device(&[3.0_f32, 9.0])?;

    let index = find_first_of(
        &exec,
        SoA1(input.slice(..)),
        SoA1(needles.slice(..)),
        common::EqualF32,
    )?;

    assert_eq!(index, Some(2));
    Ok(())
}
