use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::prelude::*;
use massively::{Executor, map, sort};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let values = exec.to_device(&[3.0_f32, 1.0, 2.0])?;

    let sorted = sort(&exec, SoA1(values.slice(..)), common::LessF32)?;
    let SoA1(output) = map(&exec, sorted.slice(..), common::AddOne, ())?;

    assert_eq!(exec.to_host(&output)?, vec![2.0, 3.0, 4.0]);
    Ok(())
}
