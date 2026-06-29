use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::prelude::*;
use massively::{Executor, map, permute};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let values = exec.to_device(&[10.0_f32, 20.0, 30.0])?;
    let indices = exec.to_device(&[2_u32, 0, 1])?;

    let permuted: SoA1<_> = permute(&exec, SoA1(values.slice(..)), indices.slice(..))?;
    let SoA1(output) = map(&exec, permuted.slice(..), common::AddOne)?;

    assert_eq!(exec.to_host(&output)?, vec![31.0, 11.0, 21.0]);
    Ok(())
}
