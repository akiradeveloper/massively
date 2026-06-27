use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use cubecl::prelude::*;
use massively::slice::transform_slice;
use massively::{Executor, remove_where};

struct OddFirst;

#[cubecl::cube]
impl<R> massively::op::UnaryOp<R, (u32, u32)> for OddFirst
where
    R: Runtime,
{
    type Output = (u32,);

    fn apply(input: (u32, u32)) -> (u32,) {
        (input.0 % 2,)
    }
}

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let left = exec.to_device(&[10_u32, 21, 30, 43])?;
    let right = exec.to_device(&[100_u32, 200, 300, 400])?;

    let stencil = transform_slice(massively::SoA2(left.slice(..), right.slice(..)), OddFirst);
    let (out_left, out_right) = remove_where(
        &exec,
        massively::SoA2(left.slice(..), right.slice(..)),
        stencil,
    )?;

    assert_eq!(exec.to_host(&out_left)?, vec![10, 30]);
    assert_eq!(exec.to_host(&out_right)?, vec![100, 300]);
    Ok(())
}
