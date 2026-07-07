use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::op::UnaryOp;
use massively::{Executor, transform};

struct IdentityU32;

#[cubecl::cube]
impl UnaryOp<WgpuRuntime, u32> for IdentityU32 {
    type Output = (u32,);

    fn apply(input: u32) -> (u32,) {
        (input,)
    }
}

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let values = exec.alloc::<(u32,)>(4)?;

    transform(
        &exec,
        massively::lazy::counting(0).take(4),
        IdentityU32,
        values.slice_mut(..),
    )?;

    assert_eq!(exec.to_host(&values.0)?, vec![0, 1, 2, 3]);
    Ok(())
}
