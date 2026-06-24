use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use cubecl::prelude::*;
use massively::Executor;
use massively::runtime::op::TabulateOp;

struct SquareIndex;

#[cubecl::cube]
impl TabulateOp<WgpuRuntime, u32> for SquareIndex {
    fn apply(index: u32) -> u32 {
        index * index
    }
}

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let values = exec.tabulate(5, SquareIndex)?;

    assert_eq!(exec.to_host(&values)?, vec![0, 1, 4, 9, 16]);
    Ok(())
}
