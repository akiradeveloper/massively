mod common;

use cubecl::prelude::*;
use massively::runtime::op::TabulateOp;
use massively::{Executor, Wgpu};

struct SquareIndex;

#[cubecl::cube]
impl TabulateOp<Wgpu, u32> for SquareIndex {
    fn apply(index: u32) -> u32 {
        index * index
    }
}

fn main() -> common::Result {
    let exec = Executor::<Wgpu>::cpu();
    let values = exec.tabulate(5, SquareIndex)?;

    assert_eq!(exec.to_host(&values)?, vec![0, 1, 4, 9, 16]);
    Ok(())
}
