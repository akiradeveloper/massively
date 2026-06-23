mod common;

use massively::{Executor, Wgpu};

fn main() -> common::Result {
    let exec = Executor::<Wgpu>::cpu();
    let values = exec.filled(4, 7_u32)?;

    assert_eq!(exec.to_host(&values)?, vec![7, 7, 7, 7]);
    Ok(())
}
