mod common;

use massively::{Executor, Wgpu};

fn main() -> common::Result {
    let exec = Executor::<Wgpu>::cpu();
    let values = exec.to_device(&[1_u32, 2, 3, 4])?;

    assert_eq!(exec.to_host(&values)?, vec![1, 2, 3, 4]);
    Ok(())
}
