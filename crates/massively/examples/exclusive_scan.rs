mod common;

use massively::{Executor, Wgpu, exclusive_scan};

fn main() -> common::Result {
    let exec = Executor::<Wgpu>::cpu();
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0])?;

    let (output,) = exclusive_scan(&exec, (values.slice(..),), (0.0,), common::TupleSumF32)?;

    assert_eq!(exec.to_host(&output)?, vec![0.0, 1.0, 3.0, 6.0]);
    Ok(())
}
