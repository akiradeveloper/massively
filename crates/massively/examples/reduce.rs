mod common;

use massively::{CubeWgpu, reduce};

fn main() -> common::Result {
    let policy = CubeWgpu::cpu();
    let values = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0])?;

    let total = reduce((values.slice(..),), (0.0,), common::TupleSumF32)?;

    assert_eq!(total, (10.0,));
    Ok(())
}
