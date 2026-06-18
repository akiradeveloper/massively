mod common;

use massively::{CubeWgpu, adjacent_find};

fn main() -> common::Result {
    let policy = CubeWgpu::cpu();
    let values = policy.to_device(&[1.0_f32, 2.0, 2.0, 3.0])?;

    let index = adjacent_find(&values, common::EqualF32)?;

    assert_eq!(index, Some(1));
    Ok(())
}
