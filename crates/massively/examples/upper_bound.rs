mod common;

use massively::{CubeWgpu, upper_bound};

fn main() -> common::Result {
    let policy = CubeWgpu::cpu();
    let sorted = policy.to_device(&[1.0_f32, 2.0, 2.0, 4.0])?;

    let index = upper_bound(&sorted, 2.0, common::LessF32)?;

    assert_eq!(index, 3);
    Ok(())
}
