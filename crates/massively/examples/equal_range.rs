mod common;

use massively::{CubeWgpu, equal_range};

fn main() -> common::Result {
    let policy = CubeWgpu::cpu();
    let sorted = policy.to_device(&[1.0_f32, 2.0, 2.0, 4.0])?;

    let range = equal_range(&sorted, (2.0,), common::LessF32)?;

    assert_eq!(range, (1, 3));
    Ok(())
}
