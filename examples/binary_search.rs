mod common;

use massively::{CubeWgpu, binary_search};

fn main() -> common::Result {
    let policy = CubeWgpu::cpu();
    let sorted = policy.to_device(&[1.0_f32, 2.0, 4.0, 8.0])?;

    let found = binary_search(&sorted, 4.0, common::LessF32)?;

    assert!(found);
    Ok(())
}
