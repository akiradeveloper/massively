mod common;

use massively::{CubeWgpu, inner_product};

fn main() -> common::Result {
    let policy = CubeWgpu::cpu();
    let x = policy.to_device(&[1.0_f32, 2.0, 3.0])?;
    let y = policy.to_device(&[10.0_f32, 20.0, 30.0])?;

    let dot = inner_product(
        (x.slice(..),),
        (y.slice(..),),
        common::MulF32,
        (0.0,),
        common::SumF32,
    )?;

    assert_eq!(dot, (140.0,));
    Ok(())
}
