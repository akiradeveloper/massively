mod common;

use massively::{Executor, Wgpu, reduce};

fn main() -> common::Result {
    let exec = Executor::<Wgpu>::cpu();
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0])?;

    let total = reduce(&exec, (values.slice(..),), (0.0,), common::TupleSumF32)?;

    assert_eq!(total, (10.0,));
    Ok(())
}
