mod common;

use massively::{Executor, Wgpu, adjacent_difference};

fn main() -> common::Result {
    let exec = Executor::<Wgpu>::cpu();
    let values = exec.to_device(&[1.0_f32, 3.0, 6.0, 10.0])?;

    let (output,) = adjacent_difference(&exec, (values.slice(..),), common::SumF32)?;

    println!(
        "adjacent_difference with SumF32: {:?}",
        exec.to_host(&output)?
    );
    Ok(())
}
