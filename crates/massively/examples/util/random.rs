use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, Zip1, transform, util::random};

struct IdentityU32;
struct IdentityF32;

#[cubecl::cube]
impl massively::op::UnaryOp<WgpuRuntime, u32> for IdentityU32 {
    type Output = (u32,);

    fn apply(input: u32) -> (u32,) {
        (input,)
    }
}

#[cubecl::cube]
impl massively::op::UnaryOp<WgpuRuntime, f32> for IdentityF32 {
    type Output = (f32,);

    fn apply(input: f32) -> (f32,) {
        (input,)
    }
}

fn main() -> Result<(), massively::Error> {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);

    let uniform = exec.full(8, 0_u32)?;
    let normal = exec.full(8, 0.0_f32)?;

    transform(
        &exec,
        random::uniform_u32(8, 10, 20, 42)?,
        IdentityU32,
        Zip1(uniform.slice_mut(..)),
    )?;
    transform(
        &exec,
        random::normal_f32(8, 0.0, 1.0, 42),
        IdentityF32,
        Zip1(normal.slice_mut(..)),
    )?;

    let uniform = exec.to_host(&uniform)?;
    let normal = exec.to_host(&normal)?;

    assert_eq!(uniform.len(), 8);
    assert!(uniform.iter().all(|&value| (10..=20).contains(&value)));
    assert_eq!(normal.len(), 8);
    assert!(normal.iter().all(|value| value.is_finite()));

    Ok(())
}
