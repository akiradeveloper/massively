use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, util::random};

fn main() -> Result<(), massively::Error> {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);

    let uniform = random::uniform_distribution_u32(&exec, 8, 10, 20, 42)?;
    let normal = random::normal_distribution_f32(&exec, 8, 0.0, 1.0, 42)?;

    let uniform = exec.to_host(&uniform)?;
    let normal = exec.to_host(&normal)?;

    assert_eq!(uniform.len(), 8);
    assert!(uniform.iter().all(|&value| (10..=20).contains(&value)));
    assert_eq!(normal.len(), 8);
    assert!(normal.iter().all(|value| value.is_finite()));

    Ok(())
}
