use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, MIter, vector::transform};

fn exec() -> Executor<WgpuRuntime> {
    Executor::new(WgpuDevice::DefaultDevice)
}

fn materialize_u32<Input>(exec: &Executor<WgpuRuntime>, input: Input) -> Vec<u32>
where
    Input: MIter<WgpuRuntime, Item = u32>,
{
    let output = exec.to_device(&vec![0_u32; input.len().unwrap() as usize]);
    transform(exec, input, massively::op::Identity, output.slice_mut(..)).unwrap();
    exec.to_host(&output).unwrap()
}

fn materialize_u64<Input>(exec: &Executor<WgpuRuntime>, input: Input) -> Vec<u64>
where
    Input: MIter<WgpuRuntime, Item = u64>,
{
    let output = exec.to_device(&vec![0_u64; input.len().unwrap() as usize]);
    transform(exec, input, massively::op::Identity, output.slice_mut(..)).unwrap();
    exec.to_host(&output).unwrap()
}

fn materialize_f32<Input>(exec: &Executor<WgpuRuntime>, input: Input) -> Vec<f32>
where
    Input: MIter<WgpuRuntime, Item = f32>,
{
    let output = exec.to_device(&vec![0_f32; input.len().unwrap() as usize]);
    transform(exec, input, massively::op::Identity, output.slice_mut(..)).unwrap();
    exec.to_host(&output).unwrap()
}

fn materialize_f64<Input>(exec: &Executor<WgpuRuntime>, input: Input) -> Vec<f64>
where
    Input: MIter<WgpuRuntime, Item = f64>,
{
    let output = exec.to_device(&vec![0_f64; input.len().unwrap() as usize]);
    transform(exec, input, massively::op::Identity, output.slice_mut(..)).unwrap();
    exec.to_host(&output).unwrap()
}

#[test]
fn uniform_u32_is_deterministic_and_bounded() {
    let exec = exec();
    let first = materialize_u32(
        &exec,
        massively::util::random::uniform_u32(10, 20, 123)
            .unwrap()
            .take(64),
    );
    let second = materialize_u32(
        &exec,
        massively::util::random::uniform_u32(10, 20, 123)
            .unwrap()
            .take(64),
    );
    let different_seed = materialize_u32(
        &exec,
        massively::util::random::uniform_u32(10, 20, 124)
            .unwrap()
            .take(64),
    );

    assert!(first.iter().all(|value| (10..=20).contains(value)));
    assert_eq!(first, second);
    assert_ne!(first, different_seed);
}

#[test]
fn random_seed_high_bits_affect_the_stream() {
    let exec = exec();
    let low_seed = materialize_u32(
        &exec,
        massively::util::random::uniform_u32(u32::MIN, u32::MAX, 7)
            .unwrap()
            .take(64),
    );
    let high_seed = materialize_u32(
        &exec,
        massively::util::random::uniform_u32(u32::MIN, u32::MAX, 7 | (1_u64 << 32))
            .unwrap()
            .take(64),
    );

    assert_ne!(low_seed, high_seed);
}

#[test]
fn random_uses_generic_taken_and_tracks_nested_slice_offsets() {
    let exec = exec();
    let full = materialize_u32(
        &exec,
        massively::util::random::uniform_u32(10, 20, 123)
            .unwrap()
            .take(16),
    );
    let taken: massively::lazy::Taken<massively::util::random::UniformU32> =
        massively::util::random::uniform_u32(10, 20, 123)
            .unwrap()
            .take(16);
    let sliced = taken.slice(3..12).slice(2..6);

    assert_eq!(materialize_u32(&exec, sliced), full[5..9]);
}

#[test]
fn uniform_u64_is_deterministic_and_bounded() {
    let exec = exec();
    let first = materialize_u64(
        &exec,
        massively::util::random::uniform_u64(1_000, 2_000, 777)
            .unwrap()
            .take(32),
    );
    let second = materialize_u64(
        &exec,
        massively::util::random::uniform_u64(1_000, 2_000, 777)
            .unwrap()
            .take(32),
    );

    assert!(first.iter().all(|value| (1_000..=2_000).contains(value)));
    assert_eq!(first, second);
}

#[test]
fn uniform_f32_is_deterministic_and_bounded() {
    let exec = exec();
    let first = materialize_f32(
        &exec,
        massively::util::random::uniform_f32(-2.5, 7.5, 1234)
            .unwrap()
            .take(40),
    );
    let second = materialize_f32(
        &exec,
        massively::util::random::uniform_f32(-2.5, 7.5, 1234)
            .unwrap()
            .take(40),
    );

    assert!(first.iter().all(|value| (-2.5..=7.5).contains(value)));
    assert_eq!(first, second);
}

#[test]
fn uniform_f64_is_deterministic_and_bounded() {
    let exec = exec();
    let first = materialize_f64(
        &exec,
        massively::util::random::uniform_f64(-10.0, -3.0, 4321)
            .unwrap()
            .take(40),
    );
    let second = materialize_f64(
        &exec,
        massively::util::random::uniform_f64(-10.0, -3.0, 4321)
            .unwrap()
            .take(40),
    );

    assert!(first.iter().all(|value| (-10.0..=-3.0).contains(value)));
    assert_eq!(first, second);
}

#[test]
fn normal_f32_is_deterministic_and_finite() {
    let exec = exec();
    let first = materialize_f32(
        &exec,
        massively::util::random::normal_f32(2.0, 0.5, 99).take(48),
    );
    let second = materialize_f32(
        &exec,
        massively::util::random::normal_f32(2.0, 0.5, 99).take(48),
    );

    assert!(first.iter().all(|value| value.is_finite()));
    assert_eq!(first, second);
}

#[test]
fn normal_f64_is_deterministic_and_finite() {
    let exec = exec();
    let first = materialize_f64(
        &exec,
        massively::util::random::normal_f64(2.0, 0.5, 99).take(48),
    );
    let second = materialize_f64(
        &exec,
        massively::util::random::normal_f64(2.0, 0.5, 99).take(48),
    );

    assert!(first.iter().all(|value| value.is_finite()));
    assert_eq!(first, second);
}

#[test]
fn uniform_rejects_invalid_ranges() {
    assert!(massively::util::random::uniform_u32(5, 4, 1).is_err());
    assert!(massively::util::random::uniform_u64(9, 4, 1).is_err());
    assert!(massively::util::random::uniform_f32(1.0, 0.0, 1).is_err());
    assert!(massively::util::random::uniform_f64(1.0, 0.0, 1).is_err());
}

#[test]
fn uniform_accepts_singleton_ranges() {
    let exec = exec();

    assert_eq!(
        materialize_u32(
            &exec,
            massively::util::random::uniform_u32(4, 4, 1)
                .unwrap()
                .take(8),
        ),
        vec![4; 8]
    );
    assert_eq!(
        materialize_u64(
            &exec,
            massively::util::random::uniform_u64(9, 9, 1)
                .unwrap()
                .take(8),
        ),
        vec![9; 8]
    );
    assert_eq!(
        materialize_f32(
            &exec,
            massively::util::random::uniform_f32(1.5, 1.5, 1)
                .unwrap()
                .take(8),
        ),
        vec![1.5; 8]
    );
    assert_eq!(
        materialize_f64(
            &exec,
            massively::util::random::uniform_f64(-2.5, -2.5, 1)
                .unwrap()
                .take(8),
        ),
        vec![-2.5; 8]
    );
}
