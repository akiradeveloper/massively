use crate::common::*;

struct IdentityU32;
struct IdentityU64;
struct IdentityF32;
struct IdentityF64;

#[cubecl::cube]
impl UnaryOp<WgpuRuntime, u32> for IdentityU32 {
    type Output = (u32,);

    fn apply(input: u32) -> (u32,) {
        (input,)
    }
}

#[cubecl::cube]
impl UnaryOp<WgpuRuntime, u64> for IdentityU64 {
    type Output = (u64,);

    fn apply(input: u64) -> (u64,) {
        (input,)
    }
}

#[cubecl::cube]
impl UnaryOp<WgpuRuntime, f32> for IdentityF32 {
    type Output = (f32,);

    fn apply(input: f32) -> (f32,) {
        (input,)
    }
}

#[cubecl::cube]
impl UnaryOp<WgpuRuntime, f64> for IdentityF64 {
    type Output = (f64,);

    fn apply(input: f64) -> (f64,) {
        (input,)
    }
}

fn materialize_u32<Input>(exec: &Executor<WgpuRuntime>, input: Input) -> Vec<u32>
where
    Input: MIter<WgpuRuntime, Item = u32>,
{
    let out = exec.full(input.len(), 0_u32).unwrap();
    transform(exec, input, IdentityU32, massively::Zip1(out.slice_mut(..))).unwrap();
    exec.to_host(&out).unwrap()
}

fn materialize_u64<Input>(exec: &Executor<WgpuRuntime>, input: Input) -> Vec<u64>
where
    Input: MIter<WgpuRuntime, Item = u64>,
{
    let out = exec.full(input.len(), 0_u64).unwrap();
    transform(exec, input, IdentityU64, massively::Zip1(out.slice_mut(..))).unwrap();
    exec.to_host(&out).unwrap()
}

fn materialize_f32<Input>(exec: &Executor<WgpuRuntime>, input: Input) -> Vec<f32>
where
    Input: MIter<WgpuRuntime, Item = f32>,
{
    let out = exec.full(input.len(), 0.0_f32).unwrap();
    transform(exec, input, IdentityF32, massively::Zip1(out.slice_mut(..))).unwrap();
    exec.to_host(&out).unwrap()
}

fn materialize_f64<Input>(exec: &Executor<WgpuRuntime>, input: Input) -> Vec<f64>
where
    Input: MIter<WgpuRuntime, Item = f64>,
{
    let out = exec.full(input.len(), 0.0_f64).unwrap();
    transform(exec, input, IdentityF64, massively::Zip1(out.slice_mut(..))).unwrap();
    exec.to_host(&out).unwrap()
}

#[test]
fn uniform_u32_is_deterministic_and_bounded() {
    let exec = exec();

    let first = materialize_u32(
        &exec,
        massively::util::random::uniform_u32(64, 10, 20, 123).unwrap(),
    );
    let second = materialize_u32(
        &exec,
        massively::util::random::uniform_u32(64, 10, 20, 123).unwrap(),
    );
    let different_seed = materialize_u32(
        &exec,
        massively::util::random::uniform_u32(64, 10, 20, 124).unwrap(),
    );

    assert_eq!(first.len(), 64);
    assert!(first.iter().all(|&value| (10..=20).contains(&value)));
    assert_eq!(first, second);
    assert_ne!(first, different_seed);
}

#[test]
fn uniform_u64_is_deterministic_and_bounded() {
    let exec = exec();

    let values = materialize_u64(
        &exec,
        massively::util::random::uniform_u64(32, 1_000, 2_000, 777).unwrap(),
    );
    let again = materialize_u64(
        &exec,
        massively::util::random::uniform_u64(32, 1_000, 2_000, 777).unwrap(),
    );

    assert_eq!(values.len(), 32);
    assert!(values.iter().all(|&value| (1_000..=2_000).contains(&value)));
    assert_eq!(values, again);
}

#[test]
fn uniform_f32_is_deterministic_and_bounded() {
    let exec = exec();

    let values = materialize_f32(
        &exec,
        massively::util::random::uniform_f32(40, -2.5, 7.5, 1234).unwrap(),
    );
    let again = materialize_f32(
        &exec,
        massively::util::random::uniform_f32(40, -2.5, 7.5, 1234).unwrap(),
    );

    assert_eq!(values.len(), 40);
    assert!(values.iter().all(|&value| (-2.5..=7.5).contains(&value)));
    assert_eq!(values, again);
}

#[test]
fn uniform_f64_is_deterministic_and_bounded() {
    let exec = exec();

    let values = materialize_f64(
        &exec,
        massively::util::random::uniform_f64(40, -10.0, -3.0, 4321).unwrap(),
    );
    let again = materialize_f64(
        &exec,
        massively::util::random::uniform_f64(40, -10.0, -3.0, 4321).unwrap(),
    );

    assert_eq!(values.len(), 40);
    assert!(values.iter().all(|&value| (-10.0..=-3.0).contains(&value)));
    assert_eq!(values, again);
}

#[test]
fn normal_f32_is_deterministic_and_finite() {
    let exec = exec();

    let values = materialize_f32(&exec, massively::util::random::normal_f32(48, 2.0, 0.5, 99));
    let again = materialize_f32(&exec, massively::util::random::normal_f32(48, 2.0, 0.5, 99));

    assert_eq!(values.len(), 48);
    assert!(values.iter().all(|value| value.is_finite()));
    assert_eq!(values, again);
}

#[test]
fn normal_f64_is_deterministic_and_finite() {
    let exec = exec();

    let values = materialize_f64(&exec, massively::util::random::normal_f64(48, 2.0, 0.5, 99));
    let again = materialize_f64(&exec, massively::util::random::normal_f64(48, 2.0, 0.5, 99));

    assert_eq!(values.len(), 48);
    assert!(values.iter().all(|value| value.is_finite()));
    assert_eq!(values, again);
}

#[test]
fn random_rejects_invalid_uniform_range() {
    assert!(massively::util::random::uniform_u32::<WgpuRuntime>(8, 5, 4, 1).is_err());
    assert!(massively::util::random::uniform_u64::<WgpuRuntime>(8, 9, 4, 1).is_err());
    assert!(massively::util::random::uniform_f32::<WgpuRuntime>(8, 1.0, 0.0, 1).is_err());
    assert!(massively::util::random::uniform_f64::<WgpuRuntime>(8, 1.0, 0.0, 1).is_err());
}

#[test]
fn uniform_accepts_singleton_range() {
    let exec = exec();

    let u32_values = materialize_u32(
        &exec,
        massively::util::random::uniform_u32(8, 4, 4, 1).unwrap(),
    );
    let u64_values = materialize_u64(
        &exec,
        massively::util::random::uniform_u64(8, 9, 9, 1).unwrap(),
    );
    let f32_values = materialize_f32(
        &exec,
        massively::util::random::uniform_f32(8, 1.5, 1.5, 1).unwrap(),
    );
    let f64_values = materialize_f64(
        &exec,
        massively::util::random::uniform_f64(8, -2.5, -2.5, 1).unwrap(),
    );

    assert_eq!(u32_values, vec![4_u32; 8]);
    assert_eq!(u64_values, vec![9_u64; 8]);
    assert_eq!(f32_values, vec![1.5_f32; 8]);
    assert_eq!(f64_values, vec![-2.5_f64; 8]);
}
