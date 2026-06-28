use crate::common::*;

struct IndexedUniformF32;

#[cubecl::cube]
impl UnaryOp<WgpuRuntime, (u32,)> for IndexedUniformF32 {
    type Output = (f32,);

    fn apply(input: (u32,)) -> (f32,) {
        (massively::util::random::uniform_f32(123, input.0),)
    }
}

struct IndexedUniformF64;

#[cubecl::cube]
impl UnaryOp<WgpuRuntime, (u32,)> for IndexedUniformF64 {
    type Output = (f64,);

    fn apply(input: (u32,)) -> (f64,) {
        (massively::util::random::uniform_f64(123, input.0),)
    }
}

struct IndexedUniformU32;

#[cubecl::cube]
impl UnaryOp<WgpuRuntime, (u32,)> for IndexedUniformU32 {
    type Output = (u32,);

    fn apply(input: (u32,)) -> (u32,) {
        (massively::util::random::uniform_u32(10, 20, 123, input.0),)
    }
}

struct IndexedUniformU64;

#[cubecl::cube]
impl UnaryOp<WgpuRuntime, (u32,)> for IndexedUniformU64 {
    type Output = (u64,);

    fn apply(input: (u32,)) -> (u64,) {
        (massively::util::random::uniform_u64(
            1_000, 2_000, 123, input.0,
        ),)
    }
}

struct IndexedUniformU32DifferentSeed;

#[cubecl::cube]
impl UnaryOp<WgpuRuntime, (u32,)> for IndexedUniformU32DifferentSeed {
    type Output = (u32,);

    fn apply(input: (u32,)) -> (u32,) {
        (massively::util::random::uniform_u32(10, 20, 124, input.0),)
    }
}

struct IndexedUniformU32Singleton;

#[cubecl::cube]
impl UnaryOp<WgpuRuntime, (u32,)> for IndexedUniformU32Singleton {
    type Output = (u32,);

    fn apply(input: (u32,)) -> (u32,) {
        (massively::util::random::uniform_u32(7, 7, 123, input.0),)
    }
}

struct IndexedNormalF32;

#[cubecl::cube]
impl UnaryOp<WgpuRuntime, (u32,)> for IndexedNormalF32 {
    type Output = (f32,);

    fn apply(input: (u32,)) -> (f32,) {
        (massively::util::random::normal_f32(2.0, 0.5, 99, input.0),)
    }
}

struct IndexedNormalF64;

#[cubecl::cube]
impl UnaryOp<WgpuRuntime, (u32,)> for IndexedNormalF64 {
    type Output = (f64,);

    fn apply(input: (u32,)) -> (f64,) {
        (massively::util::random::normal_f64(2.0, 0.5, 99, input.0),)
    }
}

#[test]
fn uniform_distribution_u32_is_deterministic_and_bounded() {
    let exec = exec();

    let first = massively::util::random::uniform_distribution_u32(&exec, 64, 10, 20, 123).unwrap();
    let second = massively::util::random::uniform_distribution_u32(&exec, 64, 10, 20, 123).unwrap();
    let different_seed =
        massively::util::random::uniform_distribution_u32(&exec, 64, 10, 20, 124).unwrap();

    let first = exec.to_host(&first).unwrap();
    let second = exec.to_host(&second).unwrap();
    let different_seed = exec.to_host(&different_seed).unwrap();

    assert_eq!(first.len(), 64);
    assert!(first.iter().all(|&value| (10..=20).contains(&value)));
    assert_eq!(first, second);
    assert_ne!(first, different_seed);
}

#[test]
fn uniform_distribution_u64_is_deterministic_and_bounded() {
    let exec = exec();

    let values =
        massively::util::random::uniform_distribution_u64(&exec, 32, 1_000, 2_000, 777).unwrap();
    let again =
        massively::util::random::uniform_distribution_u64(&exec, 32, 1_000, 2_000, 777).unwrap();

    let values = exec.to_host(&values).unwrap();
    let again = exec.to_host(&again).unwrap();

    assert_eq!(values.len(), 32);
    assert!(values.iter().all(|&value| (1_000..=2_000).contains(&value)));
    assert_eq!(values, again);
}

#[test]
fn uniform_distribution_f32_is_deterministic_and_bounded() {
    let exec = exec();

    let values = massively::util::random::uniform_distribution_f32(&exec, 40, 1234).unwrap();
    let again = massively::util::random::uniform_distribution_f32(&exec, 40, 1234).unwrap();

    let values = exec.to_host(&values).unwrap();
    let again = exec.to_host(&again).unwrap();

    assert_eq!(values.len(), 40);
    assert!(values.iter().all(|&value| (0.0..=1.0).contains(&value)));
    assert_eq!(values, again);
}

#[test]
fn uniform_distribution_f64_is_deterministic_and_bounded() {
    let exec = exec();

    let values = massively::util::random::uniform_distribution_f64(&exec, 40, 4321).unwrap();
    let again = massively::util::random::uniform_distribution_f64(&exec, 40, 4321).unwrap();

    let values = exec.to_host(&values).unwrap();
    let again = exec.to_host(&again).unwrap();

    assert_eq!(values.len(), 40);
    assert!(values.iter().all(|&value| (0.0..=1.0).contains(&value)));
    assert_eq!(values, again);
}

#[test]
fn normal_distribution_f32_is_deterministic_and_finite() {
    let exec = exec();

    let values = massively::util::random::normal_distribution_f32(&exec, 48, 2.0, 0.5, 99).unwrap();
    let again = massively::util::random::normal_distribution_f32(&exec, 48, 2.0, 0.5, 99).unwrap();

    let values = exec.to_host(&values).unwrap();
    let again = exec.to_host(&again).unwrap();

    assert_eq!(values.len(), 48);
    assert!(values.iter().all(|value| value.is_finite()));
    assert_eq!(values, again);
}

#[test]
fn normal_distribution_f64_is_deterministic_and_finite() {
    let exec = exec();

    let values = massively::util::random::normal_distribution_f64(&exec, 48, 2.0, 0.5, 99).unwrap();
    let again = massively::util::random::normal_distribution_f64(&exec, 48, 2.0, 0.5, 99).unwrap();

    let values = exec.to_host(&values).unwrap();
    let again = exec.to_host(&again).unwrap();

    assert_eq!(values.len(), 48);
    assert!(values.iter().all(|value| value.is_finite()));
    assert_eq!(values, again);
}

#[test]
fn indexed_uniform_f32_is_available_in_transform_op() {
    let exec = exec();
    let first = exec.to_device(&[0.0_f32; 64]).unwrap();
    let second = exec.to_device(&[0.0_f32; 64]).unwrap();

    transform(
        &exec,
        massively::SoA1(massively::slice::tabulate_slice(64)),
        IndexedUniformF32,
        massively::SoA1(first.slice_mut(..)),
    )
    .unwrap();
    transform(
        &exec,
        massively::SoA1(massively::slice::tabulate_slice(64)),
        IndexedUniformF32,
        massively::SoA1(second.slice_mut(..)),
    )
    .unwrap();

    let first = exec.to_host(&first).unwrap();
    let second = exec.to_host(&second).unwrap();
    assert!(first.iter().all(|&value| (0.0..=1.0).contains(&value)));
    assert_eq!(first, second);
}

#[test]
fn indexed_uniform_f64_is_available_in_transform_op() {
    let exec = exec();
    let output = exec.to_device(&[0.0_f64; 32]).unwrap();

    transform(
        &exec,
        massively::SoA1(massively::slice::tabulate_slice(32)),
        IndexedUniformF64,
        massively::SoA1(output.slice_mut(..)),
    )
    .unwrap();

    let output = exec.to_host(&output).unwrap();
    assert!(output.iter().all(|&value| (0.0..=1.0).contains(&value)));
}

#[test]
fn indexed_integer_uniforms_are_bounded_deterministic_and_seeded() {
    let exec = exec();
    let first = exec.to_device(&[0_u32; 64]).unwrap();
    let second = exec.to_device(&[0_u32; 64]).unwrap();
    let different_seed = exec.to_device(&[0_u32; 64]).unwrap();
    let singleton = exec.to_device(&[0_u32; 8]).unwrap();
    let wide = exec.to_device(&[0_u64; 32]).unwrap();

    transform(
        &exec,
        massively::SoA1(massively::slice::tabulate_slice(64)),
        IndexedUniformU32,
        massively::SoA1(first.slice_mut(..)),
    )
    .unwrap();
    transform(
        &exec,
        massively::SoA1(massively::slice::tabulate_slice(64)),
        IndexedUniformU32,
        massively::SoA1(second.slice_mut(..)),
    )
    .unwrap();
    transform(
        &exec,
        massively::SoA1(massively::slice::tabulate_slice(64)),
        IndexedUniformU32DifferentSeed,
        massively::SoA1(different_seed.slice_mut(..)),
    )
    .unwrap();
    transform(
        &exec,
        massively::SoA1(massively::slice::tabulate_slice(8)),
        IndexedUniformU32Singleton,
        massively::SoA1(singleton.slice_mut(..)),
    )
    .unwrap();
    transform(
        &exec,
        massively::SoA1(massively::slice::tabulate_slice(32)),
        IndexedUniformU64,
        massively::SoA1(wide.slice_mut(..)),
    )
    .unwrap();

    let first = exec.to_host(&first).unwrap();
    let second = exec.to_host(&second).unwrap();
    let different_seed = exec.to_host(&different_seed).unwrap();
    let singleton = exec.to_host(&singleton).unwrap();
    let wide = exec.to_host(&wide).unwrap();

    assert!(first.iter().all(|&value| (10..=20).contains(&value)));
    assert_eq!(first, second);
    assert_ne!(first, different_seed);
    assert_eq!(singleton, vec![7_u32; 8]);
    assert!(wide.iter().all(|&value| (1_000..=2_000).contains(&value)));
}

#[test]
fn indexed_normals_are_available_in_transform_op() {
    let exec = exec();
    let f32_values = exec.to_device(&[0.0_f32; 48]).unwrap();
    let f32_again = exec.to_device(&[0.0_f32; 48]).unwrap();
    let f64_values = exec.to_device(&[0.0_f64; 48]).unwrap();

    transform(
        &exec,
        massively::SoA1(massively::slice::tabulate_slice(48)),
        IndexedNormalF32,
        massively::SoA1(f32_values.slice_mut(..)),
    )
    .unwrap();
    transform(
        &exec,
        massively::SoA1(massively::slice::tabulate_slice(48)),
        IndexedNormalF32,
        massively::SoA1(f32_again.slice_mut(..)),
    )
    .unwrap();
    transform(
        &exec,
        massively::SoA1(massively::slice::tabulate_slice(48)),
        IndexedNormalF64,
        massively::SoA1(f64_values.slice_mut(..)),
    )
    .unwrap();

    let f32_values = exec.to_host(&f32_values).unwrap();
    let f32_again = exec.to_host(&f32_again).unwrap();
    let f64_values = exec.to_host(&f64_values).unwrap();

    assert!(f32_values.iter().all(|value| value.is_finite()));
    assert!(f64_values.iter().all(|value| value.is_finite()));
    assert_eq!(f32_values, f32_again);
    assert!(f32_values.iter().any(|&value| value != 2.0));
    assert!(f64_values.iter().any(|&value| value != 2.0));
}

#[test]
fn random_rejects_invalid_integer_uniform_range() {
    let exec = exec();

    assert!(massively::util::random::uniform_distribution_u32(&exec, 8, 5, 4, 1).is_err());
    assert!(massively::util::random::uniform_distribution_u64(&exec, 8, 9, 4, 1).is_err());
}

#[test]
fn integer_uniform_distribution_accepts_singleton_range() {
    let exec = exec();

    let u32_values = massively::util::random::uniform_distribution_u32(&exec, 8, 4, 4, 1).unwrap();
    let u64_values = massively::util::random::uniform_distribution_u64(&exec, 8, 9, 9, 1).unwrap();

    assert_eq!(exec.to_host(&u32_values).unwrap(), vec![4_u32; 8]);
    assert_eq!(exec.to_host(&u64_values).unwrap(), vec![9_u64; 8]);
}
