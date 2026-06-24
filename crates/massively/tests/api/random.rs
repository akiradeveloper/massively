use crate::common::*;

#[test]
fn uniform_dist_u32_is_deterministic_and_bounded() {
    let exec = exec();

    let first = massively::random::uniform_dist_u32(&exec, 10, 20, 64, 123).unwrap();
    let second = massively::random::uniform_dist_u32(&exec, 10, 20, 64, 123).unwrap();
    let different_seed = massively::random::uniform_dist_u32(&exec, 10, 20, 64, 124).unwrap();

    let first = exec.to_host(&first).unwrap();
    let second = exec.to_host(&second).unwrap();
    let different_seed = exec.to_host(&different_seed).unwrap();

    assert_eq!(first.len(), 64);
    assert!(first.iter().all(|&value| (10..20).contains(&value)));
    assert_eq!(first, second);
    assert_ne!(first, different_seed);
}

#[test]
fn uniform_dist_u64_is_deterministic_and_bounded() {
    let exec = exec();

    let values = massively::random::uniform_dist_u64(&exec, 1_000, 2_000, 32, 777).unwrap();
    let again = massively::random::uniform_dist_u64(&exec, 1_000, 2_000, 32, 777).unwrap();

    let values = exec.to_host(&values).unwrap();
    let again = exec.to_host(&again).unwrap();

    assert_eq!(values.len(), 32);
    assert!(values.iter().all(|&value| (1_000..2_000).contains(&value)));
    assert_eq!(values, again);
}

#[test]
fn normal_dist_f32_is_deterministic_and_finite() {
    let exec = exec();

    let values = massively::random::normal_dist_f32(&exec, 48, 2.0, 0.5, 99).unwrap();
    let again = massively::random::normal_dist_f32(&exec, 48, 2.0, 0.5, 99).unwrap();

    let values = exec.to_host(&values).unwrap();
    let again = exec.to_host(&again).unwrap();

    assert_eq!(values.len(), 48);
    assert!(values.iter().all(|value| value.is_finite()));
    assert_eq!(values, again);
}

#[test]
fn random_rejects_empty_uniform_range() {
    let exec = exec();

    assert!(massively::random::uniform_dist_u32(&exec, 4, 4, 8, 1).is_err());
    assert!(massively::random::uniform_dist_u64(&exec, 9, 4, 8, 1).is_err());
}
