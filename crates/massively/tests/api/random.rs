use crate::common::*;

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
