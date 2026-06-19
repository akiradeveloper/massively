#[test]
fn public_api_is_available_from_massively() {
    let policy = massively::Policy::<massively::Wgpu>::cpu();
    let input = policy.to_device(&[1_u32, 2, 3]).unwrap();

    assert_eq!(input.to_vec().unwrap(), vec![1, 2, 3]);
}
