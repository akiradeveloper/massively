use crate::common::*;

#[cfg(any())]
#[test]
fn minmax_element_accepts_borrowed_heterogeneous_soa12() {
    let policy = policy();
    let a = policy.to_device(&[9.0_f32, 8.0, 7.0, 6.0]).unwrap();
    let b = policy.to_device(&[90_u32, 80, 70, 60]).unwrap();
    let c = policy.to_device(&[0.0_f32, 0.0, 0.0, 0.0]).unwrap();
    let d = policy.to_device(&[0_u32, 0, 0, 0]).unwrap();
    let e = policy.to_device(&[0.0_f32, 0.0, 0.0, 0.0]).unwrap();
    let f = policy.to_device(&[0_u32, 0, 0, 0]).unwrap();
    let g = policy.to_device(&[0.0_f32, 0.0, 0.0, 0.0]).unwrap();
    let h = policy.to_device(&[0_u32, 0, 0, 0]).unwrap();
    let i = policy.to_device(&[0.0_f32, 0.0, 0.0, 0.0]).unwrap();
    let j = policy.to_device(&[0_u32, 0, 0, 0]).unwrap();
    let k = policy.to_device(&[4.0_f32, 1.0, 9.0, 3.0]).unwrap();
    let l = policy.to_device(&[40_u32, 10, 90, 40]).unwrap();

    assert_eq!(
        min_element(
            zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
            Tuple12MixedTailLess,
        )
        .unwrap(),
        Some(1)
    );
    assert_eq!(
        max_element(
            zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
            Tuple12MixedTailLess,
        )
        .unwrap(),
        Some(2)
    );
    assert_eq!(
        minmax_element(
            zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
            Tuple12MixedTailLess,
        )
        .unwrap(),
        Some((1, 2))
    );
}
