use crate::common::*;

#[test]
fn adjacent_find_accepts_borrowed_heterogeneous_soa12() {
    let policy = policy();
    let a = policy.to_device(&[1.0_f32, 2.0, 2.0, 3.0]).unwrap();
    let b = policy.to_device(&[10_u32, 20, 20, 30]).unwrap();
    let c = policy.to_device(&[100.0_f32, 200.0, 201.0, 300.0]).unwrap();
    let d = policy.to_device(&[1000_u32, 2000, 2001, 3000]).unwrap();
    let e = policy.to_device(&[4.0_f32, 5.0, 5.0, 6.0]).unwrap();
    let f = policy.to_device(&[40_u32, 50, 50, 60]).unwrap();
    let g = policy.to_device(&[400.0_f32, 500.0, 500.0, 600.0]).unwrap();
    let h = policy.to_device(&[4000_u32, 5000, 5000, 6000]).unwrap();
    let i = policy.to_device(&[7.0_f32, 8.0, 8.0, 9.0]).unwrap();
    let j = policy.to_device(&[70_u32, 80, 80, 90]).unwrap();
    let k = policy.to_device(&[700.0_f32, 800.0, 800.0, 900.0]).unwrap();
    let l = policy.to_device(&[7000_u32, 8000, 8000, 9000]).unwrap();

    let index = adjacent_find(
        zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
        Tuple12MixedEqual,
    )
    .unwrap();

    assert_eq!(index, Some(1));
}
