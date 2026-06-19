use crate::common::*;

#[cfg(any())]
#[test]
fn is_partitioned_accepts_borrowed_heterogeneous_soa12() {
    let policy = policy();
    let a = policy.to_device(&[3.0_f32, 4.0, 2.0, 1.0, 0.0]).unwrap();
    let b = policy.to_device(&[30_u32, 40, 20, 10, 0]).unwrap();
    let c = policy
        .to_device(&[300.0_f32, 400.0, 200.0, 100.0, 0.0])
        .unwrap();
    let d = policy.to_device(&[3000_u32, 4000, 2000, 1000, 0]).unwrap();
    let e = policy.to_device(&[3.5_f32, 4.5, 2.5, 1.5, 0.5]).unwrap();
    let f = policy.to_device(&[35_u32, 45, 25, 15, 5]).unwrap();
    let g = policy
        .to_device(&[350.0_f32, 450.0, 250.0, 150.0, 50.0])
        .unwrap();
    let h = policy
        .to_device(&[3500_u32, 4500, 2500, 1500, 500])
        .unwrap();
    let i = policy.to_device(&[6.0_f32, 8.0, 4.0, 2.0, 0.0]).unwrap();
    let j = policy.to_device(&[60_u32, 80, 40, 20, 0]).unwrap();
    let k = policy
        .to_device(&[600.0_f32, 800.0, 400.0, 200.0, 0.0])
        .unwrap();
    let l = policy.to_device(&[6000_u32, 8000, 4000, 2000, 0]).unwrap();

    let input = zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l);
    assert!(is_partitioned(input, Tuple12MixedFirstGreaterThanOne).unwrap());
}
