use crate::common::*;

#[test]
fn unique_accepts_borrowed_heterogeneous_soa12() {
    let policy = policy();
    let a = policy.to_device(&[1.0_f32, 1.0, 2.0, 2.0, 3.0]).unwrap();
    let b = policy.to_device(&[10_u32, 10, 20, 20, 30]).unwrap();
    let c = policy
        .to_device(&[100.0_f32, 101.0, 200.0, 201.0, 300.0])
        .unwrap();
    let d = policy
        .to_device(&[1000_u32, 1001, 2000, 2001, 3000])
        .unwrap();
    let e = policy.to_device(&[4.0_f32, 4.5, 5.0, 5.5, 6.0]).unwrap();
    let f = policy.to_device(&[40_u32, 41, 50, 51, 60]).unwrap();
    let g = policy
        .to_device(&[400.0_f32, 401.0, 500.0, 501.0, 600.0])
        .unwrap();
    let h = policy
        .to_device(&[4000_u32, 4001, 5000, 5001, 6000])
        .unwrap();
    let i = policy.to_device(&[7.0_f32, 7.5, 8.0, 8.5, 9.0]).unwrap();
    let j = policy.to_device(&[70_u32, 71, 80, 81, 90]).unwrap();
    let k = policy
        .to_device(&[700.0_f32, 700.0, 800.0, 800.0, 900.0])
        .unwrap();
    let l = policy
        .to_device(&[7000_u32, 7000, 8000, 8000, 9000])
        .unwrap();

    let output = unique(
        zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
        Tuple12MixedEqual,
    )
    .unwrap();
    let (a, b, c, d, e, f, g, h, i, j, k, l) = output;

    assert_eq!(a.to_vec().unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(b.to_vec().unwrap(), vec![10, 20, 30]);
    assert_eq!(c.to_vec().unwrap(), vec![100.0, 200.0, 300.0]);
    assert_eq!(d.to_vec().unwrap(), vec![1000, 2000, 3000]);
    assert_eq!(e.to_vec().unwrap(), vec![4.0, 5.0, 6.0]);
    assert_eq!(f.to_vec().unwrap(), vec![40, 50, 60]);
    assert_eq!(g.to_vec().unwrap(), vec![400.0, 500.0, 600.0]);
    assert_eq!(h.to_vec().unwrap(), vec![4000, 5000, 6000]);
    assert_eq!(i.to_vec().unwrap(), vec![7.0, 8.0, 9.0]);
    assert_eq!(j.to_vec().unwrap(), vec![70, 80, 90]);
    assert_eq!(k.to_vec().unwrap(), vec![700.0, 800.0, 900.0]);
    assert_eq!(l.to_vec().unwrap(), vec![7000, 8000, 9000]);
}
