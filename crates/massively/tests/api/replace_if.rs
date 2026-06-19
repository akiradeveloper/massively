use crate::common::*;

#[test]
fn replace_if_accepts_three_tuple_columns() {
    let policy = policy();
    let a = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let b = policy.to_device(&[10_u32, 20, 20, 30]).unwrap();
    let c = policy.to_device(&[1.0_f32, -1.0, 2.0, 3.0]).unwrap();

    let output = replace_if((&a, &b, &c), (99.0_f32, 77_u32, -99.0_f32), &b, U32IsTwenty).unwrap();
    let (a, b, c) = output;

    assert_eq!(a.to_vec().unwrap(), vec![1.0, 99.0, 99.0, 4.0]);
    assert_eq!(b.to_vec().unwrap(), vec![10, 77, 77, 30]);
    assert_eq!(c.to_vec().unwrap(), vec![1.0, -99.0, -99.0, 3.0]);
}

#[cfg(any())]
#[test]
fn replace_if_accepts_heterogeneous_soa12_tail_predicate() {
    let policy = policy();
    let a = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let b = policy.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let c = policy.to_device(&[100.0_f32, 200.0, 300.0, 400.0]).unwrap();
    let d = policy.to_device(&[1000_u32, 2000, 3000, 4000]).unwrap();
    let e = policy.to_device(&[4.0_f32, 5.0, 6.0, 7.0]).unwrap();
    let f = policy.to_device(&[40_u32, 50, 60, 70]).unwrap();
    let g = policy.to_device(&[400.0_f32, 500.0, 600.0, 700.0]).unwrap();
    let h = policy.to_device(&[4000_u32, 5000, 6000, 7000]).unwrap();
    let i = policy.to_device(&[7.0_f32, 8.0, 9.0, 10.0]).unwrap();
    let j = policy.to_device(&[70_u32, 80, 90, 100]).unwrap();
    let k = policy
        .to_device(&[700.0_f32, 800.0, 900.0, 1000.0])
        .unwrap();
    let l = policy.to_device(&[7000_u32, 8000, 9000, 10000]).unwrap();

    let output = replace_if(
        zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
        (
            -1.0_f32, 1_u32, -2.0_f32, 2_u32, -3.0_f32, 3_u32, -4.0_f32, 4_u32, -5.0_f32, 5_u32,
            -6.0_f32, 6_u32,
        ),
        &a,
        F32GreaterThanOne,
    )
    .unwrap();
    let (a, b, c, d, e, f, g, h, i, j, k, l) = output;

    assert_eq!(a.to_vec().unwrap(), vec![1.0, -1.0, -1.0, -1.0]);
    assert_eq!(b.to_vec().unwrap(), vec![10, 1, 1, 1]);
    assert_eq!(c.to_vec().unwrap(), vec![100.0, -2.0, -2.0, -2.0]);
    assert_eq!(d.to_vec().unwrap(), vec![1000, 2, 2, 2]);
    assert_eq!(e.to_vec().unwrap(), vec![4.0, -3.0, -3.0, -3.0]);
    assert_eq!(f.to_vec().unwrap(), vec![40, 3, 3, 3]);
    assert_eq!(g.to_vec().unwrap(), vec![400.0, -4.0, -4.0, -4.0]);
    assert_eq!(h.to_vec().unwrap(), vec![4000, 4, 4, 4]);
    assert_eq!(i.to_vec().unwrap(), vec![7.0, -5.0, -5.0, -5.0]);
    assert_eq!(j.to_vec().unwrap(), vec![70, 5, 5, 5]);
    assert_eq!(k.to_vec().unwrap(), vec![700.0, -6.0, -6.0, -6.0]);
    assert_eq!(l.to_vec().unwrap(), vec![7000, 6, 6, 6]);
}
