use crate::common::*;

#[test]
fn copy_if_accepts_heterogeneous_tuple_predicates() {
    let policy = policy();
    let values = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let tags = policy.to_device(&[10_u32, 20, 20, 30]).unwrap();

    let selected = copy_if(zip(&values, &tags), &tags, U32IsTwenty).unwrap();
    let (values, tags) = selected;
    assert_eq!(values.to_vec().unwrap(), vec![2.0, 3.0]);
    assert_eq!(tags.to_vec().unwrap(), vec![20, 20]);
}

#[test]
fn copy_if_accepts_soa12() {
    let policy = policy();
    let a = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let c = policy.to_device(&[100.0_f32, 200.0, 300.0]).unwrap();
    let d = policy.to_device(&[1000_u32, 2000, 3000]).unwrap();
    let e = policy.to_device(&[4.0_f32, 5.0, 6.0]).unwrap();
    let f = policy.to_device(&[40_u32, 50, 60]).unwrap();
    let g = policy.to_device(&[400.0_f32, 500.0, 600.0]).unwrap();
    let h = policy.to_device(&[4000_u32, 5000, 6000]).unwrap();
    let i = policy.to_device(&[7.0_f32, 8.0, 9.0]).unwrap();
    let j = policy.to_device(&[70_u32, 80, 90]).unwrap();
    let k = policy.to_device(&[700.0_f32, 800.0, 900.0]).unwrap();
    let l = policy.to_device(&[100_u32, 200, 300]).unwrap();

    let selected = copy_if(
        zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
        &a,
        F32GreaterThanOne,
    )
    .unwrap();
    let (a, b, _, _, _, _, _, _, _, _, _, l) = selected;
    assert_eq!(a.to_vec().unwrap(), vec![2.0, 3.0]);
    assert_eq!(b.to_vec().unwrap(), vec![20, 30]);
    assert_eq!(l.to_vec().unwrap(), vec![200, 300]);
}
