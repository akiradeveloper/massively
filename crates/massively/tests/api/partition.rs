use crate::common::*;

#[cfg(any())]
#[test]
fn partition_accepts_borrowed_heterogeneous_soa12() {
    let policy = policy();
    let a = policy.to_device(&[3.0_f32, 1.0, 4.0, 0.0, 2.0]).unwrap();
    let b = policy.to_device(&[30_u32, 10, 40, 0, 20]).unwrap();
    let c = policy
        .to_device(&[300.0_f32, 100.0, 400.0, 0.0, 200.0])
        .unwrap();
    let d = policy.to_device(&[3000_u32, 1000, 4000, 0, 2000]).unwrap();
    let e = policy.to_device(&[3.5_f32, 1.5, 4.5, 0.5, 2.5]).unwrap();
    let f = policy.to_device(&[35_u32, 15, 45, 5, 25]).unwrap();
    let g = policy
        .to_device(&[350.0_f32, 150.0, 450.0, 50.0, 250.0])
        .unwrap();
    let h = policy
        .to_device(&[3500_u32, 1500, 4500, 500, 2500])
        .unwrap();
    let i = policy.to_device(&[6.0_f32, 2.0, 8.0, 0.0, 4.0]).unwrap();
    let j = policy.to_device(&[60_u32, 20, 80, 0, 40]).unwrap();
    let k = policy
        .to_device(&[600.0_f32, 200.0, 800.0, 0.0, 400.0])
        .unwrap();
    let l = policy.to_device(&[6000_u32, 2000, 8000, 0, 4000]).unwrap();

    let (matching, failing) = partition(
        zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
        Tuple12MixedFirstGreaterThanOne,
    )
    .unwrap();
    let (a, b, c, d, e, f, g, h, i, j, k, l) = matching;
    assert_eq!(a.to_vec().unwrap(), vec![3.0, 4.0, 2.0]);
    assert_eq!(b.to_vec().unwrap(), vec![30, 40, 20]);
    assert_eq!(c.to_vec().unwrap(), vec![300.0, 400.0, 200.0]);
    assert_eq!(d.to_vec().unwrap(), vec![3000, 4000, 2000]);
    assert_eq!(e.to_vec().unwrap(), vec![3.5, 4.5, 2.5]);
    assert_eq!(f.to_vec().unwrap(), vec![35, 45, 25]);
    assert_eq!(g.to_vec().unwrap(), vec![350.0, 450.0, 250.0]);
    assert_eq!(h.to_vec().unwrap(), vec![3500, 4500, 2500]);
    assert_eq!(i.to_vec().unwrap(), vec![6.0, 8.0, 4.0]);
    assert_eq!(j.to_vec().unwrap(), vec![60, 80, 40]);
    assert_eq!(k.to_vec().unwrap(), vec![600.0, 800.0, 400.0]);
    assert_eq!(l.to_vec().unwrap(), vec![6000, 8000, 4000]);

    let (a, b, c, d, e, f, g, h, i, j, k, l) = failing;
    assert_eq!(a.to_vec().unwrap(), vec![1.0, 0.0]);
    assert_eq!(b.to_vec().unwrap(), vec![10, 0]);
    assert_eq!(c.to_vec().unwrap(), vec![100.0, 0.0]);
    assert_eq!(d.to_vec().unwrap(), vec![1000, 0]);
    assert_eq!(e.to_vec().unwrap(), vec![1.5, 0.5]);
    assert_eq!(f.to_vec().unwrap(), vec![15, 5]);
    assert_eq!(g.to_vec().unwrap(), vec![150.0, 50.0]);
    assert_eq!(h.to_vec().unwrap(), vec![1500, 500]);
    assert_eq!(i.to_vec().unwrap(), vec![2.0, 0.0]);
    assert_eq!(j.to_vec().unwrap(), vec![20, 0]);
    assert_eq!(k.to_vec().unwrap(), vec![200.0, 0.0]);
    assert_eq!(l.to_vec().unwrap(), vec![2000, 0]);
}
