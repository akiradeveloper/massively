use crate::common::*;

#[test]
fn scatter_accepts_heterogeneous_columns() {
    let policy = policy();
    let values = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let ids = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let indices = policy.to_device(&[2_u32, 0, 3]).unwrap();

    let scattered = scatter(zip(&values, &ids), &indices, 4, (0.0_f32, 0_u32)).unwrap();
    let (values, ids) = scattered;

    assert_eq!(values.to_vec().unwrap(), vec![2.0, 0.0, 1.0, 3.0]);
    assert_eq!(ids.to_vec().unwrap(), vec![20, 0, 10, 30]);
}

#[test]
fn scatter_accepts_soa12_values() {
    let policy = policy();
    let indices = policy.to_device(&[2_u32, 1, 0]).unwrap();
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
    let l = policy.to_device(&[7000_u32, 8000, 9000]).unwrap();

    let scattered = scatter(
        zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
        &indices,
        3,
        (
            0.0_f32, 0_u32, 0.0_f32, 0_u32, 0.0_f32, 0_u32, 0.0_f32, 0_u32, 0.0_f32, 0_u32,
            0.0_f32, 0_u32,
        ),
    )
    .unwrap();
    let (a_out, b_out, c_out, d_out, e_out, f_out, g_out, h_out, i_out, j_out, k_out, l_out) =
        scattered;
    assert_eq!(a_out.to_vec().unwrap(), vec![3.0, 2.0, 1.0]);
    assert_eq!(b_out.to_vec().unwrap(), vec![30, 20, 10]);
    assert_eq!(c_out.to_vec().unwrap(), vec![300.0, 200.0, 100.0]);
    assert_eq!(d_out.to_vec().unwrap(), vec![3000, 2000, 1000]);
    assert_eq!(e_out.to_vec().unwrap(), vec![6.0, 5.0, 4.0]);
    assert_eq!(f_out.to_vec().unwrap(), vec![60, 50, 40]);
    assert_eq!(g_out.to_vec().unwrap(), vec![600.0, 500.0, 400.0]);
    assert_eq!(h_out.to_vec().unwrap(), vec![6000, 5000, 4000]);
    assert_eq!(i_out.to_vec().unwrap(), vec![9.0, 8.0, 7.0]);
    assert_eq!(j_out.to_vec().unwrap(), vec![90, 80, 70]);
    assert_eq!(k_out.to_vec().unwrap(), vec![900.0, 800.0, 700.0]);
    assert_eq!(l_out.to_vec().unwrap(), vec![9000, 8000, 7000]);
}
