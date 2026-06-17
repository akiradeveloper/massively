mod common;
use common::*;

#[test]
fn zip_views_device_vec_as_one_component_soa() {
    let policy = policy();
    let input = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let soa = input;

    let output = soa;
    assert_eq!(output.to_vec().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn device_vec_is_soa1_without_zip() {
    let policy = policy();
    let input = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();

    let output = input;

    assert_eq!(output.to_vec().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn zip_flattens_soa1_columns() {
    let policy = policy();
    let left = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let right = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let indices = policy.to_device(&[0_u32, 1, 2]).unwrap();

    let (left, right) = gather(zip(&left, &right), &indices).unwrap();

    assert_eq!(left.to_vec().unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(right.to_vec().unwrap(), vec![10, 20, 30]);
}

#[test]
fn zip_materializes_heterogeneous_columns() {
    let policy = policy();
    let values = policy.to_device(&[1.5_f32, 2.5, 3.5]).unwrap();
    let ids = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let indices = policy.to_device(&[0_u32, 1, 2]).unwrap();

    let (values, ids) = gather(zip(&values, &ids), &indices).unwrap();

    assert_eq!(values.to_vec().unwrap(), vec![1.5, 2.5, 3.5]);
    assert_eq!(ids.to_vec().unwrap(), vec![10, 20, 30]);
}

#[test]
fn zip_gather_accepts_borrowed_heterogeneous_columns() {
    let policy = policy();
    let values = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let ids = policy.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let indices = policy.to_device(&[3_u32, 1, 0]).unwrap();

    let gathered = gather(zip(&values, &ids), &indices).unwrap();
    let (values, ids) = gathered;

    assert_eq!(values.to_vec().unwrap(), vec![4.0, 2.0, 1.0]);
    assert_eq!(ids.to_vec().unwrap(), vec![40, 20, 10]);
}

#[test]
fn zip_gather_accepts_heterogeneous_columns() {
    let policy = policy();
    let values = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let ids = policy.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let indices = policy.to_device(&[3_u32, 1, 0]).unwrap();

    let gathered = gather(zip(&values, &ids), &indices).unwrap();
    let (values, ids) = gathered;

    assert_eq!(values.to_vec().unwrap(), vec![4.0, 2.0, 1.0]);
    assert_eq!(ids.to_vec().unwrap(), vec![40, 20, 10]);
}

#[test]
fn scatter_accepts_heterogeneous_columns() {
    let policy = policy();
    let values = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let ids = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let indices = policy.to_device(&[2_u32, 0, 3]).unwrap();
    let initial_values = policy.device_filled(4, 0.0_f32).unwrap();
    let initial_ids = policy.device_filled(4, 0_u32).unwrap();

    let scattered = scatter(
        zip(&values, &ids),
        &indices,
        zip(&initial_values, &initial_ids),
    )
    .unwrap();
    let (values, ids) = scattered;

    assert_eq!(values.to_vec().unwrap(), vec![2.0, 0.0, 1.0, 3.0]);
    assert_eq!(ids.to_vec().unwrap(), vec![20, 0, 10, 30]);
}

#[test]
fn gather_if_and_scatter_if_accept_heterogeneous_columns() {
    let policy = policy();
    let values = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let ids = policy.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let indices = policy.to_device(&[3_u32, 1, 0]).unwrap();
    let stencil = policy.to_device(&[1_u32, 0, 1]).unwrap();
    let initial_values = policy.device_filled(3, -1.0_f32).unwrap();
    let initial_ids = policy.device_filled(3, 99_u32).unwrap();

    let gathered = gather_if(
        zip(&values, &ids),
        &indices,
        &stencil,
        zip(&initial_values, &initial_ids),
        NonZero,
    )
    .unwrap();
    let (gathered_values, gathered_ids) = gathered;
    assert_eq!(gathered_values.to_vec().unwrap(), vec![4.0, -1.0, 1.0]);
    assert_eq!(gathered_ids.to_vec().unwrap(), vec![40, 99, 10]);

    let scatter_values = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let scatter_ids = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let scatter_initial_values = policy.device_filled(4, 0.0_f32).unwrap();
    let scatter_initial_ids = policy.device_filled(4, 0_u32).unwrap();
    let scattered = scatter_if(
        zip(&scatter_values, &scatter_ids),
        &indices,
        &stencil,
        zip(&scatter_initial_values, &scatter_initial_ids),
        NonZero,
    )
    .unwrap();
    let (scattered_values, scattered_ids) = scattered;
    assert_eq!(scattered_values.to_vec().unwrap(), vec![3.0, 0.0, 0.0, 1.0]);
    assert_eq!(scattered_ids.to_vec().unwrap(), vec![30, 0, 0, 10]);
}

#[test]
fn zip_concatenates_borrowed_soas() {
    let policy = policy();
    let a = policy.to_device(&[1.0_f32, 2.0]).unwrap();
    let b = policy.to_device(&[10_u32, 20]).unwrap();
    let c = policy.to_device(&[100.0_f32, 200.0]).unwrap();
    let d = policy.to_device(&[1000_u32, 2000]).unwrap();
    let e = policy.to_device(&[10000.0_f32, 20000.0]).unwrap();
    let f = policy.to_device(&[100000_u32, 200000]).unwrap();
    let g = policy.to_device(&[1000000.0_f32, 2000000.0]).unwrap();

    let indices = policy.to_device(&[0_u32, 1]).unwrap();
    let (a, b, c, d, e, f, g) = gather(zip7(&a, &b, &c, &d, &e, &f, &g), &indices).unwrap();

    assert_eq!(a.to_vec().unwrap(), vec![1.0, 2.0]);
    assert_eq!(b.to_vec().unwrap(), vec![10, 20]);
    assert_eq!(c.to_vec().unwrap(), vec![100.0, 200.0]);
    assert_eq!(d.to_vec().unwrap(), vec![1000, 2000]);
    assert_eq!(e.to_vec().unwrap(), vec![10000.0, 20000.0]);
    assert_eq!(f.to_vec().unwrap(), vec![100000, 200000]);
    assert_eq!(g.to_vec().unwrap(), vec![1000000.0, 2000000.0]);
}

#[test]
fn zip_concatenates_column_and_soa() {
    let policy = policy();
    let a = policy.to_device(&[1.0_f32, 2.0]).unwrap();
    let b = policy.to_device(&[10_u32, 20]).unwrap();
    let c = policy.to_device(&[100.0_f32, 200.0]).unwrap();

    let indices = policy.to_device(&[0_u32, 1]).unwrap();
    let (a, b, c) = gather(zip3(&a, &b, &c), &indices).unwrap();

    assert_eq!(a.to_vec().unwrap(), vec![1.0, 2.0]);
    assert_eq!(b.to_vec().unwrap(), vec![10, 20]);
    assert_eq!(c.to_vec().unwrap(), vec![100.0, 200.0]);
}

#[test]
fn gather_and_scatter_accept_wide_borrowed_soas() {
    let policy = policy();
    let indices = policy.to_device(&[2_u32, 1, 0]).unwrap();
    let a = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let c = policy.to_device(&[100.0_f32, 200.0, 300.0]).unwrap();
    let d = policy.to_device(&[1000_u32, 2000, 3000]).unwrap();

    let gathered = gather(zip4(&a, &b, &c, &d), &indices).unwrap();
    let (a_out, b_out, c_out, d_out) = gathered;
    assert_eq!(a_out.to_vec().unwrap(), vec![3.0, 2.0, 1.0]);
    assert_eq!(b_out.to_vec().unwrap(), vec![30, 20, 10]);
    assert_eq!(c_out.to_vec().unwrap(), vec![300.0, 200.0, 100.0]);
    assert_eq!(d_out.to_vec().unwrap(), vec![3000, 2000, 1000]);

    let scattered = scatter(
        zip4(&a, &b, &c, &d),
        &indices,
        zip4(
            &policy.device_filled(3, 0.0_f32).unwrap(),
            &policy.device_filled(3, 0_u32).unwrap(),
            &policy.device_filled(3, 0.0_f32).unwrap(),
            &policy.device_filled(3, 0_u32).unwrap(),
        ),
    )
    .unwrap();
    let (a_out, b_out, c_out, d_out) = scattered;
    assert_eq!(a_out.to_vec().unwrap(), vec![3.0, 2.0, 1.0]);
    assert_eq!(b_out.to_vec().unwrap(), vec![30, 20, 10]);
    assert_eq!(c_out.to_vec().unwrap(), vec![300.0, 200.0, 100.0]);
    assert_eq!(d_out.to_vec().unwrap(), vec![3000, 2000, 1000]);

    let stencil = policy.to_device(&[1_u32, 0, 1]).unwrap();
    let gathered_if = gather_if(
        zip4(&a, &b, &c, &d),
        &indices,
        &stencil,
        zip4(
            &policy.device_filled(3, 0.0_f32).unwrap(),
            &policy.device_filled(3, 0_u32).unwrap(),
            &policy.device_filled(3, 0.0_f32).unwrap(),
            &policy.device_filled(3, 0_u32).unwrap(),
        ),
        NonZero,
    )
    .unwrap();
    let (a_out, b_out, c_out, d_out) = gathered_if;
    assert_eq!(a_out.to_vec().unwrap(), vec![3.0, 0.0, 1.0]);
    assert_eq!(b_out.to_vec().unwrap(), vec![30, 0, 10]);
    assert_eq!(c_out.to_vec().unwrap(), vec![300.0, 0.0, 100.0]);
    assert_eq!(d_out.to_vec().unwrap(), vec![3000, 0, 1000]);

    let scattered_if = scatter_if(
        zip4(&a, &b, &c, &d),
        &indices,
        &stencil,
        zip4(
            &policy.device_filled(3, 0.0_f32).unwrap(),
            &policy.device_filled(3, 0_u32).unwrap(),
            &policy.device_filled(3, 0.0_f32).unwrap(),
            &policy.device_filled(3, 0_u32).unwrap(),
        ),
        NonZero,
    )
    .unwrap();
    let (a_out, b_out, c_out, d_out) = scattered_if;
    assert_eq!(a_out.to_vec().unwrap(), vec![3.0, 0.0, 1.0]);
    assert_eq!(b_out.to_vec().unwrap(), vec![30, 0, 10]);
    assert_eq!(c_out.to_vec().unwrap(), vec![300.0, 0.0, 100.0]);
    assert_eq!(d_out.to_vec().unwrap(), vec![3000, 0, 1000]);
}

#[test]
fn gather_accepts_soa12_values() {
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

    let gathered = gather(
        zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
        &indices,
    )
    .unwrap();
    let (a_out, b_out, c_out, d_out, e_out, f_out, g_out, h_out, i_out, j_out, k_out, l_out) =
        gathered;
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
        zip12(
            &policy.device_filled(3, 0.0_f32).unwrap(),
            &policy.device_filled(3, 0_u32).unwrap(),
            &policy.device_filled(3, 0.0_f32).unwrap(),
            &policy.device_filled(3, 0_u32).unwrap(),
            &policy.device_filled(3, 0.0_f32).unwrap(),
            &policy.device_filled(3, 0_u32).unwrap(),
            &policy.device_filled(3, 0.0_f32).unwrap(),
            &policy.device_filled(3, 0_u32).unwrap(),
            &policy.device_filled(3, 0.0_f32).unwrap(),
            &policy.device_filled(3, 0_u32).unwrap(),
            &policy.device_filled(3, 0.0_f32).unwrap(),
            &policy.device_filled(3, 0_u32).unwrap(),
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

#[test]
fn gather_if_and_scatter_if_accept_soa12_values() {
    let policy = policy();
    let indices = policy.to_device(&[2_u32, 1, 0]).unwrap();
    let stencil = policy.to_device(&[1_u32, 0, 1]).unwrap();
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

    let ia = policy.device_filled(3, -1.0_f32).unwrap();
    let ib = policy.device_filled(3, 99_u32).unwrap();
    let ic = policy.device_filled(3, -2.0_f32).unwrap();
    let id = policy.device_filled(3, 98_u32).unwrap();
    let ie = policy.device_filled(3, -3.0_f32).unwrap();
    let iff = policy.device_filled(3, 97_u32).unwrap();
    let ig = policy.device_filled(3, -4.0_f32).unwrap();
    let ih = policy.device_filled(3, 96_u32).unwrap();
    let ii = policy.device_filled(3, -5.0_f32).unwrap();
    let ij = policy.device_filled(3, 95_u32).unwrap();
    let ik = policy.device_filled(3, -6.0_f32).unwrap();
    let il = policy.device_filled(3, 94_u32).unwrap();

    let gathered = gather_if(
        zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
        &indices,
        &stencil,
        zip12(&ia, &ib, &ic, &id, &ie, &iff, &ig, &ih, &ii, &ij, &ik, &il),
        NonZero,
    )
    .unwrap();
    let (a_out, b_out, c_out, d_out, e_out, f_out, g_out, h_out, i_out, j_out, k_out, l_out) =
        gathered;
    assert_eq!(a_out.to_vec().unwrap(), vec![3.0, -1.0, 1.0]);
    assert_eq!(b_out.to_vec().unwrap(), vec![30, 99, 10]);
    assert_eq!(c_out.to_vec().unwrap(), vec![300.0, -2.0, 100.0]);
    assert_eq!(d_out.to_vec().unwrap(), vec![3000, 98, 1000]);
    assert_eq!(e_out.to_vec().unwrap(), vec![6.0, -3.0, 4.0]);
    assert_eq!(f_out.to_vec().unwrap(), vec![60, 97, 40]);
    assert_eq!(g_out.to_vec().unwrap(), vec![600.0, -4.0, 400.0]);
    assert_eq!(h_out.to_vec().unwrap(), vec![6000, 96, 4000]);
    assert_eq!(i_out.to_vec().unwrap(), vec![9.0, -5.0, 7.0]);
    assert_eq!(j_out.to_vec().unwrap(), vec![90, 95, 70]);
    assert_eq!(k_out.to_vec().unwrap(), vec![900.0, -6.0, 700.0]);
    assert_eq!(l_out.to_vec().unwrap(), vec![9000, 94, 7000]);

    let ia = policy.device_filled(3, -1.0_f32).unwrap();
    let ib = policy.device_filled(3, 99_u32).unwrap();
    let ic = policy.device_filled(3, -2.0_f32).unwrap();
    let id = policy.device_filled(3, 98_u32).unwrap();
    let ie = policy.device_filled(3, -3.0_f32).unwrap();
    let iff = policy.device_filled(3, 97_u32).unwrap();
    let ig = policy.device_filled(3, -4.0_f32).unwrap();
    let ih = policy.device_filled(3, 96_u32).unwrap();
    let ii = policy.device_filled(3, -5.0_f32).unwrap();
    let ij = policy.device_filled(3, 95_u32).unwrap();
    let ik = policy.device_filled(3, -6.0_f32).unwrap();
    let il = policy.device_filled(3, 94_u32).unwrap();
    let scattered = scatter_if(
        zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
        &indices,
        &stencil,
        zip12(&ia, &ib, &ic, &id, &ie, &iff, &ig, &ih, &ii, &ij, &ik, &il),
        NonZero,
    )
    .unwrap();
    let (a_out, b_out, c_out, d_out, e_out, f_out, g_out, h_out, i_out, j_out, k_out, l_out) =
        scattered;
    assert_eq!(a_out.to_vec().unwrap(), vec![3.0, -1.0, 1.0]);
    assert_eq!(b_out.to_vec().unwrap(), vec![30, 99, 10]);
    assert_eq!(c_out.to_vec().unwrap(), vec![300.0, -2.0, 100.0]);
    assert_eq!(d_out.to_vec().unwrap(), vec![3000, 98, 1000]);
    assert_eq!(e_out.to_vec().unwrap(), vec![6.0, -3.0, 4.0]);
    assert_eq!(f_out.to_vec().unwrap(), vec![60, 97, 40]);
    assert_eq!(g_out.to_vec().unwrap(), vec![600.0, -4.0, 400.0]);
    assert_eq!(h_out.to_vec().unwrap(), vec![6000, 96, 4000]);
    assert_eq!(i_out.to_vec().unwrap(), vec![9.0, -5.0, 7.0]);
    assert_eq!(j_out.to_vec().unwrap(), vec![90, 95, 70]);
    assert_eq!(k_out.to_vec().unwrap(), vec![900.0, -6.0, 700.0]);
    assert_eq!(l_out.to_vec().unwrap(), vec![9000, 94, 7000]);
}
