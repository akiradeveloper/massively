mod common;
use common::*;

#[test]
fn zip_views_device_vec_as_one_component_device_soa() {
    let policy = policy();
    let input = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let soa = input;

    let output = unzip(soa).unwrap();
    assert_eq!(output.to_vec().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn device_vec_is_soa1_without_zip() {
    let policy = policy();
    let input = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();

    let output = unzip(input).unwrap();

    assert_eq!(output.to_vec().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn zip_flattens_soa1_columns() {
    let policy = policy();
    let left = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let right = policy.to_device(&[10_u32, 20, 30]).unwrap();

    let (left, right) = unzip(zip(left, right)).unwrap();

    assert_eq!(left.to_vec().unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(right.to_vec().unwrap(), vec![10, 20, 30]);
}

#[test]
fn zip_unzip_accepts_heterogeneous_columns() {
    let policy = policy();
    let values = policy.to_device(&[1.5_f32, 2.5, 3.5]).unwrap();
    let ids = policy.to_device(&[10_u32, 20, 30]).unwrap();

    let (values, ids) = unzip(zip(values, ids)).unwrap();

    assert_eq!(values.to_vec().unwrap(), vec![1.5, 2.5, 3.5]);
    assert_eq!(ids.to_vec().unwrap(), vec![10, 20, 30]);
}

#[test]
fn vzip_gather_accepts_heterogeneous_columns() {
    let policy = policy();
    let values = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let ids = policy.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let indices = policy.to_device(&[3_u32, 1, 0]).unwrap();

    let gathered = gather(vzip(&values, &ids), &indices).unwrap();
    let (values, ids) = unzip(gathered).unwrap();

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
        vzip(&values, &ids),
        &indices,
        zip(initial_values, initial_ids),
    )
    .unwrap();
    let (values, ids) = unzip(scattered).unwrap();

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
        vzip(&values, &ids),
        &indices,
        &stencil,
        zip(initial_values, initial_ids),
        NonZero,
    )
    .unwrap();
    let (gathered_values, gathered_ids) = unzip(gathered).unwrap();
    assert_eq!(gathered_values.to_vec().unwrap(), vec![4.0, -1.0, 1.0]);
    assert_eq!(gathered_ids.to_vec().unwrap(), vec![40, 99, 10]);

    let scatter_values = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let scatter_ids = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let scatter_initial_values = policy.device_filled(4, 0.0_f32).unwrap();
    let scatter_initial_ids = policy.device_filled(4, 0_u32).unwrap();
    let scattered = scatter_if(
        vzip(&scatter_values, &scatter_ids),
        &indices,
        &stencil,
        zip(scatter_initial_values, scatter_initial_ids),
        NonZero,
    )
    .unwrap();
    let (scattered_values, scattered_ids) = unzip(scattered).unwrap();
    assert_eq!(scattered_values.to_vec().unwrap(), vec![3.0, 0.0, 0.0, 1.0]);
    assert_eq!(scattered_ids.to_vec().unwrap(), vec![30, 0, 0, 10]);
}

#[test]
fn zip_concatenates_device_soas() {
    let policy = policy();
    let a = policy.to_device(&[1.0_f32, 2.0]).unwrap();
    let b = policy.to_device(&[10_u32, 20]).unwrap();
    let c = policy.to_device(&[100.0_f32, 200.0]).unwrap();
    let d = policy.to_device(&[1000_u32, 2000]).unwrap();
    let e = policy.to_device(&[10000.0_f32, 20000.0]).unwrap();
    let f = policy.to_device(&[100000_u32, 200000]).unwrap();
    let g = policy.to_device(&[1000000.0_f32, 2000000.0]).unwrap();

    let left = zip3(a, b, c);
    let right = zip4(d, e, f, g);
    let all = zip(left, right);
    let (a, b, c, d, e, f, g) = unzip(all).unwrap();

    assert_eq!(a.to_vec().unwrap(), vec![1.0, 2.0]);
    assert_eq!(b.to_vec().unwrap(), vec![10, 20]);
    assert_eq!(c.to_vec().unwrap(), vec![100.0, 200.0]);
    assert_eq!(d.to_vec().unwrap(), vec![1000, 2000]);
    assert_eq!(e.to_vec().unwrap(), vec![10000.0, 20000.0]);
    assert_eq!(f.to_vec().unwrap(), vec![100000, 200000]);
    assert_eq!(g.to_vec().unwrap(), vec![1000000.0, 2000000.0]);
}

#[test]
fn zip_concatenates_column_and_device_soa() {
    let policy = policy();
    let a = policy.to_device(&[1.0_f32, 2.0]).unwrap();
    let b = policy.to_device(&[10_u32, 20]).unwrap();
    let c = policy.to_device(&[100.0_f32, 200.0]).unwrap();

    let (a, b, c) = unzip(zip(a, zip(b, c))).unwrap();

    assert_eq!(a.to_vec().unwrap(), vec![1.0, 2.0]);
    assert_eq!(b.to_vec().unwrap(), vec![10, 20]);
    assert_eq!(c.to_vec().unwrap(), vec![100.0, 200.0]);
}

#[test]
fn gather_and_scatter_accept_wide_device_soas() {
    let policy = policy();
    let indices = policy.to_device(&[2_u32, 1, 0]).unwrap();
    let a = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let c = policy.to_device(&[100.0_f32, 200.0, 300.0]).unwrap();
    let d = policy.to_device(&[1000_u32, 2000, 3000]).unwrap();

    let gathered = gather(vzip4(&a, &b, &c, &d), &indices).unwrap();
    let (a_out, b_out, c_out, d_out) = unzip(gathered).unwrap();
    assert_eq!(a_out.to_vec().unwrap(), vec![3.0, 2.0, 1.0]);
    assert_eq!(b_out.to_vec().unwrap(), vec![30, 20, 10]);
    assert_eq!(c_out.to_vec().unwrap(), vec![300.0, 200.0, 100.0]);
    assert_eq!(d_out.to_vec().unwrap(), vec![3000, 2000, 1000]);

    let scattered = scatter(
        vzip4(&a, &b, &c, &d),
        &indices,
        zip4(
            policy.device_filled(3, 0.0_f32).unwrap(),
            policy.device_filled(3, 0_u32).unwrap(),
            policy.device_filled(3, 0.0_f32).unwrap(),
            policy.device_filled(3, 0_u32).unwrap(),
        ),
    )
    .unwrap();
    let (a_out, b_out, c_out, d_out) = unzip(scattered).unwrap();
    assert_eq!(a_out.to_vec().unwrap(), vec![3.0, 2.0, 1.0]);
    assert_eq!(b_out.to_vec().unwrap(), vec![30, 20, 10]);
    assert_eq!(c_out.to_vec().unwrap(), vec![300.0, 200.0, 100.0]);
    assert_eq!(d_out.to_vec().unwrap(), vec![3000, 2000, 1000]);

    let stencil = policy.to_device(&[1_u32, 0, 1]).unwrap();
    let gathered_if = gather_if(
        vzip4(&a, &b, &c, &d),
        &indices,
        &stencil,
        zip4(
            policy.device_filled(3, 0.0_f32).unwrap(),
            policy.device_filled(3, 0_u32).unwrap(),
            policy.device_filled(3, 0.0_f32).unwrap(),
            policy.device_filled(3, 0_u32).unwrap(),
        ),
        NonZero,
    )
    .unwrap();
    let (a_out, b_out, c_out, d_out) = unzip(gathered_if).unwrap();
    assert_eq!(a_out.to_vec().unwrap(), vec![3.0, 0.0, 1.0]);
    assert_eq!(b_out.to_vec().unwrap(), vec![30, 0, 10]);
    assert_eq!(c_out.to_vec().unwrap(), vec![300.0, 0.0, 100.0]);
    assert_eq!(d_out.to_vec().unwrap(), vec![3000, 0, 1000]);

    let scattered_if = scatter_if(
        vzip4(&a, &b, &c, &d),
        &indices,
        &stencil,
        zip4(
            policy.device_filled(3, 0.0_f32).unwrap(),
            policy.device_filled(3, 0_u32).unwrap(),
            policy.device_filled(3, 0.0_f32).unwrap(),
            policy.device_filled(3, 0_u32).unwrap(),
        ),
        NonZero,
    )
    .unwrap();
    let (a_out, b_out, c_out, d_out) = unzip(scattered_if).unwrap();
    assert_eq!(a_out.to_vec().unwrap(), vec![3.0, 0.0, 1.0]);
    assert_eq!(b_out.to_vec().unwrap(), vec![30, 0, 10]);
    assert_eq!(c_out.to_vec().unwrap(), vec![300.0, 0.0, 100.0]);
    assert_eq!(d_out.to_vec().unwrap(), vec![3000, 0, 1000]);
}

#[test]
fn gather_accepts_sova12_values() {
    let policy = policy();
    let indices = policy.to_device(&[2_u32, 1, 0]).unwrap();
    let a = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = policy.to_device(&[1_u32, 2, 3]).unwrap();
    let c = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let d = policy.to_device(&[1_u32, 2, 3]).unwrap();
    let e = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let f = policy.to_device(&[1_u32, 2, 3]).unwrap();
    let g = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let h = policy.to_device(&[1_u32, 2, 3]).unwrap();
    let i = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let j = policy.to_device(&[1_u32, 2, 3]).unwrap();
    let k = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let l = policy.to_device(&[1_u32, 2, 3]).unwrap();

    let gathered = gather(
        vzip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
        &indices,
    )
    .unwrap();
    let (_a, _b, _c, _d, _e, _f, _g, _h, _i, _j, _k, l_out) = unzip(gathered).unwrap();
    assert_eq!(l_out.to_vec().unwrap(), vec![3, 2, 1]);
}
