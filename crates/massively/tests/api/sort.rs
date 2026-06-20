use crate::common::*;

#[allow(unused_macros)]
macro_rules! soa12_rows {
    ($policy:expr; [$( $x:expr ),+ $(,)?]) => {{
        let a = $policy.to_device(&[$(($x as f32)),+]).unwrap();
        let b = $policy.to_device(&[$(($x as u32) * 10),+]).unwrap();
        let c = $policy.to_device(&[$(($x as f32) * 100.0),+]).unwrap();
        let d = $policy.to_device(&[$(($x as u32) * 1000),+]).unwrap();
        let e = $policy.to_device(&[$(($x as f32) + 10.0),+]).unwrap();
        let f = $policy.to_device(&[$(($x as u32) + 100),+]).unwrap();
        let g = $policy.to_device(&[$(($x as f32) + 1000.0),+]).unwrap();
        let h = $policy.to_device(&[$(($x as u32) + 10000),+]).unwrap();
        let i = $policy.to_device(&[$(($x as f32) + 20.0),+]).unwrap();
        let j = $policy.to_device(&[$(($x as u32) + 200),+]).unwrap();
        let k = $policy.to_device(&[$(($x as f32) + 2000.0),+]).unwrap();
        let l = $policy.to_device(&[$(($x as u32) + 20000),+]).unwrap();
        (a, b, c, d, e, f, g, h, i, j, k, l)
    }};
}

#[allow(unused_macros)]
macro_rules! assert_soa12_rows {
    ($output:expr; [$( $x:expr ),* $(,)?]) => {{
        let (a, b, c, d, e, f, g, h, i, j, k, l) = $output;
        assert_eq!(a.to_vec().unwrap(), vec![$(($x as f32)),*]);
        assert_eq!(b.to_vec().unwrap(), vec![$(($x as u32) * 10),*]);
        assert_eq!(c.to_vec().unwrap(), vec![$(($x as f32) * 100.0),*]);
        assert_eq!(d.to_vec().unwrap(), vec![$(($x as u32) * 1000),*]);
        assert_eq!(e.to_vec().unwrap(), vec![$(($x as f32) + 10.0),*]);
        assert_eq!(f.to_vec().unwrap(), vec![$(($x as u32) + 100),*]);
        assert_eq!(g.to_vec().unwrap(), vec![$(($x as f32) + 1000.0),*]);
        assert_eq!(h.to_vec().unwrap(), vec![$(($x as u32) + 10000),*]);
        assert_eq!(i.to_vec().unwrap(), vec![$(($x as f32) + 20.0),*]);
        assert_eq!(j.to_vec().unwrap(), vec![$(($x as u32) + 200),*]);
        assert_eq!(k.to_vec().unwrap(), vec![$(($x as f32) + 2000.0),*]);
        assert_eq!(l.to_vec().unwrap(), vec![$(($x as u32) + 20000),*]);
    }};
}

#[test]
fn sort_returns_device_storage() {
    let policy = policy();
    let x = policy.to_device(&[3.0_f32, 1.0, 2.0]).unwrap();

    let sorted = sort((&x,), Less).unwrap();
    let (sorted,) = sorted;

    assert_eq!(sorted.to_vec().unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(x.to_vec().unwrap(), vec![3.0, 1.0, 2.0]);
}

#[test]
fn tuple_sort_preserves_soa_components() {
    let policy = policy();
    let x = policy.to_device(&[3.0_f32, 1.0, 2.0]).unwrap();
    let y = policy.to_device(&[30_u32, 10, 20]).unwrap();

    let sorted = sort((&x, &y), MixedTupleLess).unwrap();
    let (x, y) = sorted;

    assert_eq!(x.to_vec().unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(y.to_vec().unwrap(), vec![10, 20, 30]);
}

#[test]
fn sort_accepts_heterogeneous_tuple_comparators_for_two_and_three_columns() {
    let policy = policy();
    let values = policy.to_device(&[2.0_f32, 1.0, 2.0, 3.0]).unwrap();
    let tags = policy.to_device(&[20_u32, 30, 10, 40]).unwrap();

    let sorted = sort((&values, &tags), MixedTupleLess).unwrap();
    let (values, tags) = sorted;
    assert_eq!(values.to_vec().unwrap(), vec![1.0, 2.0, 2.0, 3.0]);
    assert_eq!(tags.to_vec().unwrap(), vec![30, 10, 20, 40]);

    let values = policy.to_device(&[2.0_f32, 1.0, 4.0, 3.0]).unwrap();
    let tags = policy.to_device(&[20_u32, 10, 20, 10]).unwrap();
    let payload = policy.to_device(&[200.0_f32, 100.0, 400.0, 300.0]).unwrap();

    let sorted = sort((&values, &tags, &payload), MixedTuple3Less).unwrap();
    let (values, tags, payload) = sorted;
    assert_eq!(values.to_vec().unwrap(), vec![1.0, 3.0, 2.0, 4.0]);
    assert_eq!(tags.to_vec().unwrap(), vec![10, 10, 20, 20]);
    assert_eq!(payload.to_vec().unwrap(), vec![100.0, 300.0, 200.0, 400.0]);
}

#[cfg(any())]
#[test]
fn tuple_sort_accepts_wide_borrowed_soas() {
    let policy = policy();
    let a = policy.to_device(&[3.0_f32, 1.0, 2.0]).unwrap();
    let b = policy.to_device(&[30.0_f32, 10.0, 20.0]).unwrap();
    let c = policy.to_device(&[300.0_f32, 100.0, 200.0]).unwrap();
    let d = policy.to_device(&[3000.0_f32, 1000.0, 2000.0]).unwrap();

    let sorted = sort(zip4(&a, &b, &c, &d), Tuple4Less).unwrap();
    let (a, b, c, d) = sorted;

    assert_eq!(a.to_vec().unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(b.to_vec().unwrap(), vec![10.0, 20.0, 30.0]);
    assert_eq!(c.to_vec().unwrap(), vec![100.0, 200.0, 300.0]);
    assert_eq!(d.to_vec().unwrap(), vec![1000.0, 2000.0, 3000.0]);
}

#[cfg(any())]
#[test]
fn tuple_sort_accepts_soa12() {
    let policy = policy();
    let a = policy.to_device(&[3.0_f32, 1.0, 2.0, 2.0]).unwrap();
    let b = policy.to_device(&[30_u32, 10, 25, 20]).unwrap();
    let c = policy.to_device(&[300.0_f32, 100.0, 250.0, 200.0]).unwrap();
    let d = policy.to_device(&[30000_u32, 10000, 25000, 20000]).unwrap();
    let e = policy.to_device(&[6.0_f32, 4.0, 5.5, 5.0]).unwrap();
    let f = policy.to_device(&[60_u32, 40, 55, 50]).unwrap();
    let g = policy.to_device(&[600.0_f32, 400.0, 550.0, 500.0]).unwrap();
    let h = policy.to_device(&[6000_u32, 4000, 5500, 5000]).unwrap();
    let i = policy.to_device(&[9.0_f32, 7.0, 8.5, 8.0]).unwrap();
    let j = policy.to_device(&[90_u32, 70, 85, 80]).unwrap();
    let k = policy.to_device(&[900.0_f32, 700.0, 850.0, 800.0]).unwrap();
    let l = policy.to_device(&[3000_u32, 1000, 2500, 2000]).unwrap();

    let sorted = sort(
        zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
        Tuple12MixedLess,
    )
    .unwrap();
    let (a, b, c, d, e, f, g, h, i, j, k, l) = sorted;

    assert_eq!(a.to_vec().unwrap(), vec![1.0, 2.0, 2.0, 3.0]);
    assert_eq!(b.to_vec().unwrap(), vec![10, 20, 25, 30]);
    assert_eq!(c.to_vec().unwrap(), vec![100.0, 200.0, 250.0, 300.0]);
    assert_eq!(d.to_vec().unwrap(), vec![10000, 20000, 25000, 30000]);
    assert_eq!(e.to_vec().unwrap(), vec![4.0, 5.0, 5.5, 6.0]);
    assert_eq!(f.to_vec().unwrap(), vec![40, 50, 55, 60]);
    assert_eq!(g.to_vec().unwrap(), vec![400.0, 500.0, 550.0, 600.0]);
    assert_eq!(h.to_vec().unwrap(), vec![4000, 5000, 5500, 6000]);
    assert_eq!(i.to_vec().unwrap(), vec![7.0, 8.0, 8.5, 9.0]);
    assert_eq!(j.to_vec().unwrap(), vec![70, 80, 85, 90]);
    assert_eq!(k.to_vec().unwrap(), vec![700.0, 800.0, 850.0, 900.0]);
    assert_eq!(l.to_vec().unwrap(), vec![1000, 2000, 2500, 3000]);
}
