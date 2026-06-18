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
fn reverse_accepts_borrowed_heterogeneous_soa12() {
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
    let l = policy.to_device(&[7000_u32, 8000, 9000]).unwrap();

    let reversed = reverse(zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l)).unwrap();
    let (a, b, c, d, e, f, g, h, i, j, k, l) = reversed;

    assert_eq!(a.to_vec().unwrap(), vec![3.0, 2.0, 1.0]);
    assert_eq!(b.to_vec().unwrap(), vec![30, 20, 10]);
    assert_eq!(c.to_vec().unwrap(), vec![300.0, 200.0, 100.0]);
    assert_eq!(d.to_vec().unwrap(), vec![3000, 2000, 1000]);
    assert_eq!(e.to_vec().unwrap(), vec![6.0, 5.0, 4.0]);
    assert_eq!(f.to_vec().unwrap(), vec![60, 50, 40]);
    assert_eq!(g.to_vec().unwrap(), vec![600.0, 500.0, 400.0]);
    assert_eq!(h.to_vec().unwrap(), vec![6000, 5000, 4000]);
    assert_eq!(i.to_vec().unwrap(), vec![9.0, 8.0, 7.0]);
    assert_eq!(j.to_vec().unwrap(), vec![90, 80, 70]);
    assert_eq!(k.to_vec().unwrap(), vec![900.0, 800.0, 700.0]);
    assert_eq!(l.to_vec().unwrap(), vec![9000, 8000, 7000]);
}
