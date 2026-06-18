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
fn set_intersection_accepts_borrowed_heterogeneous_soa12() {
    let policy = policy();
    let la = policy.to_device(&[1.0_f32, 2.0, 4.0]).unwrap();
    let lb = policy.to_device(&[10_u32, 20, 40]).unwrap();
    let lc = policy.to_device(&[100.0_f32, 200.0, 400.0]).unwrap();
    let ld = policy.to_device(&[1000_u32, 2000, 4000]).unwrap();
    let le = policy.to_device(&[11.0_f32, 12.0, 14.0]).unwrap();
    let lf = policy.to_device(&[110_u32, 120, 140]).unwrap();
    let lg = policy.to_device(&[1100.0_f32, 1200.0, 1400.0]).unwrap();
    let lh = policy.to_device(&[11000_u32, 12000, 14000]).unwrap();
    let li = policy.to_device(&[21.0_f32, 22.0, 24.0]).unwrap();
    let lj = policy.to_device(&[210_u32, 220, 240]).unwrap();
    let lk = policy.to_device(&[2100.0_f32, 2200.0, 2400.0]).unwrap();
    let ll = policy.to_device(&[21000_u32, 22000, 24000]).unwrap();

    let ra = policy.to_device(&[2.0_f32, 3.0, 4.0]).unwrap();
    let rb = policy.to_device(&[20_u32, 30, 40]).unwrap();
    let rc = policy.to_device(&[200.0_f32, 300.0, 400.0]).unwrap();
    let rd = policy.to_device(&[2000_u32, 3000, 4000]).unwrap();
    let re = policy.to_device(&[12.0_f32, 13.0, 14.0]).unwrap();
    let rf = policy.to_device(&[120_u32, 130, 140]).unwrap();
    let rg = policy.to_device(&[1200.0_f32, 1300.0, 1400.0]).unwrap();
    let rh = policy.to_device(&[12000_u32, 13000, 14000]).unwrap();
    let ri = policy.to_device(&[22.0_f32, 23.0, 24.0]).unwrap();
    let rj = policy.to_device(&[220_u32, 230, 240]).unwrap();
    let rk = policy.to_device(&[2200.0_f32, 2300.0, 2400.0]).unwrap();
    let rl = policy.to_device(&[22000_u32, 23000, 24000]).unwrap();

    let output = set_intersection(
        zip12(&la, &lb, &lc, &ld, &le, &lf, &lg, &lh, &li, &lj, &lk, &ll),
        zip12(&ra, &rb, &rc, &rd, &re, &rf, &rg, &rh, &ri, &rj, &rk, &rl),
        Tuple12MixedLess,
    )
    .unwrap();
    let (a, b, c, d, e, f, g, h, i, j, k, l) = output;

    assert_eq!(a.to_vec().unwrap(), vec![2.0, 4.0]);
    assert_eq!(b.to_vec().unwrap(), vec![20, 40]);
    assert_eq!(c.to_vec().unwrap(), vec![200.0, 400.0]);
    assert_eq!(d.to_vec().unwrap(), vec![2000, 4000]);
    assert_eq!(e.to_vec().unwrap(), vec![12.0, 14.0]);
    assert_eq!(f.to_vec().unwrap(), vec![120, 140]);
    assert_eq!(g.to_vec().unwrap(), vec![1200.0, 1400.0]);
    assert_eq!(h.to_vec().unwrap(), vec![12000, 14000]);
    assert_eq!(i.to_vec().unwrap(), vec![22.0, 24.0]);
    assert_eq!(j.to_vec().unwrap(), vec![220, 240]);
    assert_eq!(k.to_vec().unwrap(), vec![2200.0, 2400.0]);
    assert_eq!(l.to_vec().unwrap(), vec![22000, 24000]);
}

#[test]
fn set_intersection_preserves_multiplicity_for_borrowed_heterogeneous_soa12() {
    let policy = policy();
    let (la, lb, lc, ld, le, lf, lg, lh, li, lj, lk, ll) = soa12_rows!(policy; [1, 2, 2, 2, 4]);
    let (ra, rb, rc, rd, re, rf, rg, rh, ri, rj, rk, rl) = soa12_rows!(policy; [2, 3]);

    let output = set_intersection(
        zip12(&la, &lb, &lc, &ld, &le, &lf, &lg, &lh, &li, &lj, &lk, &ll),
        zip12(&ra, &rb, &rc, &rd, &re, &rf, &rg, &rh, &ri, &rj, &rk, &rl),
        Tuple12MixedLess,
    )
    .unwrap();

    assert_soa12_rows!(output; [2]);
}
