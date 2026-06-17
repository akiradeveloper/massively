mod common;
use common::*;

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

    let sorted = sort(&x, Less).unwrap();
    let sorted = sorted;

    assert_eq!(sorted.to_vec().unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(x.to_vec().unwrap(), vec![3.0, 1.0, 2.0]);
}

#[test]
fn tuple_sort_preserves_soa_components() {
    let policy = policy();
    let x = policy.to_device(&[3.0_f32, 1.0, 2.0]).unwrap();
    let y = policy.to_device(&[30_u32, 10, 20]).unwrap();

    let sorted = sort(zip(&x, &y), MixedTupleLess).unwrap();
    let (x, y) = sorted;

    assert_eq!(x.to_vec().unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(y.to_vec().unwrap(), vec![10, 20, 30]);
}

#[test]
fn sort_accepts_heterogeneous_tuple_comparators_for_two_and_three_columns() {
    let policy = policy();
    let values = policy.to_device(&[2.0_f32, 1.0, 2.0, 3.0]).unwrap();
    let tags = policy.to_device(&[20_u32, 30, 10, 40]).unwrap();

    let sorted = sort(zip(&values, &tags), MixedTupleLess).unwrap();
    let (values, tags) = sorted;
    assert_eq!(values.to_vec().unwrap(), vec![1.0, 2.0, 2.0, 3.0]);
    assert_eq!(tags.to_vec().unwrap(), vec![30, 10, 20, 40]);

    let values = policy.to_device(&[2.0_f32, 1.0, 4.0, 3.0]).unwrap();
    let tags = policy.to_device(&[20_u32, 10, 20, 10]).unwrap();
    let payload = policy.to_device(&[200.0_f32, 100.0, 400.0, 300.0]).unwrap();

    let sorted = sort(zip3(&values, &tags, &payload), MixedTuple3Less).unwrap();
    let (values, tags, payload) = sorted;
    assert_eq!(values.to_vec().unwrap(), vec![1.0, 3.0, 2.0, 4.0]);
    assert_eq!(tags.to_vec().unwrap(), vec![10, 10, 20, 20]);
    assert_eq!(payload.to_vec().unwrap(), vec![100.0, 300.0, 200.0, 400.0]);
}

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

#[test]
fn set_union_accepts_borrowed_heterogeneous_soa12() {
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

    let ra = policy.to_device(&[2.0_f32, 3.0, 5.0]).unwrap();
    let rb = policy.to_device(&[20_u32, 30, 50]).unwrap();
    let rc = policy.to_device(&[200.0_f32, 300.0, 500.0]).unwrap();
    let rd = policy.to_device(&[2000_u32, 3000, 5000]).unwrap();
    let re = policy.to_device(&[12.0_f32, 13.0, 15.0]).unwrap();
    let rf = policy.to_device(&[120_u32, 130, 150]).unwrap();
    let rg = policy.to_device(&[1200.0_f32, 1300.0, 1500.0]).unwrap();
    let rh = policy.to_device(&[12000_u32, 13000, 15000]).unwrap();
    let ri = policy.to_device(&[22.0_f32, 23.0, 25.0]).unwrap();
    let rj = policy.to_device(&[220_u32, 230, 250]).unwrap();
    let rk = policy.to_device(&[2200.0_f32, 2300.0, 2500.0]).unwrap();
    let rl = policy.to_device(&[22000_u32, 23000, 25000]).unwrap();

    let output = set_union(
        zip12(&la, &lb, &lc, &ld, &le, &lf, &lg, &lh, &li, &lj, &lk, &ll),
        zip12(&ra, &rb, &rc, &rd, &re, &rf, &rg, &rh, &ri, &rj, &rk, &rl),
        Tuple12MixedLess,
    )
    .unwrap();
    let (a, b, c, d, e, f, g, h, i, j, k, l) = output;

    assert_eq!(a.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(b.to_vec().unwrap(), vec![10, 20, 30, 40, 50]);
    assert_eq!(c.to_vec().unwrap(), vec![100.0, 200.0, 300.0, 400.0, 500.0]);
    assert_eq!(d.to_vec().unwrap(), vec![1000, 2000, 3000, 4000, 5000]);
    assert_eq!(e.to_vec().unwrap(), vec![11.0, 12.0, 13.0, 14.0, 15.0]);
    assert_eq!(f.to_vec().unwrap(), vec![110, 120, 130, 140, 150]);
    assert_eq!(
        g.to_vec().unwrap(),
        vec![1100.0, 1200.0, 1300.0, 1400.0, 1500.0]
    );
    assert_eq!(h.to_vec().unwrap(), vec![11000, 12000, 13000, 14000, 15000]);
    assert_eq!(i.to_vec().unwrap(), vec![21.0, 22.0, 23.0, 24.0, 25.0]);
    assert_eq!(j.to_vec().unwrap(), vec![210, 220, 230, 240, 250]);
    assert_eq!(
        k.to_vec().unwrap(),
        vec![2100.0, 2200.0, 2300.0, 2400.0, 2500.0]
    );
    assert_eq!(l.to_vec().unwrap(), vec![21000, 22000, 23000, 24000, 25000]);
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
fn set_difference_accepts_borrowed_heterogeneous_soa12() {
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

    let output = set_difference(
        zip12(&la, &lb, &lc, &ld, &le, &lf, &lg, &lh, &li, &lj, &lk, &ll),
        zip12(&ra, &rb, &rc, &rd, &re, &rf, &rg, &rh, &ri, &rj, &rk, &rl),
        Tuple12MixedLess,
    )
    .unwrap();
    let (a, b, c, d, e, f, g, h, i, j, k, l) = output;

    assert_eq!(a.to_vec().unwrap(), vec![1.0]);
    assert_eq!(b.to_vec().unwrap(), vec![10]);
    assert_eq!(c.to_vec().unwrap(), vec![100.0]);
    assert_eq!(d.to_vec().unwrap(), vec![1000]);
    assert_eq!(e.to_vec().unwrap(), vec![11.0]);
    assert_eq!(f.to_vec().unwrap(), vec![110]);
    assert_eq!(g.to_vec().unwrap(), vec![1100.0]);
    assert_eq!(h.to_vec().unwrap(), vec![11000]);
    assert_eq!(i.to_vec().unwrap(), vec![21.0]);
    assert_eq!(j.to_vec().unwrap(), vec![210]);
    assert_eq!(k.to_vec().unwrap(), vec![2100.0]);
    assert_eq!(l.to_vec().unwrap(), vec![21000]);
}

#[test]
fn set_union_preserves_multiplicity_for_borrowed_heterogeneous_soa12() {
    let policy = policy();
    let (la, lb, lc, ld, le, lf, lg, lh, li, lj, lk, ll) = soa12_rows!(policy; [1, 2, 2, 4]);
    let (ra, rb, rc, rd, re, rf, rg, rh, ri, rj, rk, rl) = soa12_rows!(policy; [2, 2, 2, 3]);

    let output = set_union(
        zip12(&la, &lb, &lc, &ld, &le, &lf, &lg, &lh, &li, &lj, &lk, &ll),
        zip12(&ra, &rb, &rc, &rd, &re, &rf, &rg, &rh, &ri, &rj, &rk, &rl),
        Tuple12MixedLess,
    )
    .unwrap();

    assert_soa12_rows!(output; [1, 2, 2, 2, 3, 4]);
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

#[test]
fn set_difference_preserves_multiplicity_for_borrowed_heterogeneous_soa12() {
    let policy = policy();
    let (la, lb, lc, ld, le, lf, lg, lh, li, lj, lk, ll) = soa12_rows!(policy; [1, 2, 2, 2, 4]);
    let (ra, rb, rc, rd, re, rf, rg, rh, ri, rj, rk, rl) = soa12_rows!(policy; [2, 3]);

    let output = set_difference(
        zip12(&la, &lb, &lc, &ld, &le, &lf, &lg, &lh, &li, &lj, &lk, &ll),
        zip12(&ra, &rb, &rc, &rd, &re, &rf, &rg, &rh, &ri, &rj, &rk, &rl),
        Tuple12MixedLess,
    )
    .unwrap();

    assert_soa12_rows!(output; [1, 2, 2, 4]);
}

#[test]
fn sort_by_key_accepts_wide_borrowed_soas() {
    let policy = policy();
    let keys = policy.to_device(&[2_u32, 0, 1]).unwrap();
    let a = policy.to_device(&[20.0_f32, 0.0, 10.0]).unwrap();
    let b = policy.to_device(&[200_u32, 0, 100]).unwrap();
    let c = policy.to_device(&[2000.0_f32, 0.0, 1000.0]).unwrap();
    let d = policy.to_device(&[20000_u32, 0, 10000]).unwrap();

    let (sorted_keys, values) = sort_by_key(&keys, zip4(&a, &b, &c, &d), LessU32).unwrap();
    let (out_a, out_b, out_c, out_d) = values;

    assert_eq!(sorted_keys.to_vec().unwrap(), vec![0, 1, 2]);
    assert_eq!(out_a.to_vec().unwrap(), vec![0.0, 10.0, 20.0]);
    assert_eq!(out_b.to_vec().unwrap(), vec![0, 100, 200]);
    assert_eq!(out_c.to_vec().unwrap(), vec![0.0, 1000.0, 2000.0]);
    assert_eq!(out_d.to_vec().unwrap(), vec![0, 10000, 20000]);
    assert_eq!(keys.to_vec().unwrap(), vec![2, 0, 1]);
    assert_eq!(a.to_vec().unwrap(), vec![20.0, 0.0, 10.0]);
}

#[test]
fn sort_by_key_accepts_soa12_values() {
    let policy = policy();
    let keys = policy.to_device(&[2_u32, 0, 1]).unwrap();
    let a = policy.to_device(&[20.0_f32, 0.0, 10.0]).unwrap();
    let b = policy.to_device(&[20_u32, 0, 10]).unwrap();
    let c = policy.to_device(&[200.0_f32, 0.0, 100.0]).unwrap();
    let d = policy.to_device(&[200_u32, 0, 100]).unwrap();
    let e = policy.to_device(&[2000.0_f32, 0.0, 1000.0]).unwrap();
    let f = policy.to_device(&[2000_u32, 0, 1000]).unwrap();
    let g = policy.to_device(&[5.0_f32, 3.0, 4.0]).unwrap();
    let h = policy.to_device(&[50_u32, 30, 40]).unwrap();
    let i = policy.to_device(&[500.0_f32, 300.0, 400.0]).unwrap();
    let j = policy.to_device(&[5000_u32, 3000, 4000]).unwrap();
    let k = policy.to_device(&[8.0_f32, 6.0, 7.0]).unwrap();
    let l = policy.to_device(&[80_u32, 60, 70]).unwrap();

    let (keys, values) = sort_by_key(
        &keys,
        zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
        LessU32,
    )
    .unwrap();
    let keys = keys;
    let (a, b, c, d, e, f, g, h, i, j, k, l) = values;

    assert_eq!(keys.to_vec().unwrap(), vec![0, 1, 2]);
    assert_eq!(a.to_vec().unwrap(), vec![0.0, 10.0, 20.0]);
    assert_eq!(b.to_vec().unwrap(), vec![0, 10, 20]);
    assert_eq!(c.to_vec().unwrap(), vec![0.0, 100.0, 200.0]);
    assert_eq!(d.to_vec().unwrap(), vec![0, 100, 200]);
    assert_eq!(e.to_vec().unwrap(), vec![0.0, 1000.0, 2000.0]);
    assert_eq!(f.to_vec().unwrap(), vec![0, 1000, 2000]);
    assert_eq!(g.to_vec().unwrap(), vec![3.0, 4.0, 5.0]);
    assert_eq!(h.to_vec().unwrap(), vec![30, 40, 50]);
    assert_eq!(i.to_vec().unwrap(), vec![300.0, 400.0, 500.0]);
    assert_eq!(j.to_vec().unwrap(), vec![3000, 4000, 5000]);
    assert_eq!(k.to_vec().unwrap(), vec![6.0, 7.0, 8.0]);
    assert_eq!(l.to_vec().unwrap(), vec![60, 70, 80]);
}

#[test]
fn merge_by_key_accepts_wide_soa_values() {
    let policy = policy();
    let left_keys = policy.to_device(&[0_u32, 2]).unwrap();
    let right_keys = policy.to_device(&[1_u32, 3]).unwrap();
    let left_a = policy.to_device(&[0.0_f32, 20.0]).unwrap();
    let left_b = policy.to_device(&[0_u32, 200]).unwrap();
    let left_c = policy.to_device(&[0.0_f32, 2000.0]).unwrap();
    let left_d = policy.to_device(&[0_u32, 20000]).unwrap();
    let right_a = policy.to_device(&[10.0_f32, 30.0]).unwrap();
    let right_b = policy.to_device(&[100_u32, 300]).unwrap();
    let right_c = policy.to_device(&[1000.0_f32, 3000.0]).unwrap();
    let right_d = policy.to_device(&[10000_u32, 30000]).unwrap();

    let (keys, values) = merge_by_key(
        &left_keys,
        zip4(&left_a, &left_b, &left_c, &left_d),
        &right_keys,
        zip4(&right_a, &right_b, &right_c, &right_d),
        LessU32,
    )
    .unwrap();
    let keys = keys;
    let (a, b, c, d) = values;

    assert_eq!(keys.to_vec().unwrap(), vec![0, 1, 2, 3]);
    assert_eq!(a.to_vec().unwrap(), vec![0.0, 10.0, 20.0, 30.0]);
    assert_eq!(b.to_vec().unwrap(), vec![0, 100, 200, 300]);
    assert_eq!(c.to_vec().unwrap(), vec![0.0, 1000.0, 2000.0, 3000.0]);
    assert_eq!(d.to_vec().unwrap(), vec![0, 10000, 20000, 30000]);
}

#[test]
fn merge_by_key_accepts_soa12_values() {
    let policy = policy();
    let left_keys = policy.to_device(&[0_u32, 2]).unwrap();
    let right_keys = policy.to_device(&[1_u32, 3]).unwrap();
    let la = policy.to_device(&[0.0_f32, 20.0]).unwrap();
    let lb = policy.to_device(&[0_u32, 20]).unwrap();
    let lc = policy.to_device(&[0.0_f32, 200.0]).unwrap();
    let ld = policy.to_device(&[0_u32, 200]).unwrap();
    let le = policy.to_device(&[0.0_f32, 2000.0]).unwrap();
    let lf = policy.to_device(&[0_u32, 2000]).unwrap();
    let lg = policy.to_device(&[4.0_f32, 6.0]).unwrap();
    let lh = policy.to_device(&[40_u32, 60]).unwrap();
    let li = policy.to_device(&[400.0_f32, 600.0]).unwrap();
    let lj = policy.to_device(&[4000_u32, 6000]).unwrap();
    let lk = policy.to_device(&[7.0_f32, 9.0]).unwrap();
    let ll = policy.to_device(&[70_u32, 90]).unwrap();
    let ra = policy.to_device(&[10.0_f32, 30.0]).unwrap();
    let rb = policy.to_device(&[10_u32, 30]).unwrap();
    let rc = policy.to_device(&[100.0_f32, 300.0]).unwrap();
    let rd = policy.to_device(&[100_u32, 300]).unwrap();
    let re = policy.to_device(&[1000.0_f32, 3000.0]).unwrap();
    let rf = policy.to_device(&[1000_u32, 3000]).unwrap();
    let rg = policy.to_device(&[5.0_f32, 7.0]).unwrap();
    let rh = policy.to_device(&[50_u32, 70]).unwrap();
    let ri = policy.to_device(&[500.0_f32, 700.0]).unwrap();
    let rj = policy.to_device(&[5000_u32, 7000]).unwrap();
    let rk = policy.to_device(&[8.0_f32, 10.0]).unwrap();
    let rl = policy.to_device(&[80_u32, 100]).unwrap();

    let (keys, values) = merge_by_key(
        &left_keys,
        zip12(&la, &lb, &lc, &ld, &le, &lf, &lg, &lh, &li, &lj, &lk, &ll),
        &right_keys,
        zip12(&ra, &rb, &rc, &rd, &re, &rf, &rg, &rh, &ri, &rj, &rk, &rl),
        LessU32,
    )
    .unwrap();
    let keys = keys;
    let (a, b, c, d, e, f, g, h, i, j, k, l) = values;

    assert_eq!(keys.to_vec().unwrap(), vec![0, 1, 2, 3]);
    assert_eq!(a.to_vec().unwrap(), vec![0.0, 10.0, 20.0, 30.0]);
    assert_eq!(b.to_vec().unwrap(), vec![0, 10, 20, 30]);
    assert_eq!(c.to_vec().unwrap(), vec![0.0, 100.0, 200.0, 300.0]);
    assert_eq!(d.to_vec().unwrap(), vec![0, 100, 200, 300]);
    assert_eq!(e.to_vec().unwrap(), vec![0.0, 1000.0, 2000.0, 3000.0]);
    assert_eq!(f.to_vec().unwrap(), vec![0, 1000, 2000, 3000]);
    assert_eq!(g.to_vec().unwrap(), vec![4.0, 5.0, 6.0, 7.0]);
    assert_eq!(h.to_vec().unwrap(), vec![40, 50, 60, 70]);
    assert_eq!(i.to_vec().unwrap(), vec![400.0, 500.0, 600.0, 700.0]);
    assert_eq!(j.to_vec().unwrap(), vec![4000, 5000, 6000, 7000]);
    assert_eq!(k.to_vec().unwrap(), vec![7.0, 8.0, 9.0, 10.0]);
    assert_eq!(l.to_vec().unwrap(), vec![70, 80, 90, 100]);
}

#[test]
fn merge_by_key_accepts_soa12_values_with_equal_keys_and_uneven_lengths() {
    let policy = policy();
    let left_keys = policy.to_device(&[0_u32, 2, 2, 5]).unwrap();
    let right_keys = policy.to_device(&[1_u32, 2, 4]).unwrap();
    let la = policy.to_device(&[0.0_f32, 20.0, 21.0, 50.0]).unwrap();
    let lb = policy.to_device(&[0_u32, 20, 21, 50]).unwrap();
    let lc = policy.to_device(&[100.0_f32, 120.0, 121.0, 150.0]).unwrap();
    let ld = policy.to_device(&[100_u32, 120, 121, 150]).unwrap();
    let le = policy.to_device(&[200.0_f32, 220.0, 221.0, 250.0]).unwrap();
    let lf = policy.to_device(&[200_u32, 220, 221, 250]).unwrap();
    let lg = policy.to_device(&[300.0_f32, 320.0, 321.0, 350.0]).unwrap();
    let lh = policy.to_device(&[300_u32, 320, 321, 350]).unwrap();
    let li = policy.to_device(&[400.0_f32, 420.0, 421.0, 450.0]).unwrap();
    let lj = policy.to_device(&[400_u32, 420, 421, 450]).unwrap();
    let lk = policy.to_device(&[500.0_f32, 520.0, 521.0, 550.0]).unwrap();
    let ll = policy.to_device(&[500_u32, 520, 521, 550]).unwrap();
    let ra = policy.to_device(&[10.0_f32, 22.0, 40.0]).unwrap();
    let rb = policy.to_device(&[10_u32, 22, 40]).unwrap();
    let rc = policy.to_device(&[110.0_f32, 122.0, 140.0]).unwrap();
    let rd = policy.to_device(&[110_u32, 122, 140]).unwrap();
    let re = policy.to_device(&[210.0_f32, 222.0, 240.0]).unwrap();
    let rf = policy.to_device(&[210_u32, 222, 240]).unwrap();
    let rg = policy.to_device(&[310.0_f32, 322.0, 340.0]).unwrap();
    let rh = policy.to_device(&[310_u32, 322, 340]).unwrap();
    let ri = policy.to_device(&[410.0_f32, 422.0, 440.0]).unwrap();
    let rj = policy.to_device(&[410_u32, 422, 440]).unwrap();
    let rk = policy.to_device(&[510.0_f32, 522.0, 540.0]).unwrap();
    let rl = policy.to_device(&[510_u32, 522, 540]).unwrap();

    let (keys, values) = merge_by_key(
        &left_keys,
        zip12(&la, &lb, &lc, &ld, &le, &lf, &lg, &lh, &li, &lj, &lk, &ll),
        &right_keys,
        zip12(&ra, &rb, &rc, &rd, &re, &rf, &rg, &rh, &ri, &rj, &rk, &rl),
        LessU32,
    )
    .unwrap();
    let keys = keys;
    let (a, b, c, d, e, f, g, h, i, j, k, l) = values;

    assert_eq!(keys.to_vec().unwrap(), vec![0, 1, 2, 2, 2, 4, 5]);
    assert_eq!(
        a.to_vec().unwrap(),
        vec![0.0, 10.0, 20.0, 21.0, 22.0, 40.0, 50.0]
    );
    assert_eq!(b.to_vec().unwrap(), vec![0, 10, 20, 21, 22, 40, 50]);
    assert_eq!(
        c.to_vec().unwrap(),
        vec![100.0, 110.0, 120.0, 121.0, 122.0, 140.0, 150.0]
    );
    assert_eq!(d.to_vec().unwrap(), vec![100, 110, 120, 121, 122, 140, 150]);
    assert_eq!(
        e.to_vec().unwrap(),
        vec![200.0, 210.0, 220.0, 221.0, 222.0, 240.0, 250.0]
    );
    assert_eq!(f.to_vec().unwrap(), vec![200, 210, 220, 221, 222, 240, 250]);
    assert_eq!(
        g.to_vec().unwrap(),
        vec![300.0, 310.0, 320.0, 321.0, 322.0, 340.0, 350.0]
    );
    assert_eq!(h.to_vec().unwrap(), vec![300, 310, 320, 321, 322, 340, 350]);
    assert_eq!(
        i.to_vec().unwrap(),
        vec![400.0, 410.0, 420.0, 421.0, 422.0, 440.0, 450.0]
    );
    assert_eq!(j.to_vec().unwrap(), vec![400, 410, 420, 421, 422, 440, 450]);
    assert_eq!(
        k.to_vec().unwrap(),
        vec![500.0, 510.0, 520.0, 521.0, 522.0, 540.0, 550.0]
    );
    assert_eq!(l.to_vec().unwrap(), vec![500, 510, 520, 521, 522, 540, 550]);
}
