use crate::common::*;

#[allow(unused_macros)]
macro_rules! soa12_rows {
    ($exec:expr; [$( $x:expr ),+ $(,)?]) => {{
        let a = $exec.to_device(&[$(($x as f32)),+]).unwrap();
        let b = $exec.to_device(&[$(($x as u32) * 10),+]).unwrap();
        let c = $exec.to_device(&[$(($x as f32) * 100.0),+]).unwrap();
        let d = $exec.to_device(&[$(($x as u32) * 1000),+]).unwrap();
        let e = $exec.to_device(&[$(($x as f32) + 10.0),+]).unwrap();
        let f = $exec.to_device(&[$(($x as u32) + 100),+]).unwrap();
        let g = $exec.to_device(&[$(($x as f32) + 1000.0),+]).unwrap();
        let h = $exec.to_device(&[$(($x as u32) + 10000),+]).unwrap();
        let i = $exec.to_device(&[$(($x as f32) + 20.0),+]).unwrap();
        let j = $exec.to_device(&[$(($x as u32) + 200),+]).unwrap();
        let k = $exec.to_device(&[$(($x as f32) + 2000.0),+]).unwrap();
        let l = $exec.to_device(&[$(($x as u32) + 20000),+]).unwrap();
        (a, b, c, d, e, f, g, h, i, j, k, l)
    }};
}

#[allow(unused_macros)]
macro_rules! assert_soa12_rows {
    ($output:expr; [$( $x:expr ),* $(,)?]) => {{
        let (a, b, c, d, e, f, g, h, i, j, k, l) = $output;
        assert_eq!(exec.to_host(&a).unwrap(), vec![$(($x as f32)),*]);
        assert_eq!(exec.to_host(&b).unwrap(), vec![$(($x as u32) * 10),*]);
        assert_eq!(exec.to_host(&c).unwrap(), vec![$(($x as f32) * 100.0),*]);
        assert_eq!(exec.to_host(&d).unwrap(), vec![$(($x as u32) * 1000),*]);
        assert_eq!(exec.to_host(&e).unwrap(), vec![$(($x as f32) + 10.0),*]);
        assert_eq!(exec.to_host(&f).unwrap(), vec![$(($x as u32) + 100),*]);
        assert_eq!(exec.to_host(&g).unwrap(), vec![$(($x as f32) + 1000.0),*]);
        assert_eq!(exec.to_host(&h).unwrap(), vec![$(($x as u32) + 10000),*]);
        assert_eq!(exec.to_host(&i).unwrap(), vec![$(($x as f32) + 20.0),*]);
        assert_eq!(exec.to_host(&j).unwrap(), vec![$(($x as u32) + 200),*]);
        assert_eq!(exec.to_host(&k).unwrap(), vec![$(($x as f32) + 2000.0),*]);
        assert_eq!(exec.to_host(&l).unwrap(), vec![$(($x as u32) + 20000),*]);
    }};
}

#[test]
fn set_union_accepts_borrowed_tuple_columns() {
    let exec = exec();
    let left_a = exec.to_device(&[1.0_f32, 2.0, 4.0]).unwrap();
    let left_b = exec.to_device(&[10_u32, 20, 40]).unwrap();
    let right_a = exec.to_device(&[2.0_f32, 3.0]).unwrap();
    let right_b = exec.to_device(&[20_u32, 30]).unwrap();

    let output = set_union(
        &exec,
        massively::SoA2(left_a.slice(..), left_b.slice(..)),
        massively::SoA2(right_a.slice(..), right_b.slice(..)),
        MixedTupleLess,
    )
    .unwrap();
    let massively::SoA2(a, b) = output;

    assert_eq!(exec.to_host(&a).unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![10, 20, 30, 40]);
}

#[cfg(any())]
#[test]
fn set_union_accepts_borrowed_heterogeneous_soa12() {
    let exec = exec();
    let la = exec.to_device(&[1.0_f32, 2.0, 4.0]).unwrap();
    let lb = exec.to_device(&[10_u32, 20, 40]).unwrap();
    let lc = exec.to_device(&[100.0_f32, 200.0, 400.0]).unwrap();
    let ld = exec.to_device(&[1000_u32, 2000, 4000]).unwrap();
    let le = exec.to_device(&[11.0_f32, 12.0, 14.0]).unwrap();
    let lf = exec.to_device(&[110_u32, 120, 140]).unwrap();
    let lg = exec.to_device(&[1100.0_f32, 1200.0, 1400.0]).unwrap();
    let lh = exec.to_device(&[11000_u32, 12000, 14000]).unwrap();
    let li = exec.to_device(&[21.0_f32, 22.0, 24.0]).unwrap();
    let lj = exec.to_device(&[210_u32, 220, 240]).unwrap();
    let lk = exec.to_device(&[2100.0_f32, 2200.0, 2400.0]).unwrap();
    let ll = exec.to_device(&[21000_u32, 22000, 24000]).unwrap();

    let ra = exec.to_device(&[2.0_f32, 3.0, 5.0]).unwrap();
    let rb = exec.to_device(&[20_u32, 30, 50]).unwrap();
    let rc = exec.to_device(&[200.0_f32, 300.0, 500.0]).unwrap();
    let rd = exec.to_device(&[2000_u32, 3000, 5000]).unwrap();
    let re = exec.to_device(&[12.0_f32, 13.0, 15.0]).unwrap();
    let rf = exec.to_device(&[120_u32, 130, 150]).unwrap();
    let rg = exec.to_device(&[1200.0_f32, 1300.0, 1500.0]).unwrap();
    let rh = exec.to_device(&[12000_u32, 13000, 15000]).unwrap();
    let ri = exec.to_device(&[22.0_f32, 23.0, 25.0]).unwrap();
    let rj = exec.to_device(&[220_u32, 230, 250]).unwrap();
    let rk = exec.to_device(&[2200.0_f32, 2300.0, 2500.0]).unwrap();
    let rl = exec.to_device(&[22000_u32, 23000, 25000]).unwrap();

    let output = set_union(
        &exec,
        zip12(
            la.slice(..),
            lb.slice(..),
            lc.slice(..),
            ld.slice(..),
            le.slice(..),
            lf.slice(..),
            lg.slice(..),
            lh.slice(..),
            li.slice(..),
            lj.slice(..),
            lk.slice(..),
            ll.slice(..),
        ),
        zip12(
            ra.slice(..),
            rb.slice(..),
            rc.slice(..),
            rd.slice(..),
            re.slice(..),
            rf.slice(..),
            rg.slice(..),
            rh.slice(..),
            ri.slice(..),
            rj.slice(..),
            rk.slice(..),
            rl.slice(..),
        ),
        Tuple12MixedLess,
    )
    .unwrap();
    let (a, b, c, d, e, f, g, h, i, j, k, l) = output;

    assert_eq!(exec.to_host(&a).unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![10, 20, 30, 40, 50]);
    assert_eq!(
        exec.to_host(&c).unwrap(),
        vec![100.0, 200.0, 300.0, 400.0, 500.0]
    );
    assert_eq!(
        exec.to_host(&d).unwrap(),
        vec![1000, 2000, 3000, 4000, 5000]
    );
    assert_eq!(
        exec.to_host(&e).unwrap(),
        vec![11.0, 12.0, 13.0, 14.0, 15.0]
    );
    assert_eq!(exec.to_host(&f).unwrap(), vec![110, 120, 130, 140, 150]);
    assert_eq!(
        exec.to_host(&g).unwrap(),
        vec![1100.0, 1200.0, 1300.0, 1400.0, 1500.0]
    );
    assert_eq!(
        exec.to_host(&h).unwrap(),
        vec![11000, 12000, 13000, 14000, 15000]
    );
    assert_eq!(
        exec.to_host(&i).unwrap(),
        vec![21.0, 22.0, 23.0, 24.0, 25.0]
    );
    assert_eq!(exec.to_host(&j).unwrap(), vec![210, 220, 230, 240, 250]);
    assert_eq!(
        exec.to_host(&k).unwrap(),
        vec![2100.0, 2200.0, 2300.0, 2400.0, 2500.0]
    );
    assert_eq!(
        exec.to_host(&l).unwrap(),
        vec![21000, 22000, 23000, 24000, 25000]
    );
}

#[cfg(any())]
#[test]
fn set_union_preserves_multiplicity_for_borrowed_heterogeneous_soa12() {
    let exec = exec();
    let (la, lb, lc, ld, le, lf, lg, lh, li, lj, lk, ll) = soa12_rows!(exec; [1, 2, 2, 4]);
    let (ra, rb, rc, rd, re, rf, rg, rh, ri, rj, rk, rl) = soa12_rows!(exec; [2, 2, 2, 3]);

    let output = set_union(
        &exec,
        zip12(
            la.slice(..),
            lb.slice(..),
            lc.slice(..),
            ld.slice(..),
            le.slice(..),
            lf.slice(..),
            lg.slice(..),
            lh.slice(..),
            li.slice(..),
            lj.slice(..),
            lk.slice(..),
            ll.slice(..),
        ),
        zip12(
            ra.slice(..),
            rb.slice(..),
            rc.slice(..),
            rd.slice(..),
            re.slice(..),
            rf.slice(..),
            rg.slice(..),
            rh.slice(..),
            ri.slice(..),
            rj.slice(..),
            rk.slice(..),
            rl.slice(..),
        ),
        Tuple12MixedLess,
    )
    .unwrap();

    assert_soa12_rows!(output; [1, 2, 2, 2, 3, 4]);
}
