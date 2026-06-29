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
fn merge_by_key_accepts_tuple_values() {
    let exec = exec();
    let left_keys = exec.to_device(&[0_u32, 2, 2, 5]).unwrap();
    let right_keys = exec.to_device(&[1_u32, 2, 4]).unwrap();
    let left_values = exec.to_device(&[0.0_f32, 20.0, 21.0, 50.0]).unwrap();
    let left_ids = exec.to_device(&[0_u32, 20, 21, 50]).unwrap();
    let right_values = exec.to_device(&[10.0_f32, 22.0, 40.0]).unwrap();
    let right_ids = exec.to_device(&[10_u32, 22, 40]).unwrap();

    let (keys, values) = merge_by_key(
        &exec,
        massively::SoA1(left_keys.slice(..)),
        massively::SoA2(left_values.slice(..), left_ids.slice(..)),
        massively::SoA1(right_keys.slice(..)),
        massively::SoA2(right_values.slice(..), right_ids.slice(..)),
        LessU32,
    )
    .unwrap();
    let massively::SoA1(keys) = keys;
    let massively::SoA2(values, ids) = values;

    assert_eq!(exec.to_host(&keys).unwrap(), vec![0, 1, 2, 2, 2, 4, 5]);
    assert_eq!(
        exec.to_host(&values).unwrap(),
        vec![0.0, 10.0, 20.0, 21.0, 22.0, 40.0, 50.0]
    );
    assert_eq!(exec.to_host(&ids).unwrap(), vec![0, 10, 20, 21, 22, 40, 50]);
}

#[test]
fn merge_by_key_accepts_three_column_keys() {
    let exec = exec();
    let lk0 = exec.to_device(&[0.0_f32, 1.0, 1.0]).unwrap();
    let lk1 = exec.to_device(&[9_u32, 1, 2]).unwrap();
    let lk2 = exec.to_device(&[0.0_f32, 3.0, 0.0]).unwrap();
    let rk0 = exec.to_device(&[1.0_f32, 2.0]).unwrap();
    let rk1 = exec.to_device(&[1_u32, 0]).unwrap();
    let rk2 = exec.to_device(&[9.0_f32, 0.0]).unwrap();
    let left_values = exec.to_device(&[90_u32, 13, 20]).unwrap();
    let right_values = exec.to_device(&[19_u32, 200]).unwrap();

    let (massively::SoA3(out_k0, out_k1, out_k2), massively::SoA1(out_values)) = merge_by_key(
        &exec,
        massively::SoA3(lk0.slice(..), lk1.slice(..), lk2.slice(..)),
        massively::SoA1(left_values.slice(..)),
        massively::SoA3(rk0.slice(..), rk1.slice(..), rk2.slice(..)),
        massively::SoA1(right_values.slice(..)),
        MixedTuple3LexLess,
    )
    .unwrap();

    assert_eq!(
        exec.to_host(&out_k0).unwrap(),
        vec![0.0, 1.0, 1.0, 1.0, 2.0]
    );
    assert_eq!(exec.to_host(&out_k1).unwrap(), vec![9, 1, 1, 2, 0]);
    assert_eq!(
        exec.to_host(&out_k2).unwrap(),
        vec![0.0, 3.0, 9.0, 0.0, 0.0]
    );
    assert_eq!(
        exec.to_host(&out_values).unwrap(),
        vec![90, 13, 19, 20, 200]
    );
}

#[test]
fn merge_by_key_accepts_three_column_keys_and_tuple_values() {
    let exec = exec();
    let lk0 = exec.to_device(&[0.0_f32, 1.0, 1.0]).unwrap();
    let lk1 = exec.to_device(&[9_u32, 1, 2]).unwrap();
    let lk2 = exec.to_device(&[0.0_f32, 3.0, 0.0]).unwrap();
    let rk0 = exec.to_device(&[1.0_f32, 2.0]).unwrap();
    let rk1 = exec.to_device(&[1_u32, 0]).unwrap();
    let rk2 = exec.to_device(&[9.0_f32, 0.0]).unwrap();
    let la = exec.to_device(&[90.0_f32, 13.0, 20.0]).unwrap();
    let lb = exec.to_device(&[900_u32, 130, 200]).unwrap();
    let lc = exec.to_device(&[9000.0_f32, 1300.0, 2000.0]).unwrap();
    let ra = exec.to_device(&[19.0_f32, 200.0]).unwrap();
    let rb = exec.to_device(&[190_u32, 2000]).unwrap();
    let rc = exec.to_device(&[1900.0_f32, 20000.0]).unwrap();

    let (massively::SoA3(out_k0, out_k1, out_k2), massively::SoA3(out_a, out_b, out_c)) =
        merge_by_key(
            &exec,
            massively::SoA3(lk0.slice(..), lk1.slice(..), lk2.slice(..)),
            massively::SoA3(la.slice(..), lb.slice(..), lc.slice(..)),
            massively::SoA3(rk0.slice(..), rk1.slice(..), rk2.slice(..)),
            massively::SoA3(ra.slice(..), rb.slice(..), rc.slice(..)),
            MixedTuple3LexLess,
        )
        .unwrap();

    assert_eq!(
        exec.to_host(&out_k0).unwrap(),
        vec![0.0, 1.0, 1.0, 1.0, 2.0]
    );
    assert_eq!(exec.to_host(&out_k1).unwrap(), vec![9, 1, 1, 2, 0]);
    assert_eq!(
        exec.to_host(&out_k2).unwrap(),
        vec![0.0, 3.0, 9.0, 0.0, 0.0]
    );
    assert_eq!(
        exec.to_host(&out_a).unwrap(),
        vec![90.0, 13.0, 19.0, 20.0, 200.0]
    );
    assert_eq!(
        exec.to_host(&out_b).unwrap(),
        vec![900, 130, 190, 200, 2000]
    );
    assert_eq!(
        exec.to_host(&out_c).unwrap(),
        vec![9000.0, 1300.0, 1900.0, 2000.0, 20000.0]
    );
}

#[test]
fn merge_by_key_accepts_three_column_keys_and_seven_column_values() {
    let exec = exec();
    let lk0 = exec.to_device(&[0.0_f32, 1.0, 1.0]).unwrap();
    let lk1 = exec.to_device(&[9_u32, 1, 2]).unwrap();
    let lk2 = exec.to_device(&[0.0_f32, 3.0, 0.0]).unwrap();
    let rk0 = exec.to_device(&[1.0_f32, 2.0]).unwrap();
    let rk1 = exec.to_device(&[1_u32, 0]).unwrap();
    let rk2 = exec.to_device(&[9.0_f32, 0.0]).unwrap();
    let la = exec.to_device(&[90.0_f32, 13.0, 20.0]).unwrap();
    let lb = exec.to_device(&[900_u32, 130, 200]).unwrap();
    let lc = exec.to_device(&[9000.0_f32, 1300.0, 2000.0]).unwrap();
    let ld = exec.to_device(&[91_u32, 14, 21]).unwrap();
    let le = exec.to_device(&[9.1_f32, 1.4, 2.1]).unwrap();
    let lf = exec.to_device(&[910_u32, 140, 210]).unwrap();
    let lg = exec.to_device(&[91.0_f32, 14.0, 21.0]).unwrap();
    let ra = exec.to_device(&[19.0_f32, 200.0]).unwrap();
    let rb = exec.to_device(&[190_u32, 2000]).unwrap();
    let rc = exec.to_device(&[1900.0_f32, 20000.0]).unwrap();
    let rd = exec.to_device(&[20_u32, 201]).unwrap();
    let re = exec.to_device(&[2.0_f32, 20.1]).unwrap();
    let rf = exec.to_device(&[200_u32, 2010]).unwrap();
    let rg = exec.to_device(&[20.0_f32, 201.0]).unwrap();

    let (
        massively::SoA3(out_k0, out_k1, out_k2),
        massively::SoA7(out_a, out_b, out_c, out_d, out_e, out_f, out_g),
    ) = merge_by_key(
        &exec,
        massively::SoA3(lk0.slice(..), lk1.slice(..), lk2.slice(..)),
        massively::SoA7(
            la.slice(..),
            lb.slice(..),
            lc.slice(..),
            ld.slice(..),
            le.slice(..),
            lf.slice(..),
            lg.slice(..),
        ),
        massively::SoA3(rk0.slice(..), rk1.slice(..), rk2.slice(..)),
        massively::SoA7(
            ra.slice(..),
            rb.slice(..),
            rc.slice(..),
            rd.slice(..),
            re.slice(..),
            rf.slice(..),
            rg.slice(..),
        ),
        MixedTuple3LexLess,
    )
    .unwrap();

    assert_eq!(
        exec.to_host(&out_k0).unwrap(),
        vec![0.0, 1.0, 1.0, 1.0, 2.0]
    );
    assert_eq!(exec.to_host(&out_k1).unwrap(), vec![9, 1, 1, 2, 0]);
    assert_eq!(
        exec.to_host(&out_k2).unwrap(),
        vec![0.0, 3.0, 9.0, 0.0, 0.0]
    );
    assert_eq!(
        exec.to_host(&out_a).unwrap(),
        vec![90.0, 13.0, 19.0, 20.0, 200.0]
    );
    assert_eq!(
        exec.to_host(&out_b).unwrap(),
        vec![900, 130, 190, 200, 2000]
    );
    assert_eq!(
        exec.to_host(&out_c).unwrap(),
        vec![9000.0, 1300.0, 1900.0, 2000.0, 20000.0]
    );
    assert_eq!(exec.to_host(&out_d).unwrap(), vec![91, 14, 20, 21, 201]);
    assert_eq!(
        exec.to_host(&out_e).unwrap(),
        vec![9.1, 1.4, 2.0, 2.1, 20.1]
    );
    assert_eq!(
        exec.to_host(&out_f).unwrap(),
        vec![910, 140, 200, 210, 2010]
    );
    assert_eq!(
        exec.to_host(&out_g).unwrap(),
        vec![91.0, 14.0, 20.0, 21.0, 201.0]
    );
}
