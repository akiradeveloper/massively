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
        (left_keys.slice(..),),
        (left_values.slice(..), left_ids.slice(..)),
        (right_keys.slice(..),),
        (right_values.slice(..), right_ids.slice(..)),
        LessU32,
    )
    .unwrap();
    let (keys,) = keys;
    let (values, ids) = values;

    assert_eq!(exec.to_host(&keys).unwrap(), vec![0, 1, 2, 2, 2, 4, 5]);
    assert_eq!(
        exec.to_host(&values).unwrap(),
        vec![0.0, 10.0, 20.0, 21.0, 22.0, 40.0, 50.0]
    );
    assert_eq!(exec.to_host(&ids).unwrap(), vec![0, 10, 20, 21, 22, 40, 50]);
}

#[cfg(any())]
#[test]
fn merge_by_key_accepts_borrowed_tuple_keys() {
    let exec = exec();
    let left_key_a = exec.to_device(&[1.0_f32, 2.0]).unwrap();
    let left_key_b = exec.to_device(&[10_u32, 20]).unwrap();
    let left_values = exec.to_device(&[100_u32, 200]).unwrap();
    let right_key_a = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let right_key_b = exec.to_device(&[20_u32, 10, 30]).unwrap();
    let right_values = exec.to_device(&[120_u32, 210, 300]).unwrap();

    let (keys, values) = merge_by_key(
        &exec,
        (left_key_a.slice(..), left_key_b.slice(..)),
        (left_values.slice(..),),
        (right_key_a.slice(..), right_key_b.slice(..)),
        (right_values.slice(..),),
        MixedTupleLess,
    )
    .unwrap();
    let (key_a, key_b) = keys;
    let (values,) = values;

    assert_eq!(exec.to_host(&key_a).unwrap(), vec![1.0, 1.0, 2.0, 2.0, 3.0]);
    assert_eq!(exec.to_host(&key_b).unwrap(), vec![10, 20, 10, 20, 30]);
    assert_eq!(
        exec.to_host(&values).unwrap(),
        vec![100, 120, 210, 200, 300]
    );
}

#[cfg(any())]
#[test]
fn merge_by_tuple_key_reports_left_value_length_mismatch() {
    let exec = exec();
    let left_key_a = exec.to_device(&[1.0_f32, 2.0]).unwrap();
    let left_key_b = exec.to_device(&[10_u32, 20]).unwrap();
    let left_values = exec.to_device(&[100_u32]).unwrap();
    let right_key_a = exec.to_device(&[3.0_f32]).unwrap();
    let right_key_b = exec.to_device(&[30_u32]).unwrap();
    let right_values = exec.to_device(&[300_u32]).unwrap();

    let err = merge_by_key(
        &exec,
        (left_key_a.slice(..), left_key_b.slice(..)),
        (left_values.slice(..),),
        (right_key_a.slice(..), right_key_b.slice(..)),
        (right_values.slice(..),),
        MixedTupleLess,
    )
    .unwrap_err();

    assert!(matches!(err, massively::Error::LengthMismatch { .. }));
}

#[cfg(any())]
#[test]
fn merge_by_tuple_key_reports_right_value_length_mismatch() {
    let exec = exec();
    let left_key_a = exec.to_device(&[1.0_f32]).unwrap();
    let left_key_b = exec.to_device(&[10_u32]).unwrap();
    let left_values = exec.to_device(&[100_u32]).unwrap();
    let right_key_a = exec.to_device(&[2.0_f32, 3.0]).unwrap();
    let right_key_b = exec.to_device(&[20_u32, 30]).unwrap();
    let right_values = exec.to_device(&[200_u32]).unwrap();

    let err = merge_by_key(
        &exec,
        (left_key_a.slice(..), left_key_b.slice(..)),
        (left_values.slice(..),),
        (right_key_a.slice(..), right_key_b.slice(..)),
        (right_values.slice(..),),
        MixedTupleLess,
    )
    .unwrap_err();

    assert!(matches!(err, massively::Error::LengthMismatch { .. }));
}

#[cfg(any())]
#[test]
fn merge_by_tuple_key_reports_left_tuple_value_length_mismatch() {
    let exec = exec();
    let left_key_a = exec.to_device(&[1.0_f32, 2.0]).unwrap();
    let left_key_b = exec.to_device(&[10_u32, 20]).unwrap();
    let left_value_a = exec.to_device(&[100_u32]).unwrap();
    let left_value_b = exec.to_device(&[1000.0_f32]).unwrap();
    let right_key_a = exec.to_device(&[3.0_f32]).unwrap();
    let right_key_b = exec.to_device(&[30_u32]).unwrap();
    let right_value_a = exec.to_device(&[300_u32]).unwrap();
    let right_value_b = exec.to_device(&[3000.0_f32]).unwrap();

    let err = merge_by_key(
        &exec,
        (left_key_a.slice(..), left_key_b.slice(..)),
        (left_value_a.slice(..), left_value_b.slice(..)),
        (right_key_a.slice(..), right_key_b.slice(..)),
        (right_value_a.slice(..), right_value_b.slice(..)),
        MixedTupleLess,
    )
    .unwrap_err();

    assert!(matches!(err, massively::Error::LengthMismatch { .. }));
}

#[cfg(any())]
#[test]
fn merge_by_tuple_key_reports_right_tuple_value_length_mismatch() {
    let exec = exec();
    let left_key_a = exec.to_device(&[1.0_f32]).unwrap();
    let left_key_b = exec.to_device(&[10_u32]).unwrap();
    let left_value_a = exec.to_device(&[100_u32]).unwrap();
    let left_value_b = exec.to_device(&[1000.0_f32]).unwrap();
    let right_key_a = exec.to_device(&[2.0_f32, 3.0]).unwrap();
    let right_key_b = exec.to_device(&[20_u32, 30]).unwrap();
    let right_value_a = exec.to_device(&[200_u32]).unwrap();
    let right_value_b = exec.to_device(&[2000.0_f32]).unwrap();

    let err = merge_by_key(
        &exec,
        (left_key_a.slice(..), left_key_b.slice(..)),
        (left_value_a.slice(..), left_value_b.slice(..)),
        (right_key_a.slice(..), right_key_b.slice(..)),
        (right_value_a.slice(..), right_value_b.slice(..)),
        MixedTupleLess,
    )
    .unwrap_err();

    assert!(matches!(err, massively::Error::LengthMismatch { .. }));
}

#[cfg(any())]
#[test]
fn merge_by_key_accepts_wide_soa_values() {
    let exec = exec();
    let left_keys = exec.to_device(&[0_u32, 2]).unwrap();
    let right_keys = exec.to_device(&[1_u32, 3]).unwrap();
    let left_a = exec.to_device(&[0.0_f32, 20.0]).unwrap();
    let left_b = exec.to_device(&[0_u32, 200]).unwrap();
    let left_c = exec.to_device(&[0.0_f32, 2000.0]).unwrap();
    let left_d = exec.to_device(&[0_u32, 20000]).unwrap();
    let right_a = exec.to_device(&[10.0_f32, 30.0]).unwrap();
    let right_b = exec.to_device(&[100_u32, 300]).unwrap();
    let right_c = exec.to_device(&[1000.0_f32, 3000.0]).unwrap();
    let right_d = exec.to_device(&[10000_u32, 30000]).unwrap();

    let (keys, values) = merge_by_key(
        &exec,
        (left_keys.slice(..),),
        zip4(
            left_a.slice(..),
            left_b.slice(..),
            left_c.slice(..),
            left_d.slice(..),
        ),
        (right_keys.slice(..),),
        zip4(
            right_a.slice(..),
            right_b.slice(..),
            right_c.slice(..),
            right_d.slice(..),
        ),
        LessU32,
    )
    .unwrap();
    let keys = keys;
    let (a, b, c, d) = values;

    assert_eq!(exec.to_host(&keys).unwrap(), vec![0, 1, 2, 3]);
    assert_eq!(exec.to_host(&a).unwrap(), vec![0.0, 10.0, 20.0, 30.0]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![0, 100, 200, 300]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![0.0, 1000.0, 2000.0, 3000.0]);
    assert_eq!(exec.to_host(&d).unwrap(), vec![0, 10000, 20000, 30000]);
}

#[cfg(any())]
#[test]
fn merge_by_key_accepts_soa12_values() {
    let exec = exec();
    let left_keys = exec.to_device(&[0_u32, 2]).unwrap();
    let right_keys = exec.to_device(&[1_u32, 3]).unwrap();
    let la = exec.to_device(&[0.0_f32, 20.0]).unwrap();
    let lb = exec.to_device(&[0_u32, 20]).unwrap();
    let lc = exec.to_device(&[0.0_f32, 200.0]).unwrap();
    let ld = exec.to_device(&[0_u32, 200]).unwrap();
    let le = exec.to_device(&[0.0_f32, 2000.0]).unwrap();
    let lf = exec.to_device(&[0_u32, 2000]).unwrap();
    let lg = exec.to_device(&[4.0_f32, 6.0]).unwrap();
    let lh = exec.to_device(&[40_u32, 60]).unwrap();
    let li = exec.to_device(&[400.0_f32, 600.0]).unwrap();
    let lj = exec.to_device(&[4000_u32, 6000]).unwrap();
    let lk = exec.to_device(&[7.0_f32, 9.0]).unwrap();
    let ll = exec.to_device(&[70_u32, 90]).unwrap();
    let ra = exec.to_device(&[10.0_f32, 30.0]).unwrap();
    let rb = exec.to_device(&[10_u32, 30]).unwrap();
    let rc = exec.to_device(&[100.0_f32, 300.0]).unwrap();
    let rd = exec.to_device(&[100_u32, 300]).unwrap();
    let re = exec.to_device(&[1000.0_f32, 3000.0]).unwrap();
    let rf = exec.to_device(&[1000_u32, 3000]).unwrap();
    let rg = exec.to_device(&[5.0_f32, 7.0]).unwrap();
    let rh = exec.to_device(&[50_u32, 70]).unwrap();
    let ri = exec.to_device(&[500.0_f32, 700.0]).unwrap();
    let rj = exec.to_device(&[5000_u32, 7000]).unwrap();
    let rk = exec.to_device(&[8.0_f32, 10.0]).unwrap();
    let rl = exec.to_device(&[80_u32, 100]).unwrap();

    let (keys, values) = merge_by_key(
        &exec,
        (left_keys.slice(..),),
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
        (right_keys.slice(..),),
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
        LessU32,
    )
    .unwrap();
    let keys = keys;
    let (a, b, c, d, e, f, g, h, i, j, k, l) = values;

    assert_eq!(exec.to_host(&keys).unwrap(), vec![0, 1, 2, 3]);
    assert_eq!(exec.to_host(&a).unwrap(), vec![0.0, 10.0, 20.0, 30.0]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![0, 10, 20, 30]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![0.0, 100.0, 200.0, 300.0]);
    assert_eq!(exec.to_host(&d).unwrap(), vec![0, 100, 200, 300]);
    assert_eq!(exec.to_host(&e).unwrap(), vec![0.0, 1000.0, 2000.0, 3000.0]);
    assert_eq!(exec.to_host(&f).unwrap(), vec![0, 1000, 2000, 3000]);
    assert_eq!(exec.to_host(&g).unwrap(), vec![4.0, 5.0, 6.0, 7.0]);
    assert_eq!(exec.to_host(&h).unwrap(), vec![40, 50, 60, 70]);
    assert_eq!(exec.to_host(&i).unwrap(), vec![400.0, 500.0, 600.0, 700.0]);
    assert_eq!(exec.to_host(&j).unwrap(), vec![4000, 5000, 6000, 7000]);
    assert_eq!(exec.to_host(&k).unwrap(), vec![7.0, 8.0, 9.0, 10.0]);
    assert_eq!(exec.to_host(&l).unwrap(), vec![70, 80, 90, 100]);
}

#[cfg(any())]
#[test]
fn merge_by_key_accepts_soa12_values_with_equal_keys_and_uneven_lengths() {
    let exec = exec();
    let left_keys = exec.to_device(&[0_u32, 2, 2, 5]).unwrap();
    let right_keys = exec.to_device(&[1_u32, 2, 4]).unwrap();
    let la = exec.to_device(&[0.0_f32, 20.0, 21.0, 50.0]).unwrap();
    let lb = exec.to_device(&[0_u32, 20, 21, 50]).unwrap();
    let lc = exec.to_device(&[100.0_f32, 120.0, 121.0, 150.0]).unwrap();
    let ld = exec.to_device(&[100_u32, 120, 121, 150]).unwrap();
    let le = exec.to_device(&[200.0_f32, 220.0, 221.0, 250.0]).unwrap();
    let lf = exec.to_device(&[200_u32, 220, 221, 250]).unwrap();
    let lg = exec.to_device(&[300.0_f32, 320.0, 321.0, 350.0]).unwrap();
    let lh = exec.to_device(&[300_u32, 320, 321, 350]).unwrap();
    let li = exec.to_device(&[400.0_f32, 420.0, 421.0, 450.0]).unwrap();
    let lj = exec.to_device(&[400_u32, 420, 421, 450]).unwrap();
    let lk = exec.to_device(&[500.0_f32, 520.0, 521.0, 550.0]).unwrap();
    let ll = exec.to_device(&[500_u32, 520, 521, 550]).unwrap();
    let ra = exec.to_device(&[10.0_f32, 22.0, 40.0]).unwrap();
    let rb = exec.to_device(&[10_u32, 22, 40]).unwrap();
    let rc = exec.to_device(&[110.0_f32, 122.0, 140.0]).unwrap();
    let rd = exec.to_device(&[110_u32, 122, 140]).unwrap();
    let re = exec.to_device(&[210.0_f32, 222.0, 240.0]).unwrap();
    let rf = exec.to_device(&[210_u32, 222, 240]).unwrap();
    let rg = exec.to_device(&[310.0_f32, 322.0, 340.0]).unwrap();
    let rh = exec.to_device(&[310_u32, 322, 340]).unwrap();
    let ri = exec.to_device(&[410.0_f32, 422.0, 440.0]).unwrap();
    let rj = exec.to_device(&[410_u32, 422, 440]).unwrap();
    let rk = exec.to_device(&[510.0_f32, 522.0, 540.0]).unwrap();
    let rl = exec.to_device(&[510_u32, 522, 540]).unwrap();

    let (keys, values) = merge_by_key(
        &exec,
        (left_keys.slice(..),),
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
        (right_keys.slice(..),),
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
        LessU32,
    )
    .unwrap();
    let keys = keys;
    let (a, b, c, d, e, f, g, h, i, j, k, l) = values;

    assert_eq!(exec.to_host(&keys).unwrap(), vec![0, 1, 2, 2, 2, 4, 5]);
    assert_eq!(
        exec.to_host(&a).unwrap(),
        vec![0.0, 10.0, 20.0, 21.0, 22.0, 40.0, 50.0]
    );
    assert_eq!(exec.to_host(&b).unwrap(), vec![0, 10, 20, 21, 22, 40, 50]);
    assert_eq!(
        exec.to_host(&c).unwrap(),
        vec![100.0, 110.0, 120.0, 121.0, 122.0, 140.0, 150.0]
    );
    assert_eq!(
        exec.to_host(&d).unwrap(),
        vec![100, 110, 120, 121, 122, 140, 150]
    );
    assert_eq!(
        exec.to_host(&e).unwrap(),
        vec![200.0, 210.0, 220.0, 221.0, 222.0, 240.0, 250.0]
    );
    assert_eq!(
        exec.to_host(&f).unwrap(),
        vec![200, 210, 220, 221, 222, 240, 250]
    );
    assert_eq!(
        exec.to_host(&g).unwrap(),
        vec![300.0, 310.0, 320.0, 321.0, 322.0, 340.0, 350.0]
    );
    assert_eq!(
        exec.to_host(&h).unwrap(),
        vec![300, 310, 320, 321, 322, 340, 350]
    );
    assert_eq!(
        exec.to_host(&i).unwrap(),
        vec![400.0, 410.0, 420.0, 421.0, 422.0, 440.0, 450.0]
    );
    assert_eq!(
        exec.to_host(&j).unwrap(),
        vec![400, 410, 420, 421, 422, 440, 450]
    );
    assert_eq!(
        exec.to_host(&k).unwrap(),
        vec![500.0, 510.0, 520.0, 521.0, 522.0, 540.0, 550.0]
    );
    assert_eq!(
        exec.to_host(&l).unwrap(),
        vec![500, 510, 520, 521, 522, 540, 550]
    );
}
