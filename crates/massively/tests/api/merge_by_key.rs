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
fn merge_by_key_accepts_tuple_values() {
    let policy = policy();
    let left_keys = policy.to_device(&[0_u32, 2, 2, 5]).unwrap();
    let right_keys = policy.to_device(&[1_u32, 2, 4]).unwrap();
    let left_values = policy.to_device(&[0.0_f32, 20.0, 21.0, 50.0]).unwrap();
    let left_ids = policy.to_device(&[0_u32, 20, 21, 50]).unwrap();
    let right_values = policy.to_device(&[10.0_f32, 22.0, 40.0]).unwrap();
    let right_ids = policy.to_device(&[10_u32, 22, 40]).unwrap();

    let (keys, values) = merge_by_key(
        (&left_keys,),
        (&left_values, &left_ids),
        (&right_keys,),
        (&right_values, &right_ids),
        LessU32,
    )
    .unwrap();
    let (keys,) = keys;
    let (values, ids) = values;

    assert_eq!(keys.to_vec().unwrap(), vec![0, 1, 2, 2, 2, 4, 5]);
    assert_eq!(
        values.to_vec().unwrap(),
        vec![0.0, 10.0, 20.0, 21.0, 22.0, 40.0, 50.0]
    );
    assert_eq!(ids.to_vec().unwrap(), vec![0, 10, 20, 21, 22, 40, 50]);
}

#[cfg(any())]
#[test]
fn merge_by_key_accepts_borrowed_tuple_keys() {
    let policy = policy();
    let left_key_a = policy.to_device(&[1.0_f32, 2.0]).unwrap();
    let left_key_b = policy.to_device(&[10_u32, 20]).unwrap();
    let left_values = policy.to_device(&[100_u32, 200]).unwrap();
    let right_key_a = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let right_key_b = policy.to_device(&[20_u32, 10, 30]).unwrap();
    let right_values = policy.to_device(&[120_u32, 210, 300]).unwrap();

    let (keys, values) = merge_by_key(
        (&left_key_a, &left_key_b),
        (&left_values,),
        (&right_key_a, &right_key_b),
        &right_values,
        MixedTupleLess,
    )
    .unwrap();
    let (key_a, key_b) = keys;
    let (values,) = values;

    assert_eq!(key_a.to_vec().unwrap(), vec![1.0, 1.0, 2.0, 2.0, 3.0]);
    assert_eq!(key_b.to_vec().unwrap(), vec![10, 20, 10, 20, 30]);
    assert_eq!(values.to_vec().unwrap(), vec![100, 120, 210, 200, 300]);
}

#[cfg(any())]
#[test]
fn merge_by_tuple_key_reports_left_value_length_mismatch() {
    let policy = policy();
    let left_key_a = policy.to_device(&[1.0_f32, 2.0]).unwrap();
    let left_key_b = policy.to_device(&[10_u32, 20]).unwrap();
    let left_values = policy.to_device(&[100_u32]).unwrap();
    let right_key_a = policy.to_device(&[3.0_f32]).unwrap();
    let right_key_b = policy.to_device(&[30_u32]).unwrap();
    let right_values = policy.to_device(&[300_u32]).unwrap();

    let err = merge_by_key(
        (&left_key_a, &left_key_b),
        (&left_values,),
        (&right_key_a, &right_key_b),
        &right_values,
        MixedTupleLess,
    )
    .unwrap_err();

    assert!(matches!(err, massively::Error::LengthMismatch { .. }));
}

#[cfg(any())]
#[test]
fn merge_by_tuple_key_reports_right_value_length_mismatch() {
    let policy = policy();
    let left_key_a = policy.to_device(&[1.0_f32]).unwrap();
    let left_key_b = policy.to_device(&[10_u32]).unwrap();
    let left_values = policy.to_device(&[100_u32]).unwrap();
    let right_key_a = policy.to_device(&[2.0_f32, 3.0]).unwrap();
    let right_key_b = policy.to_device(&[20_u32, 30]).unwrap();
    let right_values = policy.to_device(&[200_u32]).unwrap();

    let err = merge_by_key(
        (&left_key_a, &left_key_b),
        (&left_values,),
        (&right_key_a, &right_key_b),
        &right_values,
        MixedTupleLess,
    )
    .unwrap_err();

    assert!(matches!(err, massively::Error::LengthMismatch { .. }));
}

#[cfg(any())]
#[test]
fn merge_by_tuple_key_reports_left_tuple_value_length_mismatch() {
    let policy = policy();
    let left_key_a = policy.to_device(&[1.0_f32, 2.0]).unwrap();
    let left_key_b = policy.to_device(&[10_u32, 20]).unwrap();
    let left_value_a = policy.to_device(&[100_u32]).unwrap();
    let left_value_b = policy.to_device(&[1000.0_f32]).unwrap();
    let right_key_a = policy.to_device(&[3.0_f32]).unwrap();
    let right_key_b = policy.to_device(&[30_u32]).unwrap();
    let right_value_a = policy.to_device(&[300_u32]).unwrap();
    let right_value_b = policy.to_device(&[3000.0_f32]).unwrap();

    let err = merge_by_key(
        (&left_key_a, &left_key_b),
        (&left_value_a, &left_value_b),
        (&right_key_a, &right_key_b),
        (&right_value_a, &right_value_b),
        MixedTupleLess,
    )
    .unwrap_err();

    assert!(matches!(err, massively::Error::LengthMismatch { .. }));
}

#[cfg(any())]
#[test]
fn merge_by_tuple_key_reports_right_tuple_value_length_mismatch() {
    let policy = policy();
    let left_key_a = policy.to_device(&[1.0_f32]).unwrap();
    let left_key_b = policy.to_device(&[10_u32]).unwrap();
    let left_value_a = policy.to_device(&[100_u32]).unwrap();
    let left_value_b = policy.to_device(&[1000.0_f32]).unwrap();
    let right_key_a = policy.to_device(&[2.0_f32, 3.0]).unwrap();
    let right_key_b = policy.to_device(&[20_u32, 30]).unwrap();
    let right_value_a = policy.to_device(&[200_u32]).unwrap();
    let right_value_b = policy.to_device(&[2000.0_f32]).unwrap();

    let err = merge_by_key(
        (&left_key_a, &left_key_b),
        (&left_value_a, &left_value_b),
        (&right_key_a, &right_key_b),
        (&right_value_a, &right_value_b),
        MixedTupleLess,
    )
    .unwrap_err();

    assert!(matches!(err, massively::Error::LengthMismatch { .. }));
}

#[cfg(any())]
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
        (&left_keys,),
        zip4(&left_a, &left_b, &left_c, &left_d),
        (&right_keys,),
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

#[cfg(any())]
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
        (&left_keys,),
        zip12(&la, &lb, &lc, &ld, &le, &lf, &lg, &lh, &li, &lj, &lk, &ll),
        (&right_keys,),
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

#[cfg(any())]
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
        (&left_keys,),
        zip12(&la, &lb, &lc, &ld, &le, &lf, &lg, &lh, &li, &lj, &lk, &ll),
        (&right_keys,),
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
