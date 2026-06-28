use crate::common::*;

#[test]
fn sort_by_key_accepts_tuple_values() {
    let exec = exec();
    let keys = exec.to_device(&[3_u32, 1, 2]).unwrap();
    let a = exec.to_device(&[30_u32, 10, 20]).unwrap();
    let b = exec.to_device(&[30.0_f32, 10.0, 20.0]).unwrap();
    let c = exec.to_device(&[300_u32, 100, 200]).unwrap();

    let (keys, values) = sort_by_key(
        &exec,
        massively::SoA1(keys.slice(..)),
        massively::SoA3(a.slice(..), b.slice(..), c.slice(..)),
        LessU32,
    )
    .unwrap();
    let (keys,) = keys;
    let (a, b, c) = values;
    assert_eq!(exec.to_host(&keys).unwrap(), vec![1, 2, 3]);
    assert_eq!(exec.to_host(&a).unwrap(), vec![10, 20, 30]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![10.0, 20.0, 30.0]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![100, 200, 300]);
}

#[test]
fn sort_by_key_accepts_seven_column_values() {
    let exec = exec();
    let keys = exec.to_device(&[3_u32, 1, 2]).unwrap();
    let a = exec.to_device(&[30.0_f32, 10.0, 20.0]).unwrap();
    let b = exec.to_device(&[30_u32, 10, 20]).unwrap();
    let c = exec.to_device(&[300.0_f32, 100.0, 200.0]).unwrap();
    let d = exec.to_device(&[300_u32, 100, 200]).unwrap();
    let e = exec.to_device(&[31.0_f32, 11.0, 21.0]).unwrap();
    let f = exec.to_device(&[31_u32, 11, 21]).unwrap();
    let g = exec.to_device(&[301.0_f32, 101.0, 201.0]).unwrap();

    let (keys, values) = sort_by_key(
        &exec,
        massively::SoA1(keys.slice(..)),
        massively::SoA7(
            a.slice(..),
            b.slice(..),
            c.slice(..),
            d.slice(..),
            e.slice(..),
            f.slice(..),
            g.slice(..),
        ),
        LessU32,
    )
    .unwrap();
    let (keys,) = keys;
    let (a, b, c, d, e, f, g) = values;
    assert_eq!(exec.to_host(&keys).unwrap(), vec![1, 2, 3]);
    assert_eq!(exec.to_host(&a).unwrap(), vec![10.0, 20.0, 30.0]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![10, 20, 30]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![100.0, 200.0, 300.0]);
    assert_eq!(exec.to_host(&d).unwrap(), vec![100, 200, 300]);
    assert_eq!(exec.to_host(&e).unwrap(), vec![11.0, 21.0, 31.0]);
    assert_eq!(exec.to_host(&f).unwrap(), vec![11, 21, 31]);
    assert_eq!(exec.to_host(&g).unwrap(), vec![101.0, 201.0, 301.0]);
}

#[test]
fn sort_by_key_accepts_three_column_keys() {
    let exec = exec();
    let k0 = exec.to_device(&[1.0_f32, 1.0, 0.0, 1.0]).unwrap();
    let k1 = exec.to_device(&[2_u32, 1, 9, 1]).unwrap();
    let k2 = exec.to_device(&[0.0_f32, 9.0, 0.0, 3.0]).unwrap();
    let values = exec.to_device(&[20_u32, 19, 90, 13]).unwrap();

    let ((out_k0, out_k1, out_k2), (out_values,)) = sort_by_key(
        &exec,
        massively::SoA3(k0.slice(..), k1.slice(..), k2.slice(..)),
        massively::SoA1(values.slice(..)),
        MixedTuple3LexLess,
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_k0).unwrap(), vec![0.0, 1.0, 1.0, 1.0]);
    assert_eq!(exec.to_host(&out_k1).unwrap(), vec![9, 1, 1, 2]);
    assert_eq!(exec.to_host(&out_k2).unwrap(), vec![0.0, 3.0, 9.0, 0.0]);
    assert_eq!(exec.to_host(&out_values).unwrap(), vec![90, 13, 19, 20]);
}

#[test]
fn sort_by_key_accepts_three_column_keys_and_tuple_values() {
    let exec = exec();
    let k0 = exec.to_device(&[1.0_f32, 1.0, 0.0, 1.0]).unwrap();
    let k1 = exec.to_device(&[2_u32, 1, 9, 1]).unwrap();
    let k2 = exec.to_device(&[0.0_f32, 9.0, 0.0, 3.0]).unwrap();
    let a = exec.to_device(&[20.0_f32, 19.0, 90.0, 13.0]).unwrap();
    let b = exec.to_device(&[200_u32, 190, 900, 130]).unwrap();
    let c = exec
        .to_device(&[2000.0_f32, 1900.0, 9000.0, 1300.0])
        .unwrap();

    let ((out_k0, out_k1, out_k2), (out_a, out_b, out_c)) = sort_by_key(
        &exec,
        massively::SoA3(k0.slice(..), k1.slice(..), k2.slice(..)),
        massively::SoA3(a.slice(..), b.slice(..), c.slice(..)),
        MixedTuple3LexLess,
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_k0).unwrap(), vec![0.0, 1.0, 1.0, 1.0]);
    assert_eq!(exec.to_host(&out_k1).unwrap(), vec![9, 1, 1, 2]);
    assert_eq!(exec.to_host(&out_k2).unwrap(), vec![0.0, 3.0, 9.0, 0.0]);
    assert_eq!(exec.to_host(&out_a).unwrap(), vec![90.0, 13.0, 19.0, 20.0]);
    assert_eq!(exec.to_host(&out_b).unwrap(), vec![900, 130, 190, 200]);
    assert_eq!(
        exec.to_host(&out_c).unwrap(),
        vec![9000.0, 1300.0, 1900.0, 2000.0]
    );
}

#[test]
fn sort_by_key_accepts_three_column_keys_and_seven_column_values() {
    let exec = exec();
    let k0 = exec.to_device(&[1.0_f32, 1.0, 0.0, 1.0]).unwrap();
    let k1 = exec.to_device(&[2_u32, 1, 9, 1]).unwrap();
    let k2 = exec.to_device(&[0.0_f32, 9.0, 0.0, 3.0]).unwrap();
    let a = exec.to_device(&[20.0_f32, 19.0, 90.0, 13.0]).unwrap();
    let b = exec.to_device(&[200_u32, 190, 900, 130]).unwrap();
    let c = exec
        .to_device(&[2000.0_f32, 1900.0, 9000.0, 1300.0])
        .unwrap();
    let d = exec.to_device(&[21_u32, 20, 91, 14]).unwrap();
    let e = exec.to_device(&[2.1_f32, 2.0, 9.1, 1.4]).unwrap();
    let f = exec.to_device(&[210_u32, 200, 910, 140]).unwrap();
    let g = exec.to_device(&[21.0_f32, 20.0, 91.0, 14.0]).unwrap();

    let ((out_k0, out_k1, out_k2), (out_a, out_b, out_c, out_d, out_e, out_f, out_g)) =
        sort_by_key(
            &exec,
            massively::SoA3(k0.slice(..), k1.slice(..), k2.slice(..)),
            massively::SoA7(
                a.slice(..),
                b.slice(..),
                c.slice(..),
                d.slice(..),
                e.slice(..),
                f.slice(..),
                g.slice(..),
            ),
            MixedTuple3LexLess,
        )
        .unwrap();

    assert_eq!(exec.to_host(&out_k0).unwrap(), vec![0.0, 1.0, 1.0, 1.0]);
    assert_eq!(exec.to_host(&out_k1).unwrap(), vec![9, 1, 1, 2]);
    assert_eq!(exec.to_host(&out_k2).unwrap(), vec![0.0, 3.0, 9.0, 0.0]);
    assert_eq!(exec.to_host(&out_a).unwrap(), vec![90.0, 13.0, 19.0, 20.0]);
    assert_eq!(exec.to_host(&out_b).unwrap(), vec![900, 130, 190, 200]);
    assert_eq!(
        exec.to_host(&out_c).unwrap(),
        vec![9000.0, 1300.0, 1900.0, 2000.0]
    );
    assert_eq!(exec.to_host(&out_d).unwrap(), vec![91, 14, 20, 21]);
    assert_eq!(exec.to_host(&out_e).unwrap(), vec![9.1, 1.4, 2.0, 2.1]);
    assert_eq!(exec.to_host(&out_f).unwrap(), vec![910, 140, 200, 210]);
    assert_eq!(exec.to_host(&out_g).unwrap(), vec![91.0, 14.0, 20.0, 21.0]);
}

#[test]
fn stable_sort_by_key_accepts_three_column_keys_and_seven_values() {
    let exec = exec();
    let k0 = exec.to_device(&[1.0_f32, 0.0, 1.0, 1.0, 1.0]).unwrap();
    let k1 = exec.to_device(&[1_u32, 9, 1, 1, 2]).unwrap();
    let k2 = exec.to_device(&[0.0_f32, 0.0, 0.0, 0.0, 0.0]).unwrap();
    let a = exec.to_device(&[10.0_f32, 20.0, 30.0, 40.0, 50.0]).unwrap();
    let b = exec.to_device(&[100_u32, 200, 300, 400, 500]).unwrap();
    let c = exec
        .to_device(&[1000.0_f32, 2000.0, 3000.0, 4000.0, 5000.0])
        .unwrap();
    let d = exec.to_device(&[11_u32, 21, 31, 41, 51]).unwrap();
    let e = exec.to_device(&[1.1_f32, 2.1, 3.1, 4.1, 5.1]).unwrap();
    let f = exec.to_device(&[110_u32, 210, 310, 410, 510]).unwrap();
    let g = exec.to_device(&[11.0_f32, 21.0, 31.0, 41.0, 51.0]).unwrap();

    let ((out_k0, out_k1, out_k2), (out_a, out_b, out_c, out_d, out_e, out_f, out_g)) =
        massively::stable_sort_by_key(
            &exec,
            massively::SoA3(k0.slice(..), k1.slice(..), k2.slice(..)),
            massively::SoA7(
                a.slice(..),
                b.slice(..),
                c.slice(..),
                d.slice(..),
                e.slice(..),
                f.slice(..),
                g.slice(..),
            ),
            MixedTuple3LexLess,
        )
        .unwrap();

    assert_eq!(
        exec.to_host(&out_k0).unwrap(),
        vec![0.0, 1.0, 1.0, 1.0, 1.0]
    );
    assert_eq!(exec.to_host(&out_k1).unwrap(), vec![9, 1, 1, 1, 2]);
    assert_eq!(
        exec.to_host(&out_k2).unwrap(),
        vec![0.0, 0.0, 0.0, 0.0, 0.0]
    );
    assert_eq!(
        exec.to_host(&out_a).unwrap(),
        vec![20.0, 10.0, 30.0, 40.0, 50.0]
    );
    assert_eq!(exec.to_host(&out_b).unwrap(), vec![200, 100, 300, 400, 500]);
    assert_eq!(
        exec.to_host(&out_c).unwrap(),
        vec![2000.0, 1000.0, 3000.0, 4000.0, 5000.0]
    );
    assert_eq!(exec.to_host(&out_d).unwrap(), vec![21, 11, 31, 41, 51]);
    assert_eq!(exec.to_host(&out_e).unwrap(), vec![2.1, 1.1, 3.1, 4.1, 5.1]);
    assert_eq!(exec.to_host(&out_f).unwrap(), vec![210, 110, 310, 410, 510]);
    assert_eq!(
        exec.to_host(&out_g).unwrap(),
        vec![21.0, 11.0, 31.0, 41.0, 51.0]
    );
}
