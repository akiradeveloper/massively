use crate::common::*;

#[test]
fn sort_by_key_accepts_tuple_values() {
    let exec = exec();
    let keys = exec.to_device(&[3_u32, 1, 2]).unwrap();
    let a = exec.to_device(&[30_u32, 10, 20]).unwrap();
    let b = exec.to_device(&[30.0_f32, 10.0, 20.0]).unwrap();
    let c = exec.to_device(&[300_u32, 100, 200]).unwrap();
    let out_keys = exec.to_device(&[0_u32; 3]).unwrap();
    let out_a = exec.to_device(&[0_u32; 3]).unwrap();
    let out_b = exec.to_device(&[0.0_f32; 3]).unwrap();
    let out_c = exec.to_device(&[0_u32; 3]).unwrap();

    sort_by_key(
        &exec,
        massively::Zip1(keys.slice(..)),
        massively::Zip3(a.slice(..), b.slice(..), c.slice(..)),
        LessU32,
        massively::Zip1(out_keys.slice_mut(..)),
        massively::Zip3(
            out_a.slice_mut(..),
            out_b.slice_mut(..),
            out_c.slice_mut(..),
        ),
    )
    .unwrap();
    assert_eq!(exec.to_host(&out_keys).unwrap(), vec![1, 2, 3]);
    assert_eq!(exec.to_host(&out_a).unwrap(), vec![10, 20, 30]);
    assert_eq!(exec.to_host(&out_b).unwrap(), vec![10.0, 20.0, 30.0]);
    assert_eq!(exec.to_host(&out_c).unwrap(), vec![100, 200, 300]);
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
    let out_keys = exec.to_device(&[0_u32; 3]).unwrap();
    let out_a = exec.to_device(&[0.0_f32; 3]).unwrap();
    let out_b = exec.to_device(&[0_u32; 3]).unwrap();
    let out_c = exec.to_device(&[0.0_f32; 3]).unwrap();
    let out_d = exec.to_device(&[0_u32; 3]).unwrap();
    let out_e = exec.to_device(&[0.0_f32; 3]).unwrap();
    let out_f = exec.to_device(&[0_u32; 3]).unwrap();
    let out_g = exec.to_device(&[0.0_f32; 3]).unwrap();

    sort_by_key(
        &exec,
        massively::Zip1(keys.slice(..)),
        massively::Zip7(
            a.slice(..),
            b.slice(..),
            c.slice(..),
            d.slice(..),
            e.slice(..),
            f.slice(..),
            g.slice(..),
        ),
        LessU32,
        massively::Zip1(out_keys.slice_mut(..)),
        massively::Zip7(
            out_a.slice_mut(..),
            out_b.slice_mut(..),
            out_c.slice_mut(..),
            out_d.slice_mut(..),
            out_e.slice_mut(..),
            out_f.slice_mut(..),
            out_g.slice_mut(..),
        ),
    )
    .unwrap();
    assert_eq!(exec.to_host(&out_keys).unwrap(), vec![1, 2, 3]);
    assert_eq!(exec.to_host(&out_a).unwrap(), vec![10.0, 20.0, 30.0]);
    assert_eq!(exec.to_host(&out_b).unwrap(), vec![10, 20, 30]);
    assert_eq!(exec.to_host(&out_c).unwrap(), vec![100.0, 200.0, 300.0]);
    assert_eq!(exec.to_host(&out_d).unwrap(), vec![100, 200, 300]);
    assert_eq!(exec.to_host(&out_e).unwrap(), vec![11.0, 21.0, 31.0]);
    assert_eq!(exec.to_host(&out_f).unwrap(), vec![11, 21, 31]);
    assert_eq!(exec.to_host(&out_g).unwrap(), vec![101.0, 201.0, 301.0]);
}

#[test]
fn sort_by_key_accepts_three_column_keys() {
    let exec = exec();
    let k0 = exec.to_device(&[1.0_f32, 1.0, 0.0, 1.0]).unwrap();
    let k1 = exec.to_device(&[2_u32, 1, 9, 1]).unwrap();
    let k2 = exec.to_device(&[0.0_f32, 9.0, 0.0, 3.0]).unwrap();
    let values = exec.to_device(&[20_u32, 19, 90, 13]).unwrap();
    let out_k0 = exec.to_device(&[0.0_f32; 4]).unwrap();
    let out_k1 = exec.to_device(&[0_u32; 4]).unwrap();
    let out_k2 = exec.to_device(&[0.0_f32; 4]).unwrap();
    let out_values = exec.to_device(&[0_u32; 4]).unwrap();

    sort_by_key(
        &exec,
        massively::Zip3(k0.slice(..), k1.slice(..), k2.slice(..)),
        massively::Zip1(values.slice(..)),
        MixedTuple3LexLess,
        massively::Zip3(
            out_k0.slice_mut(..),
            out_k1.slice_mut(..),
            out_k2.slice_mut(..),
        ),
        massively::Zip1(out_values.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_k0).unwrap(), vec![0.0, 1.0, 1.0, 1.0]);
    assert_eq!(exec.to_host(&out_k1).unwrap(), vec![9, 1, 1, 2]);
    assert_eq!(exec.to_host(&out_k2).unwrap(), vec![0.0, 3.0, 9.0, 0.0]);
    assert_eq!(exec.to_host(&out_values).unwrap(), vec![90, 13, 19, 20]);
}

#[test]
fn sort_by_key_accepts_two_column_keys_with_mixed_directions() {
    let exec = exec();
    let k0 = exec.to_device(&[2_u32, 1, 1, 2, 1]).unwrap();
    let k1 = exec.to_device(&[7_u32, 3, 9, 2, 5]).unwrap();
    let values = exec.to_device(&[27_u32, 13, 19, 22, 15]).unwrap();
    let out_k0 = exec.to_device(&[0_u32; 5]).unwrap();
    let out_k1 = exec.to_device(&[0_u32; 5]).unwrap();
    let out_values = exec.to_device(&[0_u32; 5]).unwrap();

    sort_by_key(
        &exec,
        massively::Zip2(k0.slice(..), k1.slice(..)),
        massively::Zip1(values.slice(..)),
        FirstAscSecondDescU32,
        massively::Zip2(out_k0.slice_mut(..), out_k1.slice_mut(..)),
        massively::Zip1(out_values.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_k0).unwrap(), vec![1, 1, 1, 2, 2]);
    assert_eq!(exec.to_host(&out_k1).unwrap(), vec![9, 5, 3, 7, 2]);
    assert_eq!(exec.to_host(&out_values).unwrap(), vec![19, 15, 13, 27, 22]);
}

#[test]
fn sort_by_key_is_stable_for_equal_keys() {
    let exec = exec();
    let keys = exec.to_device(&[2_u32, 1, 2, 1, 2]).unwrap();
    let values = exec.to_device(&[20_u32, 10, 21, 11, 22]).unwrap();
    let out_keys = exec.to_device(&[0_u32; 5]).unwrap();
    let out_values = exec.to_device(&[0_u32; 5]).unwrap();

    sort_by_key(
        &exec,
        massively::Zip1(keys.slice(..)),
        massively::Zip1(values.slice(..)),
        LessU32,
        massively::Zip1(out_keys.slice_mut(..)),
        massively::Zip1(out_values.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_keys).unwrap(), vec![1, 1, 2, 2, 2]);
    assert_eq!(exec.to_host(&out_values).unwrap(), vec![10, 11, 20, 21, 22]);
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
    let out_k0 = exec.to_device(&[0.0_f32; 4]).unwrap();
    let out_k1 = exec.to_device(&[0_u32; 4]).unwrap();
    let out_k2 = exec.to_device(&[0.0_f32; 4]).unwrap();
    let out_a = exec.to_device(&[0.0_f32; 4]).unwrap();
    let out_b = exec.to_device(&[0_u32; 4]).unwrap();
    let out_c = exec.to_device(&[0.0_f32; 4]).unwrap();

    sort_by_key(
        &exec,
        massively::Zip3(k0.slice(..), k1.slice(..), k2.slice(..)),
        massively::Zip3(a.slice(..), b.slice(..), c.slice(..)),
        MixedTuple3LexLess,
        massively::Zip3(
            out_k0.slice_mut(..),
            out_k1.slice_mut(..),
            out_k2.slice_mut(..),
        ),
        massively::Zip3(
            out_a.slice_mut(..),
            out_b.slice_mut(..),
            out_c.slice_mut(..),
        ),
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
    let out_k0 = exec.to_device(&[0.0_f32; 4]).unwrap();
    let out_k1 = exec.to_device(&[0_u32; 4]).unwrap();
    let out_k2 = exec.to_device(&[0.0_f32; 4]).unwrap();
    let out_a = exec.to_device(&[0.0_f32; 4]).unwrap();
    let out_b = exec.to_device(&[0_u32; 4]).unwrap();
    let out_c = exec.to_device(&[0.0_f32; 4]).unwrap();
    let out_d = exec.to_device(&[0_u32; 4]).unwrap();
    let out_e = exec.to_device(&[0.0_f32; 4]).unwrap();
    let out_f = exec.to_device(&[0_u32; 4]).unwrap();
    let out_g = exec.to_device(&[0.0_f32; 4]).unwrap();

    sort_by_key(
        &exec,
        massively::Zip3(k0.slice(..), k1.slice(..), k2.slice(..)),
        massively::Zip7(
            a.slice(..),
            b.slice(..),
            c.slice(..),
            d.slice(..),
            e.slice(..),
            f.slice(..),
            g.slice(..),
        ),
        MixedTuple3LexLess,
        massively::Zip3(
            out_k0.slice_mut(..),
            out_k1.slice_mut(..),
            out_k2.slice_mut(..),
        ),
        massively::Zip7(
            out_a.slice_mut(..),
            out_b.slice_mut(..),
            out_c.slice_mut(..),
            out_d.slice_mut(..),
            out_e.slice_mut(..),
            out_f.slice_mut(..),
            out_g.slice_mut(..),
        ),
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
