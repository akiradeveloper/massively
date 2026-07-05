use crate::common::*;

#[test]
fn unique_by_key_accepts_tuple_values() {
    let exec = exec();
    let keys = exec.to_device(&[0_u32, 0, 1]).unwrap();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let c = exec.to_device(&[100.0_f32, 200.0, 300.0]).unwrap();
    let out_keys = exec.to_device(&[0_u32; 3]).unwrap();
    let out_a = exec.to_device(&[0.0_f32; 3]).unwrap();
    let out_b = exec.to_device(&[0_u32; 3]).unwrap();
    let out_c = exec.to_device(&[0.0_f32; 3]).unwrap();

    let len = unique_by_key(
        &exec,
        massively::Zip1(keys.slice(..)),
        massively::Zip3(a.slice(..), b.slice(..), c.slice(..)),
        EqualU32,
        massively::Zip1(out_keys.slice_mut(..)),
        massively::Zip3(
            out_a.slice_mut(..),
            out_b.slice_mut(..),
            out_c.slice_mut(..),
        ),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_keys.slice(..len)).unwrap(), vec![0, 1]);
    assert_eq!(exec.to_host(&out_a.slice(..len)).unwrap(), vec![1.0, 3.0]);
    assert_eq!(exec.to_host(&out_b.slice(..len)).unwrap(), vec![10, 30]);
    assert_eq!(
        exec.to_host(&out_c.slice(..len)).unwrap(),
        vec![100.0, 300.0]
    );
}

#[test]
fn unique_by_key_accepts_tuple_values_with_multiple_runs() {
    let exec = exec();
    let keys = exec.to_device(&[0_u32, 0, 0, 2, 3, 3]).unwrap();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20, 30, 40, 50, 60]).unwrap();
    let c = exec
        .to_device(&[100.0_f32, 200.0, 300.0, 400.0, 500.0, 600.0])
        .unwrap();
    let out_keys = exec.to_device(&[0_u32; 6]).unwrap();
    let out_a = exec.to_device(&[0.0_f32; 6]).unwrap();
    let out_b = exec.to_device(&[0_u32; 6]).unwrap();
    let out_c = exec.to_device(&[0.0_f32; 6]).unwrap();

    let len = unique_by_key(
        &exec,
        massively::Zip1(keys.slice(..)),
        massively::Zip3(a.slice(..), b.slice(..), c.slice(..)),
        EqualU32,
        massively::Zip1(out_keys.slice_mut(..)),
        massively::Zip3(
            out_a.slice_mut(..),
            out_b.slice_mut(..),
            out_c.slice_mut(..),
        ),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_keys.slice(..len)).unwrap(), vec![0, 2, 3]);
    assert_eq!(
        exec.to_host(&out_a.slice(..len)).unwrap(),
        vec![1.0, 4.0, 5.0]
    );
    assert_eq!(exec.to_host(&out_b.slice(..len)).unwrap(), vec![10, 40, 50]);
    assert_eq!(
        exec.to_host(&out_c.slice(..len)).unwrap(),
        vec![100.0, 400.0, 500.0]
    );
}

#[test]
fn unique_by_key_accepts_three_column_keys() {
    let exec = exec();
    let k0 = exec.to_device(&[1.0_f32, 1.0, 1.0, 2.0, 2.0]).unwrap();
    let k1 = exec.to_device(&[10_u32, 10, 11, 10, 10]).unwrap();
    let k2 = exec
        .to_device(&[100.0_f32, 100.0, 100.0, 200.0, 200.0])
        .unwrap();
    let values = exec.to_device(&[1000_u32, 1001, 1002, 2000, 2001]).unwrap();
    let out_k0 = exec.to_device(&[0.0_f32; 5]).unwrap();
    let out_k1 = exec.to_device(&[0_u32; 5]).unwrap();
    let out_k2 = exec.to_device(&[0.0_f32; 5]).unwrap();
    let out_values = exec.to_device(&[0_u32; 5]).unwrap();

    let len = unique_by_key(
        &exec,
        massively::Zip3(k0.slice(..), k1.slice(..), k2.slice(..)),
        massively::Zip1(values.slice(..)),
        MixedTuple3Equal,
        massively::Zip3(
            out_k0.slice_mut(..),
            out_k1.slice_mut(..),
            out_k2.slice_mut(..),
        ),
        massively::Zip1(out_values.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(
        exec.to_host(&out_k0.slice(..len)).unwrap(),
        vec![1.0, 1.0, 2.0]
    );
    assert_eq!(
        exec.to_host(&out_k1.slice(..len)).unwrap(),
        vec![10, 11, 10]
    );
    assert_eq!(
        exec.to_host(&out_k2.slice(..len)).unwrap(),
        vec![100.0, 100.0, 200.0]
    );
    assert_eq!(
        exec.to_host(&out_values.slice(..len)).unwrap(),
        vec![1000, 1002, 2000]
    );
}

#[test]
fn unique_by_key_accepts_three_column_keys_and_seven_column_values() {
    let exec = exec();
    let k0 = exec.to_device(&[1.0_f32, 1.0, 1.0, 2.0, 2.0]).unwrap();
    let k1 = exec.to_device(&[10_u32, 10, 11, 10, 10]).unwrap();
    let k2 = exec
        .to_device(&[100.0_f32, 100.0, 100.0, 200.0, 200.0])
        .unwrap();
    let a = exec.to_device(&[10_u32, 11, 12, 20, 21]).unwrap();
    let b = exec.to_device(&[110_u32, 111, 112, 120, 121]).unwrap();
    let c = exec.to_device(&[210_u32, 211, 212, 220, 221]).unwrap();
    let d = exec.to_device(&[310_u32, 311, 312, 320, 321]).unwrap();
    let e = exec.to_device(&[410_u32, 411, 412, 420, 421]).unwrap();
    let f = exec.to_device(&[510_u32, 511, 512, 520, 521]).unwrap();
    let g = exec.to_device(&[610_u32, 611, 612, 620, 621]).unwrap();
    let out_k0 = exec.to_device(&[0.0_f32; 5]).unwrap();
    let out_k1 = exec.to_device(&[0_u32; 5]).unwrap();
    let out_k2 = exec.to_device(&[0.0_f32; 5]).unwrap();
    let out_a = exec.to_device(&[0_u32; 5]).unwrap();
    let out_b = exec.to_device(&[0_u32; 5]).unwrap();
    let out_c = exec.to_device(&[0_u32; 5]).unwrap();
    let out_d = exec.to_device(&[0_u32; 5]).unwrap();
    let out_e = exec.to_device(&[0_u32; 5]).unwrap();
    let out_f = exec.to_device(&[0_u32; 5]).unwrap();
    let out_g = exec.to_device(&[0_u32; 5]).unwrap();

    let len = unique_by_key(
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
        MixedTuple3Equal,
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

    assert_eq!(
        exec.to_host(&out_k0.slice(..len)).unwrap(),
        vec![1.0, 1.0, 2.0]
    );
    assert_eq!(
        exec.to_host(&out_k1.slice(..len)).unwrap(),
        vec![10, 11, 10]
    );
    assert_eq!(
        exec.to_host(&out_k2.slice(..len)).unwrap(),
        vec![100.0, 100.0, 200.0]
    );
    assert_eq!(exec.to_host(&out_a.slice(..len)).unwrap(), vec![10, 12, 20]);
    assert_eq!(
        exec.to_host(&out_b.slice(..len)).unwrap(),
        vec![110, 112, 120]
    );
    assert_eq!(
        exec.to_host(&out_c.slice(..len)).unwrap(),
        vec![210, 212, 220]
    );
    assert_eq!(
        exec.to_host(&out_d.slice(..len)).unwrap(),
        vec![310, 312, 320]
    );
    assert_eq!(
        exec.to_host(&out_e.slice(..len)).unwrap(),
        vec![410, 412, 420]
    );
    assert_eq!(
        exec.to_host(&out_f.slice(..len)).unwrap(),
        vec![510, 512, 520]
    );
    assert_eq!(
        exec.to_host(&out_g.slice(..len)).unwrap(),
        vec![610, 612, 620]
    );
}
