use crate::common::*;

fn merge_by_key_with_generic_right<
    LeftKeys,
    LeftValues,
    RightKeys,
    RightValues,
    Less,
    KeyOutput,
    ValueOutput,
>(
    exec: &Executor<WgpuRuntime>,
    left_keys: LeftKeys,
    left_values: LeftValues,
    right_keys: RightKeys,
    right_values: RightValues,
    less: Less,
    out_k: KeyOutput,
    out_v: ValueOutput,
) -> Result<(), massively::Error>
where
    LeftKeys: massively::MIter<WgpuRuntime>,
    LeftValues: massively::MIter<WgpuRuntime>,
    RightKeys: massively::MIter<WgpuRuntime, Item = LeftKeys::Item>,
    RightValues: massively::MIter<WgpuRuntime, Item = LeftValues::Item>,
    Less: BinaryPredicateOp<WgpuRuntime, LeftKeys::Item>,
    KeyOutput: massively::MIterMut<WgpuRuntime, Item = LeftKeys::Item>,
    LeftKeys::Item: massively::MAlloc<WgpuRuntime>,
    ValueOutput: massively::MIterMut<WgpuRuntime, Item = LeftValues::Item>,
    LeftValues::Item: massively::MAlloc<WgpuRuntime>,
{
    merge_by_key(
        exec,
        left_keys,
        left_values,
        right_keys,
        right_values,
        less,
        out_k,
        out_v,
    )
}

#[test]
fn merge_by_key_accepts_generic_right_without_inner_equality_bound() {
    let exec = exec();
    let left_keys = exec.to_device(&[0_u32, 2, 2, 5]).unwrap();
    let right_keys = exec.to_device(&[1_u32, 2, 4]).unwrap();
    let left_values = exec.to_device(&[0.0_f32, 20.0, 21.0, 50.0]).unwrap();
    let left_ids = exec.to_device(&[0_u32, 20, 21, 50]).unwrap();
    let right_values = exec.to_device(&[10.0_f32, 22.0, 40.0]).unwrap();
    let right_ids = exec.to_device(&[10_u32, 22, 40]).unwrap();
    let out_keys = exec.to_device(&[0_u32; 7]).unwrap();
    let out_values = exec.to_device(&[0.0_f32; 7]).unwrap();
    let out_ids = exec.to_device(&[0_u32; 7]).unwrap();

    merge_by_key_with_generic_right(
        &exec,
        massively::SoA1(left_keys.slice(..)),
        massively::SoA2(left_values.slice(..), left_ids.slice(..)),
        massively::SoA1(right_keys.slice(..)),
        massively::SoA2(right_values.slice(..), right_ids.slice(..)),
        LessU32,
        massively::SoA1(out_keys.slice_mut(..)),
        massively::SoA2(out_values.slice_mut(..), out_ids.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_keys).unwrap(), vec![0, 1, 2, 2, 2, 4, 5]);
    assert_eq!(
        exec.to_host(&out_values).unwrap(),
        vec![0.0, 10.0, 20.0, 21.0, 22.0, 40.0, 50.0]
    );
    assert_eq!(
        exec.to_host(&out_ids).unwrap(),
        vec![0, 10, 20, 21, 22, 40, 50]
    );
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
    let out_keys = exec.to_device(&[0_u32; 7]).unwrap();
    let out_values = exec.to_device(&[0.0_f32; 7]).unwrap();
    let out_ids = exec.to_device(&[0_u32; 7]).unwrap();

    merge_by_key(
        &exec,
        massively::SoA1(left_keys.slice(..)),
        massively::SoA2(left_values.slice(..), left_ids.slice(..)),
        massively::SoA1(right_keys.slice(..)),
        massively::SoA2(right_values.slice(..), right_ids.slice(..)),
        LessU32,
        massively::SoA1(out_keys.slice_mut(..)),
        massively::SoA2(out_values.slice_mut(..), out_ids.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_keys).unwrap(), vec![0, 1, 2, 2, 2, 4, 5]);
    assert_eq!(
        exec.to_host(&out_values).unwrap(),
        vec![0.0, 10.0, 20.0, 21.0, 22.0, 40.0, 50.0]
    );
    assert_eq!(
        exec.to_host(&out_ids).unwrap(),
        vec![0, 10, 20, 21, 22, 40, 50]
    );
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
    let out_k0 = exec.to_device(&[0.0_f32; 5]).unwrap();
    let out_k1 = exec.to_device(&[0_u32; 5]).unwrap();
    let out_k2 = exec.to_device(&[0.0_f32; 5]).unwrap();
    let out_values = exec.to_device(&[0_u32; 5]).unwrap();

    merge_by_key(
        &exec,
        massively::SoA3(lk0.slice(..), lk1.slice(..), lk2.slice(..)),
        massively::SoA1(left_values.slice(..)),
        massively::SoA3(rk0.slice(..), rk1.slice(..), rk2.slice(..)),
        massively::SoA1(right_values.slice(..)),
        MixedTuple3LexLess,
        massively::SoA3(
            out_k0.slice_mut(..),
            out_k1.slice_mut(..),
            out_k2.slice_mut(..),
        ),
        massively::SoA1(out_values.slice_mut(..)),
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
    let out_k0 = exec.to_device(&[0.0_f32; 5]).unwrap();
    let out_k1 = exec.to_device(&[0_u32; 5]).unwrap();
    let out_k2 = exec.to_device(&[0.0_f32; 5]).unwrap();
    let out_a = exec.to_device(&[0.0_f32; 5]).unwrap();
    let out_b = exec.to_device(&[0_u32; 5]).unwrap();
    let out_c = exec.to_device(&[0.0_f32; 5]).unwrap();

    merge_by_key(
        &exec,
        massively::SoA3(lk0.slice(..), lk1.slice(..), lk2.slice(..)),
        massively::SoA3(la.slice(..), lb.slice(..), lc.slice(..)),
        massively::SoA3(rk0.slice(..), rk1.slice(..), rk2.slice(..)),
        massively::SoA3(ra.slice(..), rb.slice(..), rc.slice(..)),
        MixedTuple3LexLess,
        massively::SoA3(
            out_k0.slice_mut(..),
            out_k1.slice_mut(..),
            out_k2.slice_mut(..),
        ),
        massively::SoA3(
            out_a.slice_mut(..),
            out_b.slice_mut(..),
            out_c.slice_mut(..),
        ),
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
    let out_k0 = exec.to_device(&[0.0_f32; 5]).unwrap();
    let out_k1 = exec.to_device(&[0_u32; 5]).unwrap();
    let out_k2 = exec.to_device(&[0.0_f32; 5]).unwrap();
    let out_a = exec.to_device(&[0.0_f32; 5]).unwrap();
    let out_b = exec.to_device(&[0_u32; 5]).unwrap();
    let out_c = exec.to_device(&[0.0_f32; 5]).unwrap();
    let out_d = exec.to_device(&[0_u32; 5]).unwrap();
    let out_e = exec.to_device(&[0.0_f32; 5]).unwrap();
    let out_f = exec.to_device(&[0_u32; 5]).unwrap();
    let out_g = exec.to_device(&[0.0_f32; 5]).unwrap();

    merge_by_key(
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
        massively::SoA3(
            out_k0.slice_mut(..),
            out_k1.slice_mut(..),
            out_k2.slice_mut(..),
        ),
        massively::SoA7(
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
