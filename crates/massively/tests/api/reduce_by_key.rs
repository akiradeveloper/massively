use crate::common::*;

fn count_reduced_values_after_mvec_slice<Values, Op, Pred>(
    exec: &massively::Executor<WgpuRuntime>,
    keys: massively::SoA1<massively::DeviceSlice<'_, WgpuRuntime, u32>>,
    values: Values,
    init: Values::Item,
    op: Op,
    pred: Pred,
) -> usize
where
    Values: massively::MIter<WgpuRuntime>,
    Op: ReductionOp<WgpuRuntime, Values::Item>,
    Pred: PredicateOp<WgpuRuntime, Values::Item, Env = ()>,
{
    use massively::MVec as _;

    let (_, values) = reduce_by_key(exec, keys, values, EqualU32, init, op).unwrap();
    let values = values.slice(..);
    count_if(exec, values, pred, ()).unwrap()
}

#[test]
fn reduce_by_key_uses_supplied_key_equality() {
    let exec = exec();
    let keys = exec.to_device(&[0_u32, 2, 4, 1, 3]).unwrap();
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0, 5.0]).unwrap();

    let (keys, values) = reduce_by_key(
        &exec,
        massively::SoA1(keys.slice(..)),
        massively::SoA1(values.slice(..)),
        SameParityU32,
        (0.0,),
        Sum,
    )
    .unwrap();
    let massively::SoA1(keys) = keys;
    let massively::SoA1(values) = values;
    assert_eq!(exec.to_host(&keys).unwrap(), vec![4, 3]);
    assert_eq!(exec.to_host(&values).unwrap(), vec![6.0, 9.0]);
}

#[test]
fn reduce_by_key_handles_singleton_runs() {
    let exec = exec();
    let keys = exec.to_device(&[0_u32, 1, 2, 3]).unwrap();
    let values = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();

    let (massively::SoA1(out_keys), massively::SoA1(out_values)) = reduce_by_key(
        &exec,
        massively::SoA1(keys.slice(..)),
        massively::SoA1(values.slice(..)),
        EqualU32,
        (0_u32,),
        Sum,
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_keys).unwrap(), vec![0, 1, 2, 3]);
    assert_eq!(exec.to_host(&out_values).unwrap(), vec![10, 20, 30, 40]);
}

#[test]
fn reduce_by_key_handles_one_run() {
    let exec = exec();
    let keys = exec.to_device(&[0_u32, 0, 0, 0]).unwrap();
    let values = exec.to_device(&[1_u32, 2, 3, 4]).unwrap();

    let (massively::SoA1(out_keys), massively::SoA1(out_values)) = reduce_by_key(
        &exec,
        massively::SoA1(keys.slice(..)),
        massively::SoA1(values.slice(..)),
        EqualU32,
        (0_u32,),
        Sum,
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_keys).unwrap(), vec![0]);
    assert_eq!(exec.to_host(&out_values).unwrap(), vec![10]);
}

#[test]
fn reduce_by_key_handles_all_same_key_long_run() {
    let exec = exec();
    let len = 512;
    let keys = vec![7_u32; len];
    let values = vec![1_u32; len];

    let keys = exec.to_device(&keys).unwrap();
    let values = exec.to_device(&values).unwrap();
    let (massively::SoA1(out_keys), massively::SoA1(out_values)) = reduce_by_key(
        &exec,
        massively::SoA1(keys.slice(..)),
        massively::SoA1(values.slice(..)),
        EqualU32,
        (0_u32,),
        Sum,
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_keys).unwrap(), vec![7]);
    assert_eq!(exec.to_host(&out_values).unwrap(), vec![len as u32]);
}

#[test]
fn reduce_by_key_handles_block_boundary_runs() {
    let exec = exec();
    let mut keys = vec![0_u32; 300];
    keys.extend(vec![1_u32; 20]);
    keys.extend(vec![0_u32; 10]);
    let values = vec![1_u32; keys.len()];

    let keys = exec.to_device(&keys).unwrap();
    let values = exec.to_device(&values).unwrap();
    let (massively::SoA1(out_keys), massively::SoA1(out_values)) = reduce_by_key(
        &exec,
        massively::SoA1(keys.slice(..)),
        massively::SoA1(values.slice(..)),
        EqualU32,
        (0_u32,),
        Sum,
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_keys).unwrap(), vec![0, 1, 0]);
    assert_eq!(exec.to_host(&out_values).unwrap(), vec![300, 20, 10]);
}

#[test]
fn reduce_by_key_accepts_tuple_values() {
    let exec = exec();
    let keys = exec.to_device(&[1_u32, 1, 2, 2, 2]).unwrap();
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let ids = exec.to_device(&[10_u32, 20, 30, 40, 50]).unwrap();

    let (keys, values) = reduce_by_key(
        &exec,
        massively::SoA1(keys.slice(..)),
        massively::SoA2(values.slice(..), ids.slice(..)),
        EqualU32,
        (0.0_f32, 0_u32),
        TupleSum,
    )
    .unwrap();
    let massively::SoA1(keys) = keys;
    let massively::SoA2(values, ids) = values;
    assert_eq!(exec.to_host(&keys).unwrap(), vec![1, 2]);
    assert_eq!(exec.to_host(&values).unwrap(), vec![3.0, 12.0]);
    assert_eq!(exec.to_host(&ids).unwrap(), vec![30, 120]);
}

#[test]
fn reduce_by_key_output_values_support_generic_mvec_slice_for_single_column() {
    let exec = exec();
    let keys = exec.to_device(&[1_u32, 1, 2, 2]).unwrap();
    let values = exec.to_device(&[1_u32, 2, 10, 20]).unwrap();

    let count = count_reduced_values_after_mvec_slice(
        &exec,
        massively::SoA1(keys.slice(..)),
        massively::SoA1(values.slice(..)),
        (0_u32,),
        Sum,
        NonZero,
    );

    assert_eq!(count, 2);
}

#[test]
fn reduce_by_key_output_values_support_generic_mvec_slice_for_multi_column() {
    let exec = exec();
    let keys = exec.to_device(&[1_u32, 1, 2, 2]).unwrap();
    let values = exec.to_device(&[1.0_f32, 2.0, 10.0, 20.0]).unwrap();
    let ids = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();

    let count = count_reduced_values_after_mvec_slice(
        &exec,
        massively::SoA1(keys.slice(..)),
        massively::SoA2(values.slice(..), ids.slice(..)),
        (0.0_f32, 0_u32),
        TupleSum,
        PairMixedFirstPositive,
    );

    assert_eq!(count, 2);
}

#[test]
fn reduce_by_key_accepts_three_tuple_values() {
    let exec = exec();
    let keys = exec.to_device(&[1_u32, 1, 2, 2, 2]).unwrap();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20, 30, 40, 50]).unwrap();
    let c = exec
        .to_device(&[100.0_f32, 200.0, 300.0, 400.0, 500.0])
        .unwrap();

    let (keys, values) = reduce_by_key(
        &exec,
        massively::SoA1(keys.slice(..)),
        massively::SoA3(a.slice(..), b.slice(..), c.slice(..)),
        EqualU32,
        (0.0_f32, 0_u32, 0.0_f32),
        TupleSum,
    )
    .unwrap();
    let massively::SoA1(keys) = keys;
    let massively::SoA3(a, b, c) = values;
    assert_eq!(exec.to_host(&keys).unwrap(), vec![1, 2]);
    assert_eq!(exec.to_host(&a).unwrap(), vec![3.0, 12.0]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![30, 120]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![300.0, 1200.0]);
}

#[test]
fn reduce_by_key_accepts_three_column_keys() {
    let exec = exec();
    let k0 = exec.to_device(&[1.0_f32, 1.0, 1.0, 1.0, 2.0, 2.0]).unwrap();
    let k1 = exec.to_device(&[0_u32, 0, 1, 1, 0, 0]).unwrap();
    let k2 = exec.to_device(&[5.0_f32, 5.0, 5.0, 6.0, 1.0, 1.0]).unwrap();
    let values = exec.to_device(&[10_u32, 20, 30, 40, 50, 60]).unwrap();

    let (massively::SoA3(out_k0, out_k1, out_k2), massively::SoA1(out_values)) = reduce_by_key(
        &exec,
        massively::SoA3(k0.slice(..), k1.slice(..), k2.slice(..)),
        massively::SoA1(values.slice(..)),
        MixedTuple3Equal,
        (0_u32,),
        Sum,
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_k0).unwrap(), vec![1.0, 1.0, 1.0, 2.0]);
    assert_eq!(exec.to_host(&out_k1).unwrap(), vec![0, 1, 1, 0]);
    assert_eq!(exec.to_host(&out_k2).unwrap(), vec![5.0, 5.0, 6.0, 1.0]);
    assert_eq!(exec.to_host(&out_values).unwrap(), vec![30, 30, 40, 110]);
}

#[test]
fn reduce_by_key_accepts_three_column_keys_and_tuple_values() {
    let exec = exec();
    let k0 = exec.to_device(&[1.0_f32, 1.0, 1.0, 1.0, 2.0, 2.0]).unwrap();
    let k1 = exec.to_device(&[0_u32, 0, 1, 1, 0, 0]).unwrap();
    let k2 = exec.to_device(&[5.0_f32, 5.0, 5.0, 6.0, 1.0, 1.0]).unwrap();
    let a = exec
        .to_device(&[10.0_f32, 20.0, 30.0, 40.0, 50.0, 60.0])
        .unwrap();
    let b = exec.to_device(&[1_u32, 2, 3, 4, 5, 6]).unwrap();
    let c = exec
        .to_device(&[100.0_f32, 200.0, 300.0, 400.0, 500.0, 600.0])
        .unwrap();

    let (massively::SoA3(out_k0, out_k1, out_k2), massively::SoA3(out_a, out_b, out_c)) =
        reduce_by_key(
            &exec,
            massively::SoA3(k0.slice(..), k1.slice(..), k2.slice(..)),
            massively::SoA3(a.slice(..), b.slice(..), c.slice(..)),
            MixedTuple3Equal,
            (0.0_f32, 0_u32, 0.0_f32),
            TupleSum,
        )
        .unwrap();

    assert_eq!(exec.to_host(&out_k0).unwrap(), vec![1.0, 1.0, 1.0, 2.0]);
    assert_eq!(exec.to_host(&out_k1).unwrap(), vec![0, 1, 1, 0]);
    assert_eq!(exec.to_host(&out_k2).unwrap(), vec![5.0, 5.0, 6.0, 1.0]);
    assert_eq!(exec.to_host(&out_a).unwrap(), vec![30.0, 30.0, 40.0, 110.0]);
    assert_eq!(exec.to_host(&out_b).unwrap(), vec![3, 3, 4, 11]);
    assert_eq!(
        exec.to_host(&out_c).unwrap(),
        vec![300.0, 300.0, 400.0, 1100.0]
    );
}

#[test]
fn reduce_by_key_accepts_three_column_keys_and_seven_column_values() {
    let exec = exec();
    let k0 = exec.to_device(&[1.0_f32, 1.0, 1.0, 1.0, 2.0, 2.0]).unwrap();
    let k1 = exec.to_device(&[0_u32, 0, 1, 1, 0, 0]).unwrap();
    let k2 = exec.to_device(&[5.0_f32, 5.0, 5.0, 6.0, 1.0, 1.0]).unwrap();
    let a = exec
        .to_device(&[10.0_f32, 20.0, 30.0, 40.0, 50.0, 60.0])
        .unwrap();
    let b = exec.to_device(&[1_u32, 2, 3, 4, 5, 6]).unwrap();
    let c = exec
        .to_device(&[100.0_f32, 200.0, 300.0, 400.0, 500.0, 600.0])
        .unwrap();
    let d = exec.to_device(&[10_u32, 20, 30, 40, 50, 60]).unwrap();
    let e = exec.to_device(&[1.5_f32, 2.5, 3.5, 4.5, 5.5, 6.5]).unwrap();
    let f = exec.to_device(&[7_u32, 8, 9, 10, 11, 12]).unwrap();
    let g = exec
        .to_device(&[70.0_f32, 80.0, 90.0, 100.0, 110.0, 120.0])
        .unwrap();

    let (
        massively::SoA3(out_k0, out_k1, out_k2),
        massively::SoA7(out_a, out_b, out_c, out_d, out_e, out_f, out_g),
    ) = reduce_by_key(
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
        MixedTuple3Equal,
        (0.0_f32, 0_u32, 0.0_f32, 0_u32, 0.0_f32, 0_u32, 0.0_f32),
        TupleSum,
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_k0).unwrap(), vec![1.0, 1.0, 1.0, 2.0]);
    assert_eq!(exec.to_host(&out_k1).unwrap(), vec![0, 1, 1, 0]);
    assert_eq!(exec.to_host(&out_k2).unwrap(), vec![5.0, 5.0, 6.0, 1.0]);
    assert_eq!(exec.to_host(&out_a).unwrap(), vec![30.0, 30.0, 40.0, 110.0]);
    assert_eq!(exec.to_host(&out_b).unwrap(), vec![3, 3, 4, 11]);
    assert_eq!(
        exec.to_host(&out_c).unwrap(),
        vec![300.0, 300.0, 400.0, 1100.0]
    );
    assert_eq!(exec.to_host(&out_d).unwrap(), vec![30, 30, 40, 110]);
    assert_eq!(exec.to_host(&out_e).unwrap(), vec![4.0, 3.5, 4.5, 12.0]);
    assert_eq!(exec.to_host(&out_f).unwrap(), vec![15, 9, 10, 23]);
    assert_eq!(
        exec.to_host(&out_g).unwrap(),
        vec![150.0, 90.0, 100.0, 230.0]
    );
}
