use crate::common::*;

fn lower_bound_with_generic_values<Input, Values, Less>(
    exec: &Executor<WgpuRuntime>,
    input: Input,
    values: Values,
    less: Less,
) -> Vec<u32>
where
    Input: massively::MIter<WgpuRuntime>,
    Values: massively::MIter<WgpuRuntime, Item = Input::Item>,
    Less: BinaryPredicateOp<WgpuRuntime, Input::Item>,
{
    let output = lower_bound(exec, input, values, less).unwrap();
    exec.to_host(&output).unwrap()
}

#[test]
fn lower_bound_handles_multiple_values() {
    let exec = exec();
    let xs = exec.to_device(&[0_u32, 0, 2, 2, 2]).unwrap();
    let vs = exec.to_device(&[0_u32, 1, 2]).unwrap();
    let output = lower_bound(
        &exec,
        massively::SoA1(xs.slice(..)),
        massively::SoA1(vs.slice(..)),
        LessU32,
    )
    .unwrap();
    assert_eq!(exec.to_host(&output).unwrap(), vec![0, 2, 2]);
}

#[test]
fn lower_bound_handles_empty_inputs() {
    let exec = exec();
    let xs = exec.to_device(&[] as &[u32]).unwrap();
    let vs = exec.to_device(&[0_u32, 1, 2]).unwrap();
    let output = lower_bound(
        &exec,
        massively::SoA1(xs.slice(..)),
        massively::SoA1(vs.slice(..)),
        LessU32,
    )
    .unwrap();
    assert_eq!(exec.to_host(&output).unwrap(), vec![0, 0, 0]);

    let xs = exec.to_device(&[1_u32, 2, 3]).unwrap();
    let vs = exec.to_device(&[] as &[u32]).unwrap();
    let output = lower_bound(
        &exec,
        massively::SoA1(xs.slice(..)),
        massively::SoA1(vs.slice(..)),
        LessU32,
    )
    .unwrap();
    assert!(exec.to_host(&output).unwrap().is_empty());
}

#[test]
fn lower_bound_accepts_generic_values_without_inner_equality_bound() {
    let exec = exec();
    let xs = exec.to_device(&[0_u32, 0, 2, 2, 2]).unwrap();
    let vs = exec.to_device(&[0_u32, 1, 2]).unwrap();
    let output = lower_bound_with_generic_values(
        &exec,
        massively::SoA1(xs.slice(..)),
        massively::SoA1(vs.slice(..)),
        LessU32,
    );
    assert_eq!(output, vec![0, 2, 2]);
}

#[test]
fn lower_bound_accepts_borrowed_tuple_columns() {
    let exec = exec();
    let k = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let l = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let qk = exec.to_device(&[0.5_f32, 3.0, 5.0]).unwrap();
    let ql = exec.to_device(&[5_u32, 30, 50]).unwrap();
    let input = massively::SoA2(k.slice(..), l.slice(..));
    let values = massively::SoA2(qk.slice(..), ql.slice(..));
    let output = lower_bound(&exec, input, values, MixedTupleLess).unwrap();
    assert_eq!(exec.to_host(&output).unwrap(), vec![0, 2, 4]);
}

#[test]
fn lower_bound_accepts_generic_tuple_values_without_inner_equality_bound() {
    let exec = exec();
    let k = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let l = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let qk = exec.to_device(&[0.5_f32, 3.0, 5.0]).unwrap();
    let ql = exec.to_device(&[5_u32, 30, 50]).unwrap();
    let output = lower_bound_with_generic_values(
        &exec,
        massively::SoA2(k.slice(..), l.slice(..)),
        massively::SoA2(qk.slice(..), ql.slice(..)),
        MixedTupleLess,
    );
    assert_eq!(output, vec![0, 2, 4]);
}
