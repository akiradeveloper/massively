use crate::common::*;

#[test]
fn gather_where_accepts_bool_stencil() {
    let exec = exec();
    let a = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let b = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let indices = exec.to_device(&[3_u32, 2, 1, 0]).unwrap();
    let stencil = bool_stencil(4, IndexGe2);

    let out_a = exec.to_device(&[0_u32; 4]).unwrap();
    let out_b = exec.to_device(&[0.0_f32; 4]).unwrap();
    gather_where(
        &exec,
        massively::Zip2(a.slice(..), b.slice(..)),
        indices.slice(..),
        stencil,
        massively::Zip2(out_a.slice_mut(..), out_b.slice_mut(..)),
    )
    .unwrap();
    assert_eq!(exec.to_host(&out_a).unwrap(), vec![0, 0, 20, 10]);
    assert_eq!(exec.to_host(&out_b).unwrap(), vec![0.0, 0.0, 2.0, 1.0]);
}

#[test]
fn gather_where_leaves_output_unchanged_when_no_flags_are_selected() {
    let exec = exec();
    let values = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let indices = exec.to_device(&[3_u32, 2, 1, 0]).unwrap();
    let stencil = massively::lazy::constant(false).take(4);
    let output = exec.to_device(&[99_u32, 98, 97, 96]).unwrap();

    gather_where(
        &exec,
        massively::Zip1(values.slice(..)),
        indices.slice(..),
        stencil,
        massively::Zip1(output.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![99, 98, 97, 96]);
}

#[test]
fn gather_where_gathers_all_values_when_all_flags_are_selected() {
    let exec = exec();
    let values = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let indices = exec.to_device(&[3_u32, 2, 1, 0]).unwrap();
    let stencil = massively::lazy::constant(true).take(4);
    let output = exec.to_device(&[0_u32; 4]).unwrap();

    gather_where(
        &exec,
        massively::Zip1(values.slice(..)),
        indices.slice(..),
        stencil,
        massively::Zip1(output.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![40, 30, 20, 10]);
}

#[test]
fn gather_where_accepts_sliced_input_and_output() {
    let exec = exec();
    let values = exec.to_device(&[99_u32, 10, 20, 30, 40, 88]).unwrap();
    let indices = exec.to_device(&[2_u32, 1, 0]).unwrap();
    let stencil = bool_stencil(3, IndexNot1);
    let output = exec.to_device(&[7_u32, 7, 7, 7, 7]).unwrap();

    gather_where(
        &exec,
        massively::Zip1(values.slice(1..5)),
        indices.slice(..),
        stencil,
        massively::Zip1(output.slice_mut(1..4)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![7, 30, 7, 10, 7]);
}

#[test]
fn gather_where_accepts_lazy_indices_and_stencil() {
    let exec = exec();
    let values = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let output = exec.to_device(&[99_u32; 4]).unwrap();

    gather_where(
        &exec,
        massively::Zip1(values.slice(..)),
        massively::lazy::counting(0).take(4),
        bool_stencil(4, IndexNonZero),
        massively::Zip1(output.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![99, 20, 30, 40]);
}

#[test]
fn gather_where_accepts_lazy_constant_indices() {
    let exec = exec();
    let values = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let output = exec.to_device(&[99_u32; 4]).unwrap();

    gather_where(
        &exec,
        massively::Zip1(values.slice(..)),
        massively::lazy::constant(2_u32).take(4),
        massively::lazy::constant(true).take(4),
        massively::Zip1(output.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![30, 30, 30, 30]);
}
