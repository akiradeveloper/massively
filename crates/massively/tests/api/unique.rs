use crate::common::*;

#[test]
fn unique_keeps_one_value_when_all_values_are_equal() {
    let exec = exec();
    let values = exec.to_device(&[7.0_f32, 7.0, 7.0, 7.0]).unwrap();
    let output = exec.to_device(&[0.0_f32; 4]).unwrap();

    let len = unique(
        &exec,
        massively::Zip1(values.slice(..)),
        EqualF32,
        massively::Zip1(output.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output.slice(..len)).unwrap(), vec![7.0]);
}

#[test]
fn unique_keeps_all_values_when_no_adjacent_values_are_equal() {
    let exec = exec();
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let output = exec.to_device(&[0.0_f32; 4]).unwrap();

    let len = unique(
        &exec,
        massively::Zip1(values.slice(..)),
        EqualF32,
        massively::Zip1(output.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(
        exec.to_host(&output.slice(..len)).unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
}

#[test]
fn unique_accepts_seven_tuple_columns() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 1.0, 2.0, 2.0, 3.0]).unwrap();
    let b = exec.to_device(&[10_u32, 10, 20, 20, 30]).unwrap();
    let c = exec
        .to_device(&[100.0_f32, 100.0, 200.0, 201.0, 300.0])
        .unwrap();
    let d = exec.to_device(&[1000_u32, 1000, 2000, 2001, 3000]).unwrap();
    let e = exec
        .to_device(&[10000.0_f32, 10000.0, 20000.0, 20001.0, 30000.0])
        .unwrap();
    let f = exec
        .to_device(&[100000_u32, 100000, 200000, 200001, 300000])
        .unwrap();
    let g = exec
        .to_device(&[1000000.0_f32, 1000000.0, 2000000.0, 2000001.0, 3000000.0])
        .unwrap();
    let out_a = exec.to_device(&[0.0_f32; 5]).unwrap();
    let out_b = exec.to_device(&[0_u32; 5]).unwrap();
    let out_c = exec.to_device(&[0.0_f32; 5]).unwrap();
    let out_d = exec.to_device(&[0_u32; 5]).unwrap();
    let out_e = exec.to_device(&[0.0_f32; 5]).unwrap();
    let out_f = exec.to_device(&[0_u32; 5]).unwrap();
    let out_g = exec.to_device(&[0.0_f32; 5]).unwrap();

    let len = unique(
        &exec,
        massively::Zip7(
            a.slice(..),
            b.slice(..),
            c.slice(..),
            d.slice(..),
            e.slice(..),
            f.slice(..),
            g.slice(..),
        ),
        Tuple7MixedEqual,
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
        exec.to_host(&out_a.slice(..len)).unwrap(),
        vec![1.0, 2.0, 2.0, 3.0]
    );
    assert_eq!(
        exec.to_host(&out_b.slice(..len)).unwrap(),
        vec![10, 20, 20, 30]
    );
    assert_eq!(
        exec.to_host(&out_c.slice(..len)).unwrap(),
        vec![100.0, 200.0, 201.0, 300.0]
    );
    assert_eq!(
        exec.to_host(&out_d.slice(..len)).unwrap(),
        vec![1000, 2000, 2001, 3000]
    );
    assert_eq!(
        exec.to_host(&out_e.slice(..len)).unwrap(),
        vec![10000.0, 20000.0, 20001.0, 30000.0]
    );
    assert_eq!(
        exec.to_host(&out_f.slice(..len)).unwrap(),
        vec![100000, 200000, 200001, 300000]
    );
    assert_eq!(
        exec.to_host(&out_g.slice(..len)).unwrap(),
        vec![1000000.0, 2000000.0, 2000001.0, 3000000.0]
    );
}
