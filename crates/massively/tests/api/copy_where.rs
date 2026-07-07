use crate::common::*;

#[test]
fn copy_where_accepts_u32_flags_for_heterogeneous_tuple_values() {
    let exec = exec();
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let tags = exec.to_device(&[10_u32, 20, 20, 30]).unwrap();
    let stencil = exec.to_device(&[0_u32, 1, 1, 0]).unwrap();
    let out_values = exec.to_device(&[0.0_f32; 4]).unwrap();
    let out_tags = exec.to_device(&[0_u32; 4]).unwrap();

    let len = copy_where(
        &exec,
        massively::Zip2(values.slice(..), tags.slice(..)),
        stencil.slice(..),
        massively::Zip2(out_values.slice_mut(..), out_tags.slice_mut(..)),
    )
    .unwrap();
    assert_eq!(
        exec.to_host(&out_values.slice(..len)).unwrap(),
        vec![2.0, 3.0]
    );
    assert_eq!(exec.to_host(&out_tags.slice(..len)).unwrap(), vec![20, 20]);
}

#[test]
fn copy_where_accepts_three_tuple_columns() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let c = exec.to_device(&[100.0_f32, 200.0, 300.0]).unwrap();
    let stencil = exec.to_device(&[0_u32, 1, 1]).unwrap();
    let out_a = exec.to_device(&[0.0_f32; 3]).unwrap();
    let out_b = exec.to_device(&[0_u32; 3]).unwrap();
    let out_c = exec.to_device(&[0.0_f32; 3]).unwrap();

    let len = copy_where(
        &exec,
        massively::Zip3(a.slice(..), b.slice(..), c.slice(..)),
        stencil.slice(..),
        massively::Zip3(
            out_a.slice_mut(..),
            out_b.slice_mut(..),
            out_c.slice_mut(..),
        ),
    )
    .unwrap();
    assert_eq!(exec.to_host(&out_a.slice(..len)).unwrap(), vec![2.0, 3.0]);
    assert_eq!(exec.to_host(&out_b.slice(..len)).unwrap(), vec![20, 30]);
    assert_eq!(
        exec.to_host(&out_c.slice(..len)).unwrap(),
        vec![200.0, 300.0]
    );
}

#[test]
fn copy_where_accepts_seven_tuple_columns() {
    let exec = exec();
    let a = exec.to_device(&[1_u32, 2, 3, 4, 5]).unwrap();
    let b = exec.to_device(&[11_u32, 12, 13, 14, 15]).unwrap();
    let c = exec.to_device(&[21_u32, 22, 23, 24, 25]).unwrap();
    let d = exec.to_device(&[31_u32, 32, 33, 34, 35]).unwrap();
    let e = exec.to_device(&[41_u32, 42, 43, 44, 45]).unwrap();
    let f = exec.to_device(&[51_u32, 52, 53, 54, 55]).unwrap();
    let g = exec.to_device(&[61_u32, 62, 63, 64, 65]).unwrap();
    let stencil = exec.to_device(&[1_u32, 0, 1, 0, 1]).unwrap();
    let out_a = exec.to_device(&[0_u32; 5]).unwrap();
    let out_b = exec.to_device(&[0_u32; 5]).unwrap();
    let out_c = exec.to_device(&[0_u32; 5]).unwrap();
    let out_d = exec.to_device(&[0_u32; 5]).unwrap();
    let out_e = exec.to_device(&[0_u32; 5]).unwrap();
    let out_f = exec.to_device(&[0_u32; 5]).unwrap();
    let out_g = exec.to_device(&[0_u32; 5]).unwrap();

    let len = copy_where(
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
        stencil.slice(..),
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
    assert_eq!(exec.to_host(&out_a.slice(..len)).unwrap(), vec![1, 3, 5]);
    assert_eq!(exec.to_host(&out_b.slice(..len)).unwrap(), vec![11, 13, 15]);
    assert_eq!(exec.to_host(&out_c.slice(..len)).unwrap(), vec![21, 23, 25]);
    assert_eq!(exec.to_host(&out_d.slice(..len)).unwrap(), vec![31, 33, 35]);
    assert_eq!(exec.to_host(&out_e.slice(..len)).unwrap(), vec![41, 43, 45]);
    assert_eq!(exec.to_host(&out_f.slice(..len)).unwrap(), vec![51, 53, 55]);
    assert_eq!(exec.to_host(&out_g.slice(..len)).unwrap(), vec![61, 63, 65]);
}

#[test]
fn copy_where_accepts_u32_stencil() {
    let exec = exec();
    let values = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let ids = exec.to_device(&[1_u32, 2, 3, 4]).unwrap();
    let stencil = exec.to_device(&[0_u32, 0, 1, 1]).unwrap();
    let out_values = exec.to_device(&[0_u32; 4]).unwrap();
    let out_ids = exec.to_device(&[0_u32; 4]).unwrap();

    let len = copy_where(
        &exec,
        massively::Zip2(values.slice(..), ids.slice(..)),
        stencil.slice(..),
        massively::Zip2(out_values.slice_mut(..), out_ids.slice_mut(..)),
    )
    .unwrap();
    assert_eq!(
        exec.to_host(&out_values.slice(..len)).unwrap(),
        vec![30, 40]
    );
    assert_eq!(exec.to_host(&out_ids.slice(..len)).unwrap(), vec![3, 4]);
}

#[test]
fn copy_where_returns_empty_when_no_flags_are_selected() {
    let exec = exec();
    let values = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let stencil = exec.to_device(&[0_u32, 0, 0]).unwrap();
    let selected = exec.to_device(&[0_u32; 3]).unwrap();

    let len = copy_where(
        &exec,
        massively::Zip1(values.slice(..)),
        stencil.slice(..),
        massively::Zip1(selected.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(
        exec.to_host(&selected.slice(..len)).unwrap(),
        Vec::<u32>::new()
    );
}

#[test]
fn copy_where_keeps_all_values_when_all_flags_are_selected() {
    let exec = exec();
    let values = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let stencil = exec.to_device(&[1_u32, 1, 1]).unwrap();
    let selected = exec.to_device(&[0_u32; 3]).unwrap();

    let len = copy_where(
        &exec,
        massively::Zip1(values.slice(..)),
        stencil.slice(..),
        massively::Zip1(selected.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(
        exec.to_host(&selected.slice(..len)).unwrap(),
        vec![10, 20, 30]
    );
}

#[test]
fn copy_where_accepts_lazy_constant_stencil() {
    let exec = exec();
    let values = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let selected = exec.to_device(&[0_u32; 3]).unwrap();

    let len = copy_where(
        &exec,
        massively::Zip1(values.slice(..)),
        massively::lazy::constant(1_u32).take(3),
        massively::Zip1(selected.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(len, 3);
    assert_eq!(exec.to_host(&selected).unwrap(), vec![10, 20, 30]);
}

#[test]
fn copy_where_accepts_lazy_counting_stencil() {
    let exec = exec();
    let values = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let selected = exec.to_device(&[0_u32; 4]).unwrap();

    let len = copy_where(
        &exec,
        massively::Zip1(values.slice(..)),
        massively::lazy::counting(0).take(4),
        massively::Zip1(selected.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(len, 3);
    assert_eq!(
        exec.to_host(&selected.slice(..len)).unwrap(),
        vec![20, 30, 40]
    );
}
