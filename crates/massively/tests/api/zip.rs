use crate::common::*;

#[test]
fn tuple_flattens_single_column_inputs() {
    let exec = exec();
    let left = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let right = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let indices = exec.to_device(&[0_u32, 1, 2]).unwrap();

    let out_left = exec.to_device(&[0.0_f32; 3]).unwrap();
    let out_right = exec.to_device(&[0_u32; 3]).unwrap();
    gather(
        &exec,
        massively::SoA2(left.slice(..), right.slice(..)),
        indices.slice(..),
        massively::SoA2(out_left.slice_mut(..), out_right.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_left).unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(exec.to_host(&out_right).unwrap(), vec![10, 20, 30]);
}

#[test]
fn tuple_materializes_heterogeneous_columns() {
    let exec = exec();
    let values = exec.to_device(&[1.5_f32, 2.5, 3.5]).unwrap();
    let ids = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let indices = exec.to_device(&[0_u32, 1, 2]).unwrap();

    let out_values = exec.to_device(&[0.0_f32; 3]).unwrap();
    let out_ids = exec.to_device(&[0_u32; 3]).unwrap();
    gather(
        &exec,
        massively::SoA2(values.slice(..), ids.slice(..)),
        indices.slice(..),
        massively::SoA2(out_values.slice_mut(..), out_ids.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_values).unwrap(), vec![1.5, 2.5, 3.5]);
    assert_eq!(exec.to_host(&out_ids).unwrap(), vec![10, 20, 30]);
}

#[test]
fn tuple_gather_accepts_borrowed_heterogeneous_columns() {
    let exec = exec();
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let ids = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let indices = exec.to_device(&[3_u32, 1, 0]).unwrap();

    let out_values = exec.to_device(&[0.0_f32; 3]).unwrap();
    let out_ids = exec.to_device(&[0_u32; 3]).unwrap();
    gather(
        &exec,
        massively::SoA2(values.slice(..), ids.slice(..)),
        indices.slice(..),
        massively::SoA2(out_values.slice_mut(..), out_ids.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_values).unwrap(), vec![4.0, 2.0, 1.0]);
    assert_eq!(exec.to_host(&out_ids).unwrap(), vec![40, 20, 10]);
}

#[test]
fn tuple_gather_accepts_heterogeneous_columns() {
    let exec = exec();
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let ids = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let indices = exec.to_device(&[3_u32, 1, 0]).unwrap();

    let out_values = exec.to_device(&[0.0_f32; 3]).unwrap();
    let out_ids = exec.to_device(&[0_u32; 3]).unwrap();
    gather(
        &exec,
        massively::SoA2(values.slice(..), ids.slice(..)),
        indices.slice(..),
        massively::SoA2(out_values.slice_mut(..), out_ids.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_values).unwrap(), vec![4.0, 2.0, 1.0]);
    assert_eq!(exec.to_host(&out_ids).unwrap(), vec![40, 20, 10]);
}

#[test]
fn tuple_concatenates_borrowed_columns() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20]).unwrap();
    let c = exec.to_device(&[100.0_f32, 200.0]).unwrap();

    let indices = exec.to_device(&[0_u32, 1]).unwrap();
    let out_a = exec.to_device(&[0.0_f32; 2]).unwrap();
    let out_b = exec.to_device(&[0_u32; 2]).unwrap();
    let out_c = exec.to_device(&[0.0_f32; 2]).unwrap();
    gather(
        &exec,
        massively::SoA3(a.slice(..), b.slice(..), c.slice(..)),
        indices.slice(..),
        massively::SoA3(
            out_a.slice_mut(..),
            out_b.slice_mut(..),
            out_c.slice_mut(..),
        ),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_a).unwrap(), vec![1.0, 2.0]);
    assert_eq!(exec.to_host(&out_b).unwrap(), vec![10, 20]);
    assert_eq!(exec.to_host(&out_c).unwrap(), vec![100.0, 200.0]);
}

#[test]
fn tuple_concatenates_column_and_columns() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20]).unwrap();
    let c = exec.to_device(&[100.0_f32, 200.0]).unwrap();

    let indices = exec.to_device(&[0_u32, 1]).unwrap();
    let out_a = exec.to_device(&[0.0_f32; 2]).unwrap();
    let out_b = exec.to_device(&[0_u32; 2]).unwrap();
    let out_c = exec.to_device(&[0.0_f32; 2]).unwrap();
    gather(
        &exec,
        massively::SoA3(a.slice(..), b.slice(..), c.slice(..)),
        indices.slice(..),
        massively::SoA3(
            out_a.slice_mut(..),
            out_b.slice_mut(..),
            out_c.slice_mut(..),
        ),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_a).unwrap(), vec![1.0, 2.0]);
    assert_eq!(exec.to_host(&out_b).unwrap(), vec![10, 20]);
    assert_eq!(exec.to_host(&out_c).unwrap(), vec![100.0, 200.0]);
}

#[test]
fn soa7_gather_and_scatter_move_all_columns() {
    let exec = exec();
    let a = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let b = exec.to_device(&[11_u32, 21, 31, 41]).unwrap();
    let c = exec.to_device(&[12_u32, 22, 32, 42]).unwrap();
    let d = exec.to_device(&[13_u32, 23, 33, 43]).unwrap();
    let e = exec.to_device(&[14_u32, 24, 34, 44]).unwrap();
    let f = exec.to_device(&[15_u32, 25, 35, 45]).unwrap();
    let g = exec.to_device(&[16_u32, 26, 36, 46]).unwrap();
    let indices = exec.to_device(&[3_u32, 1, 0]).unwrap();

    let out_a = exec.to_device(&[0_u32; 3]).unwrap();
    let out_b = exec.to_device(&[0_u32; 3]).unwrap();
    let out_c = exec.to_device(&[0_u32; 3]).unwrap();
    let out_d = exec.to_device(&[0_u32; 3]).unwrap();
    let out_e = exec.to_device(&[0_u32; 3]).unwrap();
    let out_f = exec.to_device(&[0_u32; 3]).unwrap();
    let out_g = exec.to_device(&[0_u32; 3]).unwrap();
    gather(
        &exec,
        massively::SoA7(
            a.slice(..),
            b.slice(..),
            c.slice(..),
            d.slice(..),
            e.slice(..),
            f.slice(..),
            g.slice(..),
        ),
        indices.slice(..),
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

    assert_eq!(exec.to_host(&out_a).unwrap(), vec![40, 20, 10]);
    assert_eq!(exec.to_host(&out_b).unwrap(), vec![41, 21, 11]);
    assert_eq!(exec.to_host(&out_c).unwrap(), vec![42, 22, 12]);
    assert_eq!(exec.to_host(&out_d).unwrap(), vec![43, 23, 13]);
    assert_eq!(exec.to_host(&out_e).unwrap(), vec![44, 24, 14]);
    assert_eq!(exec.to_host(&out_f).unwrap(), vec![45, 25, 15]);
    assert_eq!(exec.to_host(&out_g).unwrap(), vec![46, 26, 16]);

    let scatter_a = exec.to_device(&[0_u32; 4]).unwrap();
    let scatter_b = exec.to_device(&[0_u32; 4]).unwrap();
    let scatter_c = exec.to_device(&[0_u32; 4]).unwrap();
    let scatter_d = exec.to_device(&[0_u32; 4]).unwrap();
    let scatter_e = exec.to_device(&[0_u32; 4]).unwrap();
    let scatter_f = exec.to_device(&[0_u32; 4]).unwrap();
    let scatter_g = exec.to_device(&[0_u32; 4]).unwrap();
    scatter(
        &exec,
        massively::SoA7(
            out_a.slice(..),
            out_b.slice(..),
            out_c.slice(..),
            out_d.slice(..),
            out_e.slice(..),
            out_f.slice(..),
            out_g.slice(..),
        ),
        indices.slice(..),
        massively::SoA7(
            scatter_a.slice_mut(..),
            scatter_b.slice_mut(..),
            scatter_c.slice_mut(..),
            scatter_d.slice_mut(..),
            scatter_e.slice_mut(..),
            scatter_f.slice_mut(..),
            scatter_g.slice_mut(..),
        ),
    )
    .unwrap();

    assert_eq!(exec.to_host(&scatter_a).unwrap(), vec![10, 20, 0, 40]);
    assert_eq!(exec.to_host(&scatter_b).unwrap(), vec![11, 21, 0, 41]);
    assert_eq!(exec.to_host(&scatter_c).unwrap(), vec![12, 22, 0, 42]);
    assert_eq!(exec.to_host(&scatter_d).unwrap(), vec![13, 23, 0, 43]);
    assert_eq!(exec.to_host(&scatter_e).unwrap(), vec![14, 24, 0, 44]);
    assert_eq!(exec.to_host(&scatter_f).unwrap(), vec![15, 25, 0, 45]);
    assert_eq!(exec.to_host(&scatter_g).unwrap(), vec![16, 26, 0, 46]);
}

#[test]
fn soa7_gather_where_and_scatter_where_move_selected_columns() {
    let exec = exec();
    let a = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let b = exec.to_device(&[11_u32, 21, 31, 41]).unwrap();
    let c = exec.to_device(&[12_u32, 22, 32, 42]).unwrap();
    let d = exec.to_device(&[13_u32, 23, 33, 43]).unwrap();
    let e = exec.to_device(&[14_u32, 24, 34, 44]).unwrap();
    let f = exec.to_device(&[15_u32, 25, 35, 45]).unwrap();
    let g = exec.to_device(&[16_u32, 26, 36, 46]).unwrap();
    let indices = exec.to_device(&[3_u32, 1, 0]).unwrap();
    let stencil = exec.to_device(&[1_u32, 0, 1]).unwrap();

    let out_a = exec.to_device(&[100_u32; 3]).unwrap();
    let out_b = exec.to_device(&[100_u32; 3]).unwrap();
    let out_c = exec.to_device(&[100_u32; 3]).unwrap();
    let out_d = exec.to_device(&[100_u32; 3]).unwrap();
    let out_e = exec.to_device(&[100_u32; 3]).unwrap();
    let out_f = exec.to_device(&[100_u32; 3]).unwrap();
    let out_g = exec.to_device(&[100_u32; 3]).unwrap();
    gather_where(
        &exec,
        massively::SoA7(
            a.slice(..),
            b.slice(..),
            c.slice(..),
            d.slice(..),
            e.slice(..),
            f.slice(..),
            g.slice(..),
        ),
        indices.slice(..),
        stencil.slice(..),
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

    assert_eq!(exec.to_host(&out_a).unwrap(), vec![40, 100, 10]);
    assert_eq!(exec.to_host(&out_b).unwrap(), vec![41, 100, 11]);
    assert_eq!(exec.to_host(&out_c).unwrap(), vec![42, 100, 12]);
    assert_eq!(exec.to_host(&out_d).unwrap(), vec![43, 100, 13]);
    assert_eq!(exec.to_host(&out_e).unwrap(), vec![44, 100, 14]);
    assert_eq!(exec.to_host(&out_f).unwrap(), vec![45, 100, 15]);
    assert_eq!(exec.to_host(&out_g).unwrap(), vec![46, 100, 16]);

    let scatter_a = exec.to_device(&[0_u32; 4]).unwrap();
    let scatter_b = exec.to_device(&[0_u32; 4]).unwrap();
    let scatter_c = exec.to_device(&[0_u32; 4]).unwrap();
    let scatter_d = exec.to_device(&[0_u32; 4]).unwrap();
    let scatter_e = exec.to_device(&[0_u32; 4]).unwrap();
    let scatter_f = exec.to_device(&[0_u32; 4]).unwrap();
    let scatter_g = exec.to_device(&[0_u32; 4]).unwrap();
    scatter_where(
        &exec,
        massively::SoA7(
            out_a.slice(..),
            out_b.slice(..),
            out_c.slice(..),
            out_d.slice(..),
            out_e.slice(..),
            out_f.slice(..),
            out_g.slice(..),
        ),
        indices.slice(..),
        stencil.slice(..),
        massively::SoA7(
            scatter_a.slice_mut(..),
            scatter_b.slice_mut(..),
            scatter_c.slice_mut(..),
            scatter_d.slice_mut(..),
            scatter_e.slice_mut(..),
            scatter_f.slice_mut(..),
            scatter_g.slice_mut(..),
        ),
    )
    .unwrap();

    assert_eq!(exec.to_host(&scatter_a).unwrap(), vec![10, 0, 0, 40]);
    assert_eq!(exec.to_host(&scatter_b).unwrap(), vec![11, 0, 0, 41]);
    assert_eq!(exec.to_host(&scatter_c).unwrap(), vec![12, 0, 0, 42]);
    assert_eq!(exec.to_host(&scatter_d).unwrap(), vec![13, 0, 0, 43]);
    assert_eq!(exec.to_host(&scatter_e).unwrap(), vec![14, 0, 0, 44]);
    assert_eq!(exec.to_host(&scatter_f).unwrap(), vec![15, 0, 0, 45]);
    assert_eq!(exec.to_host(&scatter_g).unwrap(), vec![16, 0, 0, 46]);
}

#[test]
fn soa7_reverse_returns_all_columns() {
    let exec = exec();
    let a = exec.to_device(&[1_u32, 2, 3]).unwrap();
    let b = exec.to_device(&[11_u32, 12, 13]).unwrap();
    let c = exec.to_device(&[21_u32, 22, 23]).unwrap();
    let d = exec.to_device(&[31_u32, 32, 33]).unwrap();
    let e = exec.to_device(&[41_u32, 42, 43]).unwrap();
    let f = exec.to_device(&[51_u32, 52, 53]).unwrap();
    let g = exec.to_device(&[61_u32, 62, 63]).unwrap();

    let out_a = exec.to_device(&[0_u32; 3]).unwrap();
    let out_b = exec.to_device(&[0_u32; 3]).unwrap();
    let out_c = exec.to_device(&[0_u32; 3]).unwrap();
    let out_d = exec.to_device(&[0_u32; 3]).unwrap();
    let out_e = exec.to_device(&[0_u32; 3]).unwrap();
    let out_f = exec.to_device(&[0_u32; 3]).unwrap();
    let out_g = exec.to_device(&[0_u32; 3]).unwrap();
    reverse(
        &exec,
        massively::SoA7(
            a.slice(..),
            b.slice(..),
            c.slice(..),
            d.slice(..),
            e.slice(..),
            f.slice(..),
            g.slice(..),
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

    assert_eq!(exec.to_host(&out_a).unwrap(), vec![3, 2, 1]);
    assert_eq!(exec.to_host(&out_b).unwrap(), vec![13, 12, 11]);
    assert_eq!(exec.to_host(&out_c).unwrap(), vec![23, 22, 21]);
    assert_eq!(exec.to_host(&out_d).unwrap(), vec![33, 32, 31]);
    assert_eq!(exec.to_host(&out_e).unwrap(), vec![43, 42, 41]);
    assert_eq!(exec.to_host(&out_f).unwrap(), vec![53, 52, 51]);
    assert_eq!(exec.to_host(&out_g).unwrap(), vec![63, 62, 61]);
}
