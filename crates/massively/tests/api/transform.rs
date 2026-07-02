use crate::common::*;

#[test]
fn transform_zip_output_returns_storage() {
    let exec = exec();
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let tags = exec.to_device(&[10_u32, 20, 30]).unwrap();

    let out_values = exec.to_device(&[0.0_f32; 3]).unwrap();
    let out_tags = exec.to_device(&[0_u32; 3]).unwrap();
    transform(
        &exec,
        massively::SoA2(values.slice(..), tags.slice(..)),
        PairMixedSplit,
        (),
        massively::SoA2(out_values.slice_mut(..), out_tags.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_values).unwrap(), vec![11.0, 12.0, 13.0]);
    assert_eq!(exec.to_host(&out_tags).unwrap(), vec![11, 21, 31]);
}

#[test]
fn transform_returns_device_storage() {
    let exec = exec();
    let left = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let right = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let values = exec.to_device(&[0.0_f32; 3]).unwrap();
    let tags = exec.to_device(&[0_u32; 3]).unwrap();
    transform(
        &exec,
        massively::SoA2(left.slice(..), right.slice(..)),
        PairMixedSplit,
        (),
        massively::SoA2(values.slice_mut(..), tags.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&values).unwrap(), vec![11.0, 12.0, 13.0]);
    assert_eq!(exec.to_host(&tags).unwrap(), vec![11, 21, 31]);
}

#[test]
fn transform_tuple_output_maps_to_mitem_storage() {
    let exec = exec();
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let tags = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let bias = exec.to_device(&[100.0_f32, 200.0, 300.0]).unwrap();

    let values_out = exec.to_device(&[0.0_f32; 3]).unwrap();
    let flags = exec.to_device(&[0_u32; 3]).unwrap();
    let bias_out = exec.to_device(&[0.0_f32; 3]).unwrap();
    transform(
        &exec,
        massively::SoA3(values.slice(..), tags.slice(..), bias.slice(..)),
        Tuple3MixedSplit,
        (),
        massively::SoA3(
            values_out.slice_mut(..),
            flags.slice_mut(..),
            bias_out.slice_mut(..),
        ),
    )
    .unwrap();
    assert_eq!(
        exec.to_host(&values_out).unwrap(),
        vec![101.0, 202.0, 303.0]
    );
    assert_eq!(exec.to_host(&flags).unwrap(), vec![11, 21, 31]);
    assert_eq!(exec.to_host(&bias_out).unwrap(), vec![101.0, 202.0, 303.0]);
}

#[test]
fn tuple1_transform_returns_soa1_storage() {
    let exec = exec();
    let input = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();

    let output = exec.to_device(&[0.0_f32; 3]).unwrap();
    transform(
        &exec,
        massively::SoA1(input.slice(..)),
        Double,
        (),
        massively::SoA1(output.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![2.0, 4.0, 6.0]);
}

#[test]
fn transform_can_write_in_place_for_single_column() {
    let exec = exec();
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();

    transform(
        &exec,
        massively::SoA1(values.slice(..)),
        Double,
        (),
        massively::SoA1(values.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&values).unwrap(), vec![2.0, 4.0, 6.0]);
}

#[test]
fn transform_can_write_in_place_for_multi_column() {
    let exec = exec();
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let tags = exec.to_device(&[10_u32, 20, 30]).unwrap();

    transform(
        &exec,
        massively::SoA2(values.slice(..), tags.slice(..)),
        PairMixedSplit,
        (),
        massively::SoA2(values.slice_mut(..), tags.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&values).unwrap(), vec![11.0, 12.0, 13.0]);
    assert_eq!(exec.to_host(&tags).unwrap(), vec![11, 21, 31]);
}

#[test]
fn unary_transform_accepts_seven_tuple_output() {
    let exec = exec();
    let input = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let a = exec.to_device(&[0_u32; 3]).unwrap();
    let b = exec.to_device(&[0.0_f32; 3]).unwrap();
    let c = exec.to_device(&[0_u32; 3]).unwrap();
    let d = exec.to_device(&[0.0_f32; 3]).unwrap();
    let e = exec.to_device(&[0_u32; 3]).unwrap();
    let f = exec.to_device(&[0.0_f32; 3]).unwrap();
    let g = exec.to_device(&[0_u32; 3]).unwrap();

    transform(
        &exec,
        massively::SoA1(input.slice(..)),
        ScalarToTuple7Mixed,
        (),
        massively::SoA7(
            a.slice_mut(..),
            b.slice_mut(..),
            c.slice_mut(..),
            d.slice_mut(..),
            e.slice_mut(..),
            f.slice_mut(..),
            g.slice_mut(..),
        ),
    )
    .unwrap();

    assert_eq!(exec.to_host(&a).unwrap(), vec![11, 21, 31]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![12.0, 22.0, 32.0]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![13, 23, 33]);
    assert_eq!(exec.to_host(&d).unwrap(), vec![14.0, 24.0, 34.0]);
    assert_eq!(exec.to_host(&e).unwrap(), vec![15, 25, 35]);
    assert_eq!(exec.to_host(&f).unwrap(), vec![16.0, 26.0, 36.0]);
    assert_eq!(exec.to_host(&g).unwrap(), vec![17, 27, 37]);
}

#[test]
fn transform_where_accepts_seven_tuple_output() {
    let exec = exec();
    let input = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let stencil = exec.to_device(&[1_u32, 0, 1]).unwrap();
    let a = exec.to_device(&[100_u32; 3]).unwrap();
    let b = exec.to_device(&[100.0_f32; 3]).unwrap();
    let c = exec.to_device(&[100_u32; 3]).unwrap();
    let d = exec.to_device(&[100.0_f32; 3]).unwrap();
    let e = exec.to_device(&[100_u32; 3]).unwrap();
    let f = exec.to_device(&[100.0_f32; 3]).unwrap();
    let g = exec.to_device(&[100_u32; 3]).unwrap();

    massively::transform_where(
        &exec,
        massively::SoA1(input.slice(..)),
        ScalarToTuple7Mixed,
        (),
        stencil.slice(..),
        massively::SoA7(
            a.slice_mut(..),
            b.slice_mut(..),
            c.slice_mut(..),
            d.slice_mut(..),
            e.slice_mut(..),
            f.slice_mut(..),
            g.slice_mut(..),
        ),
    )
    .unwrap();

    assert_eq!(exec.to_host(&a).unwrap(), vec![11, 100, 31]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![12.0, 100.0, 32.0]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![13, 100, 33]);
    assert_eq!(exec.to_host(&d).unwrap(), vec![14.0, 100.0, 34.0]);
    assert_eq!(exec.to_host(&e).unwrap(), vec![15, 100, 35]);
    assert_eq!(exec.to_host(&f).unwrap(), vec![16.0, 100.0, 36.0]);
    assert_eq!(exec.to_host(&g).unwrap(), vec![17, 100, 37]);
}

#[test]
fn transform_accepts_seven_tuple_input_and_output() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20]).unwrap();
    let c = exec.to_device(&[100.0_f32, 200.0]).unwrap();
    let d = exec.to_device(&[1000_u32, 2000]).unwrap();
    let e = exec.to_device(&[10000.0_f32, 20000.0]).unwrap();
    let f = exec.to_device(&[100000_u32, 200000]).unwrap();
    let g = exec.to_device(&[1000000.0_f32, 2000000.0]).unwrap();
    let out_a = exec.to_device(&[0.0_f32; 2]).unwrap();
    let out_b = exec.to_device(&[0_u32; 2]).unwrap();
    let out_c = exec.to_device(&[0.0_f32; 2]).unwrap();
    let out_d = exec.to_device(&[0_u32; 2]).unwrap();
    let out_e = exec.to_device(&[0.0_f32; 2]).unwrap();
    let out_f = exec.to_device(&[0_u32; 2]).unwrap();
    let out_g = exec.to_device(&[0.0_f32; 2]).unwrap();

    transform(
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
        TupleWideMixedSplit,
        (),
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

    assert_eq!(exec.to_host(&out_a).unwrap(), vec![2.0, 3.0]);
    assert_eq!(exec.to_host(&out_b).unwrap(), vec![12, 22]);
    assert_eq!(exec.to_host(&out_c).unwrap(), vec![103.0, 203.0]);
    assert_eq!(exec.to_host(&out_d).unwrap(), vec![1004, 2004]);
    assert_eq!(exec.to_host(&out_e).unwrap(), vec![10005.0, 20005.0]);
    assert_eq!(exec.to_host(&out_f).unwrap(), vec![100006, 200006]);
    assert_eq!(exec.to_host(&out_g).unwrap(), vec![1000007.0, 2000007.0]);
}

#[test]
fn transform_where_accepts_seven_tuple_input_and_output() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let c = exec.to_device(&[100.0_f32, 200.0, 300.0]).unwrap();
    let d = exec.to_device(&[1000_u32, 2000, 3000]).unwrap();
    let e = exec.to_device(&[10000.0_f32, 20000.0, 30000.0]).unwrap();
    let f = exec.to_device(&[100000_u32, 200000, 300000]).unwrap();
    let g = exec
        .to_device(&[1000000.0_f32, 2000000.0, 3000000.0])
        .unwrap();
    let stencil = exec.to_device(&[1_u32, 0, 1]).unwrap();
    let out_a = exec.to_device(&[0.0_f32; 3]).unwrap();
    let out_b = exec.to_device(&[0_u32; 3]).unwrap();
    let out_c = exec.to_device(&[0.0_f32; 3]).unwrap();
    let out_d = exec.to_device(&[0_u32; 3]).unwrap();
    let out_e = exec.to_device(&[0.0_f32; 3]).unwrap();
    let out_f = exec.to_device(&[0_u32; 3]).unwrap();
    let out_g = exec.to_device(&[0.0_f32; 3]).unwrap();

    massively::transform_where(
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
        TupleWideMixedSplit,
        (),
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

    assert_eq!(exec.to_host(&out_a).unwrap(), vec![2.0, 0.0, 4.0]);
    assert_eq!(exec.to_host(&out_b).unwrap(), vec![12, 0, 32]);
    assert_eq!(exec.to_host(&out_c).unwrap(), vec![103.0, 0.0, 303.0]);
    assert_eq!(exec.to_host(&out_d).unwrap(), vec![1004, 0, 3004]);
    assert_eq!(exec.to_host(&out_e).unwrap(), vec![10005.0, 0.0, 30005.0]);
    assert_eq!(exec.to_host(&out_f).unwrap(), vec![100006, 0, 300006]);
    assert_eq!(
        exec.to_host(&out_g).unwrap(),
        vec![1000007.0, 0.0, 3000007.0]
    );
}

#[test]
fn unary_transform_accepts_wide_tuple_outputs() {
    let exec = exec();
    let input = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let a = exec.to_device(&[0.0_f32; 3]).unwrap();
    let b = exec.to_device(&[0_u32; 3]).unwrap();
    let c = exec.to_device(&[0.0_f32; 3]).unwrap();
    let d = exec.to_device(&[0_u32; 3]).unwrap();
    let e = exec.to_device(&[0.0_f32; 3]).unwrap();

    transform(
        &exec,
        massively::SoA1(input.slice(..)),
        ScalarToTuple5Mixed,
        (),
        massively::SoA5(
            a.slice_mut(..),
            b.slice_mut(..),
            c.slice_mut(..),
            d.slice_mut(..),
            e.slice_mut(..),
        ),
    )
    .unwrap();

    assert_eq!(exec.to_host(&a).unwrap(), vec![2.0, 3.0, 4.0]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![3, 4, 5]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![4.0, 5.0, 6.0]);
    assert_eq!(exec.to_host(&d).unwrap(), vec![5, 6, 7]);
    assert_eq!(exec.to_host(&e).unwrap(), vec![6.0, 7.0, 8.0]);
}

#[test]
fn tuple_transform_uses_flat_soa_input() {
    let exec = exec();
    let lhs = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let rhs = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let bias = exec.to_device(&[100.0_f32, 200.0, 300.0]).unwrap();

    let values = exec.to_device(&[0.0_f32; 3]).unwrap();
    let tags = exec.to_device(&[0_u32; 3]).unwrap();
    let adjusted_bias = exec.to_device(&[0.0_f32; 3]).unwrap();
    transform(
        &exec,
        massively::SoA3(lhs.slice(..), rhs.slice(..), bias.slice(..)),
        Tuple3MixedSplit,
        (),
        massively::SoA3(
            values.slice_mut(..),
            tags.slice_mut(..),
            adjusted_bias.slice_mut(..),
        ),
    )
    .unwrap();

    assert_eq!(exec.to_host(&values).unwrap(), vec![101.0, 202.0, 303.0]);
    assert_eq!(exec.to_host(&tags).unwrap(), vec![11, 21, 31]);
    assert_eq!(
        exec.to_host(&adjusted_bias).unwrap(),
        vec![101.0, 202.0, 303.0]
    );
}

#[test]
fn transform_accepts_heterogeneous_tuple_inputs() {
    let exec = exec();
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let tags = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let bias = exec.to_device(&[100.0_f32, 200.0, 300.0]).unwrap();
    let pair_values = exec.to_device(&[0.0_f32; 3]).unwrap();
    let pair_tags = exec.to_device(&[0_u32; 3]).unwrap();
    transform(
        &exec,
        massively::SoA2(values.slice(..), tags.slice(..)),
        PairMixedSplit,
        (),
        massively::SoA2(pair_values.slice_mut(..), pair_tags.slice_mut(..)),
    )
    .unwrap();
    assert_eq!(exec.to_host(&pair_values).unwrap(), vec![11.0, 12.0, 13.0]);
    assert_eq!(exec.to_host(&pair_tags).unwrap(), vec![11, 21, 31]);

    let tuple_values = exec.to_device(&[0.0_f32; 3]).unwrap();
    let tuple_tags = exec.to_device(&[0_u32; 3]).unwrap();
    let tuple_bias = exec.to_device(&[0.0_f32; 3]).unwrap();
    transform(
        &exec,
        massively::SoA3(values.slice(..), tags.slice(..), bias.slice(..)),
        Tuple3MixedSplit,
        (),
        massively::SoA3(
            tuple_values.slice_mut(..),
            tuple_tags.slice_mut(..),
            tuple_bias.slice_mut(..),
        ),
    )
    .unwrap();
    assert_eq!(
        exec.to_host(&tuple_values).unwrap(),
        vec![101.0, 202.0, 303.0]
    );
    assert_eq!(exec.to_host(&tuple_tags).unwrap(), vec![11, 21, 31]);
    assert_eq!(
        exec.to_host(&tuple_bias).unwrap(),
        vec![101.0, 202.0, 303.0]
    );
}

#[test]
fn transform_accepts_soa4_heterogeneous_inputs_and_checks_every_column() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let c = exec.to_device(&[100.0_f32, 200.0, 300.0]).unwrap();
    let d = exec.to_device(&[1000_u32, 2000, 3000]).unwrap();
    let out_a = exec.to_device(&[0.0_f32; 3]).unwrap();
    let out_b = exec.to_device(&[0_u32; 3]).unwrap();
    let out_c = exec.to_device(&[0.0_f32; 3]).unwrap();
    let out_d = exec.to_device(&[0_u32; 3]).unwrap();

    transform(
        &exec,
        massively::SoA4(a.slice(..), b.slice(..), c.slice(..), d.slice(..)),
        Tuple4MixedSplit,
        (),
        massively::SoA4(
            out_a.slice_mut(..),
            out_b.slice_mut(..),
            out_c.slice_mut(..),
            out_d.slice_mut(..),
        ),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_a).unwrap(), vec![101.0, 202.0, 303.0]);
    assert_eq!(exec.to_host(&out_b).unwrap(), vec![12, 22, 32]);
    assert_eq!(exec.to_host(&out_c).unwrap(), vec![101.0, 202.0, 303.0]);
    assert_eq!(exec.to_host(&out_d).unwrap(), vec![1004, 2004, 3004]);
}

#[test]
fn transform_accepts_mismatched_input_and_output_tuple_widths() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20]).unwrap();
    let c = exec.to_device(&[100.0_f32, 200.0]).unwrap();
    let d = exec.to_device(&[1000_u32, 2000]).unwrap();
    let e = exec.to_device(&[10000.0_f32, 20000.0]).unwrap();

    let x = exec.to_device(&[0.0_f32; 2]).unwrap();
    let y = exec.to_device(&[0_u32; 2]).unwrap();
    let z = exec.to_device(&[0.0_f32; 2]).unwrap();
    transform(
        &exec,
        massively::SoA5(
            a.slice(..),
            b.slice(..),
            c.slice(..),
            d.slice(..),
            e.slice(..),
        ),
        Tuple5To3MixedSplit,
        (),
        massively::SoA3(x.slice_mut(..), y.slice_mut(..), z.slice_mut(..)),
    )
    .unwrap();
    assert_eq!(exec.to_host(&x).unwrap(), vec![10101.0, 20202.0]);
    assert_eq!(exec.to_host(&y).unwrap(), vec![1010, 2020]);
    assert_eq!(exec.to_host(&z).unwrap(), vec![9999.0, 19998.0]);

    let x = exec.to_device(&[0.0_f32; 2]).unwrap();
    let y = exec.to_device(&[0_u32; 2]).unwrap();
    let z = exec.to_device(&[0.0_f32; 2]).unwrap();
    let w = exec.to_device(&[0_u32; 2]).unwrap();
    let v = exec.to_device(&[0.0_f32; 2]).unwrap();
    transform(
        &exec,
        massively::SoA3(a.slice(..), b.slice(..), c.slice(..)),
        Tuple3To5MixedSplit,
        (),
        massively::SoA5(
            x.slice_mut(..),
            y.slice_mut(..),
            z.slice_mut(..),
            w.slice_mut(..),
            v.slice_mut(..),
        ),
    )
    .unwrap();
    assert_eq!(exec.to_host(&x).unwrap(), vec![101.0, 202.0]);
    assert_eq!(exec.to_host(&y).unwrap(), vec![20, 30]);
    assert_eq!(exec.to_host(&z).unwrap(), vec![99.0, 198.0]);
    assert_eq!(exec.to_host(&w).unwrap(), vec![30, 40]);
    assert_eq!(exec.to_host(&v).unwrap(), vec![100.0, 400.0]);
}

#[test]
fn transform_accepts_soa5_to_soa7_heterogeneous_tuple_outputs() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32]).unwrap();
    let b = exec.to_device(&[10_u32]).unwrap();
    let c = exec.to_device(&[100.0_f32]).unwrap();
    let d = exec.to_device(&[1000_u32]).unwrap();
    let e = exec.to_device(&[10000.0_f32]).unwrap();
    let f = exec.to_device(&[100000_u32]).unwrap();
    let g = exec.to_device(&[1000000.0_f32]).unwrap();

    let a5 = exec.to_device(&[0.0_f32]).unwrap();
    let b5 = exec.to_device(&[0_u32]).unwrap();
    let c5 = exec.to_device(&[0.0_f32]).unwrap();
    let d5 = exec.to_device(&[0_u32]).unwrap();
    let e5 = exec.to_device(&[0.0_f32]).unwrap();
    transform(
        &exec,
        massively::SoA5(
            a.slice(..),
            b.slice(..),
            c.slice(..),
            d.slice(..),
            e.slice(..),
        ),
        TupleWideMixedSplit,
        (),
        massively::SoA5(
            a5.slice_mut(..),
            b5.slice_mut(..),
            c5.slice_mut(..),
            d5.slice_mut(..),
            e5.slice_mut(..),
        ),
    )
    .unwrap();
    assert_eq!(exec.to_host(&a5).unwrap(), vec![2.0]);
    assert_eq!(exec.to_host(&b5).unwrap(), vec![12]);
    assert_eq!(exec.to_host(&c5).unwrap(), vec![103.0]);
    assert_eq!(exec.to_host(&d5).unwrap(), vec![1004]);
    assert_eq!(exec.to_host(&e5).unwrap(), vec![10005.0]);

    let a6 = exec.to_device(&[0.0_f32]).unwrap();
    let b6 = exec.to_device(&[0_u32]).unwrap();
    let c6 = exec.to_device(&[0.0_f32]).unwrap();
    let d6 = exec.to_device(&[0_u32]).unwrap();
    let e6 = exec.to_device(&[0.0_f32]).unwrap();
    let f6 = exec.to_device(&[0_u32]).unwrap();
    transform(
        &exec,
        massively::SoA6(
            a.slice(..),
            b.slice(..),
            c.slice(..),
            d.slice(..),
            e.slice(..),
            f.slice(..),
        ),
        TupleWideMixedSplit,
        (),
        massively::SoA6(
            a6.slice_mut(..),
            b6.slice_mut(..),
            c6.slice_mut(..),
            d6.slice_mut(..),
            e6.slice_mut(..),
            f6.slice_mut(..),
        ),
    )
    .unwrap();
    assert_eq!(exec.to_host(&a6).unwrap(), vec![2.0]);
    assert_eq!(exec.to_host(&b6).unwrap(), vec![12]);
    assert_eq!(exec.to_host(&c6).unwrap(), vec![103.0]);
    assert_eq!(exec.to_host(&d6).unwrap(), vec![1004]);
    assert_eq!(exec.to_host(&e6).unwrap(), vec![10005.0]);
    assert_eq!(exec.to_host(&f6).unwrap(), vec![100006]);

    let a7 = exec.to_device(&[0.0_f32]).unwrap();
    let b7 = exec.to_device(&[0_u32]).unwrap();
    let c7 = exec.to_device(&[0.0_f32]).unwrap();
    let d7 = exec.to_device(&[0_u32]).unwrap();
    let e7 = exec.to_device(&[0.0_f32]).unwrap();
    let f7 = exec.to_device(&[0_u32]).unwrap();
    let g7 = exec.to_device(&[0.0_f32]).unwrap();
    transform(
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
        TupleWideMixedSplit,
        (),
        massively::SoA7(
            a7.slice_mut(..),
            b7.slice_mut(..),
            c7.slice_mut(..),
            d7.slice_mut(..),
            e7.slice_mut(..),
            f7.slice_mut(..),
            g7.slice_mut(..),
        ),
    )
    .unwrap();
    assert_eq!(exec.to_host(&a7).unwrap(), vec![2.0]);
    assert_eq!(exec.to_host(&b7).unwrap(), vec![12]);
    assert_eq!(exec.to_host(&c7).unwrap(), vec![103.0]);
    assert_eq!(exec.to_host(&d7).unwrap(), vec![1004]);
    assert_eq!(exec.to_host(&e7).unwrap(), vec![10005.0]);
    assert_eq!(exec.to_host(&f7).unwrap(), vec![100006]);
    assert_eq!(exec.to_host(&g7).unwrap(), vec![1000007.0]);
}

#[test]
fn transform_zip_flattens_soa1_columns() {
    let exec = exec();
    let left = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let right = exec.to_device(&[10_u32, 20, 30]).unwrap();

    let values = exec.to_device(&[0.0_f32; 3]).unwrap();
    let tags = exec.to_device(&[0_u32; 3]).unwrap();
    transform(
        &exec,
        massively::SoA2(left.slice(..), right.slice(..)),
        PairMixedSplit,
        (),
        massively::SoA2(values.slice_mut(..), tags.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&values).unwrap(), vec![11.0, 12.0, 13.0]);
    assert_eq!(exec.to_host(&tags).unwrap(), vec![11, 21, 31]);
}
