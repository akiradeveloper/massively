use crate::common::*;

#[test]
fn scatter_if_accepts_soa12_values() {
    let policy = policy();
    let a = policy.to_device(&[1.0_f32, 0.0, 3.0]).unwrap();
    let b = policy.to_device(&[10_u32, 0, 30]).unwrap();
    let indices = policy.to_device(&[2_u32, 1, 0]).unwrap();
    let stencil = policy.to_device(&[1_u32, 0, 1]).unwrap();
    let output = scatter_if(
        (&a, &b),
        (&indices,),
        3,
        (0.0_f32, 0_u32),
        (&stencil,),
        NonZero,
    )
    .unwrap();
    let (a, b) = output;
    assert_eq!(a.to_vec().unwrap(), vec![3.0, 0.0, 1.0]);
    assert_eq!(b.to_vec().unwrap(), vec![30, 0, 10]);
}

#[test]
fn scatter_if_accepts_tuple_stencil() {
    let policy = policy();
    let a = policy.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let b = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let indices = policy.to_device(&[3_u32, 2, 1, 0]).unwrap();
    let stencil_a = policy.to_device(&[0.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let stencil_b = policy.to_device(&[1_u32, 0, 1, 1]).unwrap();

    let output = scatter_if(
        (&a, &b),
        (&indices,),
        4,
        (0_u32, 0.0_f32),
        (&stencil_a, &stencil_b),
        MixedStencilKeep,
    )
    .unwrap();
    let (a, b) = output;
    assert_eq!(a.to_vec().unwrap(), vec![40, 30, 0, 0]);
    assert_eq!(b.to_vec().unwrap(), vec![4.0, 3.0, 0.0, 0.0]);
}
