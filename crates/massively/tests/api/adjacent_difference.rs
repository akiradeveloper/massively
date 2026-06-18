use crate::common::*;

#[test]
fn adjacent_difference_accepts_soa12() {
    let policy = policy();
    let a = policy.to_device(&[1.0_f32, 3.0, 6.0, 10.0]).unwrap();
    let b = policy.to_device(&[10_u32, 30, 60, 100]).unwrap();
    let c = policy.to_device(&[2.0_f32, 5.0, 9.0, 14.0]).unwrap();
    let d = policy.to_device(&[20_u32, 50, 90, 140]).unwrap();
    let e = policy.to_device(&[4.0_f32, 9.0, 15.0, 22.0]).unwrap();
    let f = policy.to_device(&[40_u32, 90, 150, 220]).unwrap();
    let g = policy.to_device(&[7.0_f32, 11.0, 18.0, 26.0]).unwrap();
    let h = policy.to_device(&[70_u32, 110, 180, 260]).unwrap();
    let i = policy.to_device(&[8.0_f32, 13.0, 21.0, 34.0]).unwrap();
    let j = policy.to_device(&[80_u32, 130, 210, 340]).unwrap();
    let k = policy.to_device(&[12.0_f32, 19.0, 31.0, 50.0]).unwrap();
    let l = policy.to_device(&[120_u32, 190, 310, 500]).unwrap();

    let output =
        adjacent_difference(zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l), Sum).unwrap();
    let (a_out, b_out, c_out, d_out, e_out, f_out, g_out, h_out, i_out, j_out, k_out, l_out) =
        output;

    assert_eq!(a_out.to_vec().unwrap(), vec![1.0, 4.0, 9.0, 16.0]);
    assert_eq!(b_out.to_vec().unwrap(), vec![10, 40, 90, 160]);
    assert_eq!(c_out.to_vec().unwrap(), vec![2.0, 7.0, 14.0, 23.0]);
    assert_eq!(d_out.to_vec().unwrap(), vec![20, 70, 140, 230]);
    assert_eq!(e_out.to_vec().unwrap(), vec![4.0, 13.0, 24.0, 37.0]);
    assert_eq!(f_out.to_vec().unwrap(), vec![40, 130, 240, 370]);
    assert_eq!(g_out.to_vec().unwrap(), vec![7.0, 18.0, 29.0, 44.0]);
    assert_eq!(h_out.to_vec().unwrap(), vec![70, 180, 290, 440]);
    assert_eq!(i_out.to_vec().unwrap(), vec![8.0, 21.0, 34.0, 55.0]);
    assert_eq!(j_out.to_vec().unwrap(), vec![80, 210, 340, 550]);
    assert_eq!(k_out.to_vec().unwrap(), vec![12.0, 31.0, 50.0, 81.0]);
    assert_eq!(l_out.to_vec().unwrap(), vec![120, 310, 500, 810]);
}
