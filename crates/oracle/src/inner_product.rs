pub fn inner_product(left: &[u32], right: &[u32], init: u32) -> u32 {
    let mut acc = init;
    for i in 0..left.len() {
        acc = crate::max_op(acc, crate::max_op(left[i], right[i]));
    }
    acc
}
