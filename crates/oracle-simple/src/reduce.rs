pub fn reduce(input: &[u32], init: u32) -> u32 {
    let mut acc = init;
    for i in 0..input.len() {
        acc = crate::max_op(acc, input[i]);
    }
    acc
}
