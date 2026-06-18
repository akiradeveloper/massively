pub fn exclusive_scan(input: &[u32], init: u32) -> Vec<u32> {
    let mut output = vec![0; input.len()];
    let mut acc = init;
    for i in 0..input.len() {
        output[i] = acc;
        acc = crate::max_op(acc, input[i]);
    }
    output
}
