pub fn adjacent_difference(input: &[u32]) -> Vec<u32> {
    let mut output = vec![0; input.len()];
    if input.is_empty() {
        return output;
    }

    output[0] = input[0];
    for i in 1..input.len() {
        output[i] = crate::max_op(input[i], input[i - 1]);
    }
    output
}
