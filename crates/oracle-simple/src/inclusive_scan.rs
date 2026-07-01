pub fn inclusive_scan(input: &[u32]) -> Vec<u32> {
    let mut output = vec![0; input.len()];
    if input.is_empty() {
        return output;
    }

    output[0] = input[0];
    for i in 1..input.len() {
        output[i] = crate::max_op(output[i - 1], input[i]);
    }
    output
}
