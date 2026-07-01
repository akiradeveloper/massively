pub fn transform(input: &[u32]) -> Vec<u32> {
    let mut output = vec![0; input.len()];
    for i in 0..input.len() {
        output[i] = crate::xor_mask(input[i]);
    }
    output
}
