pub fn scatter(values: &[u32], indices: &[u32], len: usize, default: u32) -> Vec<u32> {
    let mut output = vec![default; len];
    for i in 0..values.len() {
        output[indices[i] as usize] = values[i];
    }
    output
}
