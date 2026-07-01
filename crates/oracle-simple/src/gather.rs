pub fn gather(input: &[u32], indices: &[u32]) -> Vec<u32> {
    let mut output = vec![0; indices.len()];
    for i in 0..indices.len() {
        output[i] = input[indices[i] as usize];
    }
    output
}
