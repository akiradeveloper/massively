pub fn gather_where(input: &[u32], indices: &[u32], stencil: &[u32]) -> Vec<u32> {
    let mut output = vec![0; indices.len()];
    for i in 0..indices.len() {
        let value = input[indices[i] as usize];
        if stencil[i] != 0 {
            output[i] = value;
        }
    }
    output
}
