pub fn scatter_if(
    values: &[u32],
    indices: &[u32],
    len: usize,
    default: u32,
    stencil: &[u32],
) -> Vec<u32> {
    let mut output = vec![default; len];
    for i in 0..values.len() {
        let value = values[i];
        if crate::keep(stencil[i]) {
            output[indices[i] as usize] = value;
        }
    }
    output
}
