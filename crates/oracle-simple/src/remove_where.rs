pub fn remove_where(input: &[u32], stencil: &[u32]) -> Vec<u32> {
    let mut output = Vec::new();
    for i in 0..input.len() {
        if stencil[i] == 0 {
            output.push(input[i]);
        }
    }
    output
}
