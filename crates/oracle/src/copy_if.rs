pub fn copy_if(input: &[u32], stencil: &[u32]) -> Vec<u32> {
    let mut output = Vec::new();
    for i in 0..input.len() {
        if crate::keep(stencil[i]) {
            output.push(input[i]);
        }
    }
    output
}
