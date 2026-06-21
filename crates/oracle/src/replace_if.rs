pub fn replace_if(input: &[u32], replacement: u32, stencil: &[u32]) -> Vec<u32> {
    let mut output = vec![0; input.len()];
    for i in 0..input.len() {
        output[i] = if stencil[i] != 0 {
            replacement
        } else {
            input[i]
        };
    }
    output
}
