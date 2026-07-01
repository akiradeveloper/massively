pub fn reverse(input: &[u32]) -> Vec<u32> {
    let mut output = vec![0; input.len()];
    for i in 0..input.len() {
        output[i] = input[input.len() - 1 - i];
    }
    output
}
