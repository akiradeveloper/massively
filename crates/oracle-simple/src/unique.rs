pub fn unique(input: &[u32]) -> Vec<u32> {
    let mut output = Vec::new();
    if input.is_empty() {
        return output;
    }

    output.push(input[0]);
    for i in 1..input.len() {
        if !crate::same_low_nibble(input[i - 1], input[i]) {
            output.push(input[i]);
        }
    }
    output
}
