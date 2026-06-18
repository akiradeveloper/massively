pub fn remove_if(input: &[u32]) -> Vec<u32> {
    let mut output = Vec::new();
    for i in 0..input.len() {
        if !crate::keep(input[i]) {
            output.push(input[i]);
        }
    }
    output
}
