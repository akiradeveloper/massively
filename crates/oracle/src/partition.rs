pub fn partition(input: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let mut matching = Vec::new();
    let mut failing = Vec::new();
    for i in 0..input.len() {
        if crate::keep(input[i]) {
            matching.push(input[i]);
        } else {
            failing.push(input[i]);
        }
    }
    (matching, failing)
}
