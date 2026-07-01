pub fn adjacent_find(input: &[u32]) -> Option<usize> {
    for i in 1..input.len() {
        if crate::same_low_nibble(input[i - 1], input[i]) {
            return Some(i - 1);
        }
    }
    None
}
