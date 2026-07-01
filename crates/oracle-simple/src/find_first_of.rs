pub fn find_first_of(input: &[u32], needles: &[u32]) -> Option<usize> {
    for i in 0..input.len() {
        for j in 0..needles.len() {
            if crate::same_low_nibble(input[i], needles[j]) {
                return Some(i);
            }
        }
    }
    None
}
