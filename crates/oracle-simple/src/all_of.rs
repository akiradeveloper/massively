pub fn all_of(input: &[u32]) -> bool {
    for i in 0..input.len() {
        if !crate::keep(input[i]) {
            return false;
        }
    }
    true
}
