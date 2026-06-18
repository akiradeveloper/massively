pub fn equal(left: &[u32], right: &[u32]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    for i in 0..left.len() {
        if !crate::same_low_nibble(left[i], right[i]) {
            return false;
        }
    }
    true
}
