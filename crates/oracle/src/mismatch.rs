pub fn mismatch(left: &[u32], right: &[u32]) -> Option<usize> {
    let len = left.len().min(right.len());
    for i in 0..len {
        if !crate::same_low_nibble(left[i], right[i]) {
            return Some(i);
        }
    }
    if left.len() == right.len() {
        None
    } else {
        Some(len)
    }
}
