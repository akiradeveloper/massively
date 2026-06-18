pub fn lexicographical_compare(left: &[u32], right: &[u32]) -> bool {
    let len = left.len().min(right.len());
    for i in 0..len {
        if crate::bucket_then_value_less(left[i], right[i]) {
            return true;
        }
        if crate::bucket_then_value_less(right[i], left[i]) {
            return false;
        }
    }
    left.len() < right.len()
}
