pub fn is_sorted_until(input: &[u32]) -> usize {
    for i in 1..input.len() {
        if crate::bucket_then_value_less(input[i], input[i - 1]) {
            return i;
        }
    }
    input.len()
}
