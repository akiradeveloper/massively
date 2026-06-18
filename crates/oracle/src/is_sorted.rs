pub fn is_sorted(input: &[u32]) -> bool {
    crate::is_sorted_until(input) == input.len()
}
