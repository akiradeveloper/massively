pub fn equal_range(input: &[u32], value: u32) -> (usize, usize) {
    (
        crate::lower_bound(input, value),
        crate::upper_bound(input, value),
    )
}
