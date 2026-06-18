pub fn is_partitioned(input: &[u32]) -> bool {
    let mut seen_rejected = false;
    for i in 0..input.len() {
        if crate::keep(input[i]) {
            if seen_rejected {
                return false;
            }
        } else {
            seen_rejected = true;
        }
    }
    true
}
