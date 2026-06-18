pub fn find_if(input: &[u32]) -> Option<usize> {
    for i in 0..input.len() {
        if crate::keep(input[i]) {
            return Some(i);
        }
    }
    None
}
