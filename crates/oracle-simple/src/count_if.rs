pub fn count_if(input: &[u32]) -> usize {
    let mut count = 0;
    for i in 0..input.len() {
        if crate::keep(input[i]) {
            count += 1;
        }
    }
    count
}
