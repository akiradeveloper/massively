pub(crate) fn lower_bound_one(input: &[u32], value: u32) -> usize {
    let mut first = 0;
    let mut count = input.len();
    while count > 0 {
        let step = count / 2;
        let mid = first + step;
        if crate::bucket_then_value_less(input[mid], value) {
            first = mid + 1;
            count -= step + 1;
        } else {
            count = step;
        }
    }
    first
}

pub fn lower_bound(input: &[u32], values: &[u32]) -> Vec<u32> {
    values
        .iter()
        .map(|&value| lower_bound_one(input, value) as u32)
        .collect()
}
