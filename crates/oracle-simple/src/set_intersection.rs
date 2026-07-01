pub fn set_intersection(left: &[u32], right: &[u32]) -> Vec<u32> {
    let mut output = Vec::new();
    let mut i = 0;
    let mut j = 0;
    while i < left.len() && j < right.len() {
        if crate::bucket_then_value_less(left[i], right[j]) {
            i += 1;
        } else if crate::bucket_then_value_less(right[j], left[i]) {
            j += 1;
        } else {
            output.push(left[i]);
            i += 1;
            j += 1;
        }
    }
    output
}
