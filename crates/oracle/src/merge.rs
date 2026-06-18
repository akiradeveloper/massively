pub fn merge(left: &[u32], right: &[u32]) -> Vec<u32> {
    let mut output = Vec::with_capacity(left.len() + right.len());
    let mut i = 0;
    let mut j = 0;
    while i < left.len() && j < right.len() {
        if crate::bucket_then_value_less(right[j], left[i]) {
            output.push(right[j]);
            j += 1;
        } else {
            output.push(left[i]);
            i += 1;
        }
    }
    while i < left.len() {
        output.push(left[i]);
        i += 1;
    }
    while j < right.len() {
        output.push(right[j]);
        j += 1;
    }
    output
}
