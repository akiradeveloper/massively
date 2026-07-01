pub fn fill(value: u32, output: &mut [u32]) {
    for item in output {
        *item = value;
    }
}
