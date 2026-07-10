use std::time::Duration;

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, ReductionOp, UnaryOp, lazy, reduce, zip2};

const LEN: u32 = 4_000_000_000;

struct Sum;
struct DetectHit;

#[cubecl::cube]
impl ReductionOp<u32> for Sum {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        lhs + rhs
    }
}

#[cubecl::cube]
impl UnaryOp<(f32, f32)> for DetectHit {
    type Output = u32;

    fn apply(input: (f32, f32)) -> u32 {
        let d2 = input.0 * input.0 + input.1 * input.1;
        if d2 <= 1.0 { 1_u32 } else { 0_u32 }
    }
}

fn bench_reduce_4g(c: &mut Criterion) {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);

    let mut group = c.benchmark_group("reduce_4g");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_secs(4));
    group.throughput(Throughput::Elements(LEN as u64));
    group.bench_function("constant_u32", |b| {
        b.iter(|| {
            let input = lazy::constant(black_box(1_u32)).take(LEN);
            let result = reduce(&exec, input, 0_u32, Sum).unwrap();
            assert_eq!(black_box(result), LEN);
        })
    });
    group.bench_function("monte_carlo_pi", |b| {
        b.iter(|| {
            let x = massively::util::random::uniform_f32(0.0, 1.0, black_box(0))
                .unwrap()
                .take(LEN);
            let y = massively::util::random::uniform_f32(0.0, 1.0, black_box(1))
                .unwrap()
                .take(LEN);
            let hits = lazy::transform(zip2(x, y), DetectHit);
            let count = reduce(&exec, hits, 0_u32, Sum).unwrap();
            let pi = count as f64 * 4.0 / LEN as f64;
            assert!((3.10..3.18).contains(&black_box(pi)), "pi={pi}");
        })
    });
    group.finish();
}

criterion_group!(benches, bench_reduce_4g);
criterion_main!(benches);
