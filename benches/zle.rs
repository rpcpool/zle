//! Compress and decompress throughput, in uncompressed bytes per second, over inputs that
//! differ in their block sizes: the per-block cost dominates short blocks, the copy and
//! fill speed dominates long ones. Each at the default minimum run (groups `compress` and
//! `decompress`) and at 16 (`compress_min_run_16` and `decompress_min_run_16`).
//!
//! `.cargo/config.toml` runs every binary under Intel SDE, which makes timings meaningless.
//! Run natively, on the hardware you care about:
//!
//! ```text
//! CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER=env cargo bench
//! ```
//!
//! On a CPU without AVX-512 this measures the scalar compressor; `decompress` is scalar
//! everywhere.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

mod common;

use common::{INPUT_LEN, MIN_ZERO_RUNS, SHAPES, compressed, group_name, input};

fn bench(c: &mut Criterion) {
    for min_zero_run in MIN_ZERO_RUNS {
        bench_min_run(c, min_zero_run);
    }
}

/// Benchmarks compress and decompress on every shape at one minimum run.
///
/// # Arguments
///
/// * `c` - Criterion.
/// * `min_zero_run` - The shortest run left out.
fn bench_min_run(c: &mut Criterion, min_zero_run: usize) {
    let mut compress_group = c.benchmark_group(group_name("compress", min_zero_run));
    compress_group.throughput(Throughput::Bytes(INPUT_LEN as u64));
    for shape in &SHAPES {
        let src = input(shape);
        let mut dst = vec![0u8; zle::compress_bound(src.len())];
        compress_group.bench_with_input(BenchmarkId::from_parameter(shape.name), &src, |b, src| {
            b.iter(|| zle::compress_with_min_run(black_box(src), black_box(&mut dst), min_zero_run))
        });
    }
    compress_group.finish();

    let mut decompress_group = c.benchmark_group(group_name("decompress", min_zero_run));
    decompress_group.throughput(Throughput::Bytes(INPUT_LEN as u64));
    for shape in &SHAPES {
        let src = input(shape);
        let packed = compressed(&src, min_zero_run);
        let mut dst = vec![0u8; src.len()];
        // Check once outside the timed loop: a benchmark of a wrong result means nothing.
        assert_eq!(
            zle::decompress(&packed, &mut dst),
            Ok(src.len()),
            "{}",
            shape.name
        );
        assert_eq!(dst, src, "{}", shape.name);
        decompress_group.bench_with_input(
            BenchmarkId::from_parameter(shape.name),
            &packed,
            |b, packed| b.iter(|| zle::decompress(black_box(packed), black_box(&mut dst))),
        );
    }
    decompress_group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
