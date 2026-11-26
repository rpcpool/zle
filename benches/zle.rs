//! Compress and decompress throughput, in uncompressed bytes per second, over inputs that
//! differ in their block sizes: the per-block cost dominates short blocks, the copy and
//! fill speed dominates long ones.
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

use common::{INPUT_LEN, SHAPES, compressed, input};

fn bench(c: &mut Criterion) {
    let mut compress_group = c.benchmark_group("compress");
    compress_group.throughput(Throughput::Bytes(INPUT_LEN as u64));
    for shape in &SHAPES {
        let src = input(shape);
        let mut dst = vec![0u8; zle::compress_bound(src.len())];
        compress_group.bench_with_input(BenchmarkId::from_parameter(shape.name), &src, |b, src| {
            b.iter(|| zle::compress(black_box(src), black_box(&mut dst)))
        });
    }
    compress_group.finish();

    let mut decompress_group = c.benchmark_group("decompress");
    decompress_group.throughput(Throughput::Bytes(INPUT_LEN as u64));
    for shape in &SHAPES {
        let src = input(shape);
        let packed = compressed(&src);
        let mut dst = vec![0u8; src.len()];
        // Check once outside the timed loop: a benchmark of a wrong result means nothing.
        assert_eq!(zle::decompress(&packed, &mut dst), Ok(src.len()), "{}", shape.name);
        assert_eq!(dst, src, "{}", shape.name);
        decompress_group.bench_with_input(BenchmarkId::from_parameter(shape.name), &packed, |b, packed| {
            b.iter(|| zle::decompress(black_box(packed), black_box(&mut dst)))
        });
    }
    decompress_group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
