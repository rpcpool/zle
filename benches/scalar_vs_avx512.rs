//! The scalar and AVX-512 implementations side by side, on the inputs of `benches/zle.rs`:
//! four groups, `scalar_compress`, `scalar_decompress`, `avx512_compress` and
//! `avx512_decompress`, in uncompressed bytes per second.
//!
//! For x86_64 CPUs with AVX-512F, AVX-512BW and BMI2 only (x86-64-v4); it stops at once on
//! any other. Build it without `-C target-cpu`: the AVX-512 functions enable their features
//! themselves, and a baseline build keeps the compiler from vectorising the scalar code with
//! AVX-512, which would blur the comparison.
//!
//! ```text
//! cargo bench --features internals --bench scalar_vs_avx512 --no-run
//! ```
//!
//! Then run the executable cargo prints, on the AVX-512 machine, with `--bench`.

#[cfg(not(target_arch = "x86_64"))]
compile_error!("this benchmark compares against AVX-512: build it for x86_64");

mod common;

use common::{INPUT_LEN, SHAPES, compressed, input};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use zle::DecompressError;
use zle::internals::{avx512_compress, avx512_decompress, avx512_is_supported, scalar_compress, scalar_decompress};

/// A compressor under test.
type Compress = fn(&[u8], &mut [u8]) -> usize;
/// A decompressor under test.
type Decompress = fn(&[u8], &mut [u8]) -> Result<usize, DecompressError>;

fn avx512_compress_checked(src: &[u8], dst: &mut [u8]) -> usize {
    // SAFETY: `bench` asserted avx512_is_supported before running anything.
    unsafe { avx512_compress(src, dst) }
}

fn avx512_decompress_checked(src: &[u8], dst: &mut [u8]) -> Result<usize, DecompressError> {
    // SAFETY: `bench` asserted avx512_is_supported before running anything.
    unsafe { avx512_decompress(src, dst) }
}

/// Benchmarks one compressor on every shape, checking its output first.
///
/// # Arguments
///
/// * `c` - Criterion.
/// * `group_name` - The group's name, which names the table.
/// * `compress` - The compressor.
fn bench_compress(c: &mut Criterion, group_name: &str, compress: Compress) {
    let mut group = c.benchmark_group(group_name);
    group.throughput(Throughput::Bytes(INPUT_LEN as u64));
    for shape in &SHAPES {
        let src = input(shape);
        let mut dst = vec![0u8; zle::compress_bound(src.len())];
        // A benchmark of a wrong result means nothing: both write the same bytes.
        let written = compress(&src, &mut dst);
        assert_eq!(dst[..written], compressed(&src), "{group_name}/{}", shape.name);
        group.bench_with_input(BenchmarkId::from_parameter(shape.name), &src, |b, src| {
            b.iter(|| compress(black_box(src), black_box(&mut dst)))
        });
    }
    group.finish();
}

/// Benchmarks one decompressor on every shape, checking its output first.
///
/// # Arguments
///
/// * `c` - Criterion.
/// * `group_name` - The group's name, which names the table.
/// * `decompress` - The decompressor.
fn bench_decompress(c: &mut Criterion, group_name: &str, decompress: Decompress) {
    let mut group = c.benchmark_group(group_name);
    group.throughput(Throughput::Bytes(INPUT_LEN as u64));
    for shape in &SHAPES {
        let src = input(shape);
        let packed = compressed(&src);
        let mut dst = vec![0u8; src.len()];
        assert_eq!(decompress(&packed, &mut dst), Ok(src.len()), "{group_name}/{}", shape.name);
        assert_eq!(dst, src, "{group_name}/{}", shape.name);
        group.bench_with_input(BenchmarkId::from_parameter(shape.name), &packed, |b, packed| {
            b.iter(|| decompress(black_box(packed), black_box(&mut dst)))
        });
    }
    group.finish();
}

fn bench(c: &mut Criterion) {
    assert!(avx512_is_supported(), "this CPU has no AVX-512F, AVX-512BW and BMI2: run on an x86-64-v4 machine");
    bench_compress(c, "scalar_compress", scalar_compress);
    bench_decompress(c, "scalar_decompress", scalar_decompress);
    bench_compress(c, "avx512_compress", avx512_compress_checked);
    bench_decompress(c, "avx512_decompress", avx512_decompress_checked);
}

criterion_group!(benches, bench);
criterion_main!(benches);
