//! The scalar and AVX-512 implementations side by side, on the inputs of `benches/zle.rs`:
//! four groups, `scalar_compress`, `scalar_decompress`, `avx512_compress` and
//! `avx512_decompress`, in uncompressed bytes per second, at the default minimum run, and
//! the same four with `_min_run_16` at a minimum of 16. Then `scalar_compress_aligned_16`
//! and `avx512_compress_aligned_16`: `compress_aligned` at 2 words.
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

use common::{INPUT_LEN, MIN_ZERO_RUNS, SHAPES, compressed, group_name, input};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use zle::DecompressError;
use zle::internals::{
    avx512_compress, avx512_compress_aligned, avx512_decompress, avx512_is_supported,
    scalar_compress, scalar_compress_aligned, scalar_decompress,
};

/// A compressor under test, given its minimum: a run in bytes, or in words for the aligned
/// ones.
type Compress = fn(&[u8], &mut [u8], usize) -> usize;

/// The minimum of the aligned groups, in words: 16 bytes.
const ALIGNED_MIN_ZERO_WORDS: usize = 2;
/// A decompressor under test.
type Decompress = fn(&[u8], &mut [u8]) -> Result<usize, DecompressError>;

fn avx512_compress_checked(src: &[u8], dst: &mut [u8], min_zero_run: usize) -> usize {
    // SAFETY: `bench` asserted avx512_is_supported before running anything.
    unsafe { avx512_compress(src, dst, min_zero_run) }
}

fn avx512_compress_aligned_checked(src: &[u8], dst: &mut [u8], min_zero_words: usize) -> usize {
    // SAFETY: `bench` asserted avx512_is_supported before running anything.
    unsafe { avx512_compress_aligned(src, dst, min_zero_words) }
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
/// * `expected` - The crate's public function for the same rule, whose bytes it must write.
/// * `minimum` - The minimum both are given.
fn bench_compress(
    c: &mut Criterion,
    group_name: &str,
    compress: Compress,
    expected: Compress,
    minimum: usize,
) {
    let mut group = c.benchmark_group(group_name);
    group.throughput(Throughput::Bytes(INPUT_LEN as u64));
    for shape in &SHAPES {
        let src = input(shape);
        let mut dst = vec![0u8; zle::compress_bound(src.len())];
        let mut expected_dst = vec![0u8; zle::compress_bound(src.len())];
        // A benchmark of a wrong result means nothing: both write the same bytes.
        let written = compress(&src, &mut dst, minimum);
        let expected_written = expected(&src, &mut expected_dst, minimum);
        assert_eq!(
            dst[..written],
            expected_dst[..expected_written],
            "{group_name}/{}",
            shape.name
        );
        group.bench_with_input(BenchmarkId::from_parameter(shape.name), &src, |b, src| {
            b.iter(|| compress(black_box(src), black_box(&mut dst), minimum))
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
/// * `min_zero_run` - The shortest run left out of the blocks it reads.
fn bench_decompress(
    c: &mut Criterion,
    group_name: &str,
    decompress: Decompress,
    min_zero_run: usize,
) {
    let mut group = c.benchmark_group(group_name);
    group.throughput(Throughput::Bytes(INPUT_LEN as u64));
    for shape in &SHAPES {
        let src = input(shape);
        let packed = compressed(&src, min_zero_run);
        let mut dst = vec![0u8; src.len()];
        assert_eq!(
            decompress(&packed, &mut dst),
            Ok(src.len()),
            "{group_name}/{}",
            shape.name
        );
        assert_eq!(dst, src, "{group_name}/{}", shape.name);
        group.bench_with_input(
            BenchmarkId::from_parameter(shape.name),
            &packed,
            |b, packed| b.iter(|| decompress(black_box(packed), black_box(&mut dst))),
        );
    }
    group.finish();
}

fn bench(c: &mut Criterion) {
    assert!(
        avx512_is_supported(),
        "this CPU has no AVX-512F, AVX-512BW and BMI2: run on an x86-64-v4 machine"
    );
    for min_zero_run in MIN_ZERO_RUNS {
        bench_compress(
            c,
            &group_name("scalar_compress", min_zero_run),
            scalar_compress,
            zle::compress_with_min_run,
            min_zero_run,
        );
        bench_decompress(
            c,
            &group_name("scalar_decompress", min_zero_run),
            scalar_decompress,
            min_zero_run,
        );
        bench_compress(
            c,
            &group_name("avx512_compress", min_zero_run),
            avx512_compress_checked,
            zle::compress_with_min_run,
            min_zero_run,
        );
        bench_decompress(
            c,
            &group_name("avx512_decompress", min_zero_run),
            avx512_decompress_checked,
            min_zero_run,
        );
    }
    bench_compress(
        c,
        "scalar_compress_aligned_16",
        scalar_compress_aligned,
        zle::compress_aligned,
        ALIGNED_MIN_ZERO_WORDS,
    );
    bench_compress(
        c,
        "avx512_compress_aligned_16",
        avx512_compress_aligned_checked,
        zle::compress_aligned,
        ALIGNED_MIN_ZERO_WORDS,
    );
}

criterion_group!(benches, bench);
criterion_main!(benches);
