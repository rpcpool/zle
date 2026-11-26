//! zle against a plain copy, on the `account_like` input of `benches/zle.rs`: dense bytes with 1 to 6 zeros every 1 to 24 bytes, and runs of 16 to
//! 512 zeros in between, as in Solana account data.
//!
//! Two groups, `account_like_compress` and `account_like_decompress`, in uncompressed
//! bytes per second, each with:
//!
//! | benchmark | what |
//! |---|---|
//! | `copy` | a plain `memcpy` of the input: what no packing costs |
//! | `zle` | zle at the default minimum run |
//! | `zle_min_run_16` | zle at a minimum of 16, the one excelsior uses |
//! | `zle_aligned_16` | `zle::compress_aligned` at 2 words: runs of whole zero words |
//!
//! Before timing, it prints how many bytes each one writes, and checks each round trip.
//!
//! excelsior's server is built for x86-64-v4, so build this the same way and run it on an
//! AVX-512 machine: zle then calls its AVX-512 code directly, and the copy is compiled with
//! the same features.
//!
//! ```text
//! RUSTFLAGS="-C target-cpu=x86-64-v4" cargo bench --bench account_like --no-run \
//!     --target x86_64-unknown-linux-gnu --target-dir target/v4
//! ```
//!
//! Then run the executable cargo prints, on that machine, with `--bench`. Built without
//! `-C target-cpu`, zle checks the CPU at run time, and on one without AVX-512 this
//! measures its scalar code.

mod common;

use common::{INPUT_LEN, MIN_ZERO_RUNS, SHAPES, compressed, group_name, input};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

/// The shape of `benches/zle.rs` this benchmark runs on.
const SHAPE_NAME: &str = "account_like";

/// The minimum of `zle_aligned_16`, in words: 16 bytes.
const ALIGNED_MIN_ZERO_WORDS: usize = 2;

/// Compresses an input with `zle::compress_aligned` into a fresh buffer.
///
/// # Arguments
///
/// * `src` - The input.
///
/// # Returns
///
/// The compressed bytes.
fn aligned_compressed(src: &[u8]) -> Vec<u8> {
    let mut dst = vec![0u8; zle::compress_bound(src.len())];
    let written = zle::compress_aligned(src, &mut dst, ALIGNED_MIN_ZERO_WORDS);
    dst.truncate(written);
    dst
}

/// Prints how many bytes a packer writes for the input.
///
/// # Arguments
///
/// * `name` - The benchmark's name for the packer.
/// * `packed_len` - The bytes it wrote.
fn print_size(name: &str, packed_len: usize) {
    println!(
        "{SHAPE_NAME}: {name} writes {packed_len} bytes of {INPUT_LEN} ({:.1}%)",
        packed_len as f64 * 100.0 / INPUT_LEN as f64
    );
}

/// Benchmarks the copy and zle on the `account_like` input.
///
/// # Arguments
///
/// * `c` - Criterion.
///
/// # Panics
///
/// Panics if a packer's output does not unpack to the input.
fn bench(c: &mut Criterion) {
    let shape = SHAPES
        .iter()
        .find(|shape| shape.name == SHAPE_NAME)
        .expect("benches/common has the account_like shape");
    let src = input(shape);
    let mut unpacked = vec![0u8; src.len()];

    // A benchmark of a wrong result means nothing: check each round trip once, outside
    // the timed loops, into a buffer of garbage.
    let zle_packed = MIN_ZERO_RUNS.map(|min_zero_run| compressed(&src, min_zero_run));
    for (packed, min_zero_run) in zle_packed.iter().zip(MIN_ZERO_RUNS) {
        unpacked.fill(0xcc);
        assert_eq!(zle::decompress(packed, &mut unpacked), Ok(src.len()));
        assert_eq!(unpacked, src, "zle at {min_zero_run}");
        print_size(&group_name("zle", min_zero_run), packed.len());
    }
    let aligned_packed = aligned_compressed(&src);
    unpacked.fill(0xcc);
    assert_eq!(
        zle::decompress(&aligned_packed, &mut unpacked),
        Ok(src.len())
    );
    assert_eq!(unpacked, src, "zle_aligned_16");
    print_size("zle_aligned_16", aligned_packed.len());

    let mut group = c.benchmark_group("account_like_compress");
    group.throughput(Throughput::Bytes(INPUT_LEN as u64));
    let mut dst = vec![0u8; zle::compress_bound(src.len())];
    group.bench_function("copy", |b| {
        b.iter(|| black_box(&mut dst)[..src.len()].copy_from_slice(black_box(&src)))
    });
    for min_zero_run in MIN_ZERO_RUNS {
        group.bench_function(group_name("zle", min_zero_run), |b| {
            b.iter(|| {
                zle::compress_with_min_run(black_box(&src), black_box(&mut dst), min_zero_run)
            })
        });
    }
    group.bench_function("zle_aligned_16", |b| {
        b.iter(|| {
            zle::compress_aligned(black_box(&src), black_box(&mut dst), ALIGNED_MIN_ZERO_WORDS)
        })
    });
    group.finish();

    let mut group = c.benchmark_group("account_like_decompress");
    group.throughput(Throughput::Bytes(INPUT_LEN as u64));
    group.bench_function("copy", |b| {
        b.iter(|| black_box(&mut unpacked).copy_from_slice(black_box(&src)))
    });
    for (packed, min_zero_run) in zle_packed.iter().zip(MIN_ZERO_RUNS) {
        group.bench_function(group_name("zle", min_zero_run), |b| {
            b.iter(|| zle::decompress(black_box(packed), black_box(&mut unpacked)))
        });
    }
    group.bench_function("zle_aligned_16", |b| {
        b.iter(|| zle::decompress(black_box(&aligned_packed), black_box(&mut unpacked)))
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
