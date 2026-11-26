//! Every way of choosing which zeros are left out: exact runs at minimums of 3, 4, 6, 8 and
//! 16 bytes (`compress_with_min_run`), and runs of whole aligned words at minimums of 2 to 4
//! words (`compress_aligned`), each with the scalar and the AVX-512 code, on two inputs:
//!
//! | input | what |
//! |---|---|
//! | `account_like` | the input of `benches/zle.rs`: dense bytes with 1 to 6 zeros every 1 to 24 bytes, and runs of 16 to 512 zeros |
//! | `mixed_runs` | literals of 1 to 32 non-zero bytes and zero runs of 1 to 64, uniform: every minimum leaves out a different share |
//!
//! On `account_like`, every minimum from 7 to 16 writes the same bytes: it has no zero runs
//! of 7 to 15. `mixed_runs` shows what each step of the minimum costs and saves.
//!
//! It first prints a table per input of what each rule writes: bytes, share of the input
//! and blocks. Then four groups, in uncompressed bytes per second, each with one benchmark
//! per rule and input, named `<rule>/<input>`, the rule by its minimum in bytes
//! (`min_run_3` .. `min_run_16`, `aligned_16` .. `aligned_32`):
//!
//! | group | what |
//! |---|---|
//! | `compress_scalar` | the scalar compressor |
//! | `compress_avx512` | the AVX-512 compressor |
//! | `decompress_scalar` | the scalar decompressor, on that rule's output |
//! | `decompress_avx512` | the AVX-512 decompressor, on that rule's output |
//!
//! For x86_64 CPUs with AVX-512F, AVX-512BW and BMI2 only (x86-64-v4); it stops at once on
//! any other. Built for the server:
//!
//! ```text
//! RUSTFLAGS="-C target-cpu=x86-64-v4" cargo bench --features internals \
//!     --bench min_run_sweep --no-run \
//!     --target x86_64-unknown-linux-gnu --target-dir target/v4
//! ```
//!
//! Then run the executable cargo prints, on that machine, with `--bench`; each benchmark
//! warms up for 1 s and measures for 3 s unless criterion's flags say otherwise. Built that way,
//! the compiler may vectorise the scalar code with AVX-512 too: the scalar rows are then
//! what a v4 build of the scalar code does, not what a CPU without AVX-512 does.

#[cfg(not(target_arch = "x86_64"))]
compile_error!("this benchmark compares against AVX-512: build it for x86_64");

// Only the inputs: the other helpers serve the other benchmarks.
#[allow(dead_code)]
mod common;

use common::{INPUT_LEN, SHAPES, Shape, input};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use std::time::Duration;
use zle::internals::{
    avx512_compress, avx512_compress_aligned, avx512_decompress, avx512_is_supported,
    scalar_compress, scalar_compress_aligned, scalar_decompress,
};

/// The shape of `benches/zle.rs` this benchmark runs on, beside [`MIXED_RUNS`].
const ACCOUNT_LIKE: &str = "account_like";

/// Zero runs of every length from 1 to 64 between short literals: each minimum leaves out
/// a different share of them.
const MIXED_RUNS: Shape = Shape {
    name: "mixed_runs",
    literal_len: (1, 32),
    zero_len: (1, 64),
    inner_gap: (0, 0),
    inner_zero_len: (0, 0),
};

/// The minimums of the exact rule, in bytes: 3, 4 and 6 for 2-byte lanes, where each step
/// changes the speed; then one per wider lane width of the scan, which run at the same
/// speed across their range: 8 stands for 7 to 14 (4-byte lanes), 16 for 15 and more
/// (8-byte lanes).
const MIN_ZERO_RUNS: [usize; 5] = [3, 4, 6, 8, 16];

/// The minimums of the aligned rule, in words.
const MIN_ZERO_WORDS: [usize; 3] = [2, 3, 4];

/// One way of choosing which zeros are left out.
#[derive(Clone, Copy)]
enum Rule {
    /// Every run of at least this many zeros: `compress_with_min_run`.
    MinRun(usize),
    /// Runs of at least this many whole aligned zero words: `compress_aligned`.
    Aligned(usize),
}

impl Rule {
    /// Returns every rule the benchmark measures.
    ///
    /// # Returns
    ///
    /// The exact rules, then the aligned ones, each by increasing minimum.
    fn all() -> Vec<Rule> {
        MIN_ZERO_RUNS
            .iter()
            .map(|&min_zero_run| Rule::MinRun(min_zero_run))
            .chain(
                MIN_ZERO_WORDS
                    .iter()
                    .map(|&min_zero_words| Rule::Aligned(min_zero_words)),
            )
            .collect()
    }

    /// Returns the rule's benchmark name.
    ///
    /// # Returns
    ///
    /// `min_run_` or `aligned_`, then the minimum in bytes.
    fn name(self) -> String {
        match self {
            Rule::MinRun(min_zero_run) => format!("min_run_{min_zero_run}"),
            Rule::Aligned(min_zero_words) => format!("aligned_{}", 8 * min_zero_words),
        }
    }

    /// Compresses with the crate's public function for the rule.
    ///
    /// # Arguments
    ///
    /// * `src` - The input.
    /// * `dst` - At least `zle::compress_bound` of the input.
    ///
    /// # Returns
    ///
    /// How many bytes were written.
    fn compress_public(self, src: &[u8], dst: &mut [u8]) -> usize {
        match self {
            Rule::MinRun(min_zero_run) => zle::compress_with_min_run(src, dst, min_zero_run),
            Rule::Aligned(min_zero_words) => zle::compress_aligned(src, dst, min_zero_words),
        }
    }

    /// Compresses with the scalar code.
    ///
    /// # Arguments
    ///
    /// * `src` - The input.
    /// * `dst` - At least `zle::compress_bound` of the input.
    ///
    /// # Returns
    ///
    /// How many bytes were written.
    fn compress_scalar(self, src: &[u8], dst: &mut [u8]) -> usize {
        match self {
            Rule::MinRun(min_zero_run) => scalar_compress(src, dst, min_zero_run),
            Rule::Aligned(min_zero_words) => scalar_compress_aligned(src, dst, min_zero_words),
        }
    }

    /// Compresses with the AVX-512 code.
    ///
    /// # Arguments
    ///
    /// * `src` - The input.
    /// * `dst` - At least `zle::compress_bound` of the input.
    ///
    /// # Returns
    ///
    /// How many bytes were written.
    fn compress_avx512(self, src: &[u8], dst: &mut [u8]) -> usize {
        // SAFETY: `bench` asserted avx512_is_supported before running anything.
        unsafe {
            match self {
                Rule::MinRun(min_zero_run) => avx512_compress(src, dst, min_zero_run),
                Rule::Aligned(min_zero_words) => avx512_compress_aligned(src, dst, min_zero_words),
            }
        }
    }
}

/// Counts the blocks of a well-formed stream.
///
/// # Arguments
///
/// * `packed` - The blocks.
///
/// # Returns
///
/// How many blocks it holds.
fn block_count(packed: &[u8]) -> usize {
    // Reads a length: the nibble, plus a LEB128 extension when the nibble is 15.
    fn length(nibble: u8, packed: &[u8], position: &mut usize) -> usize {
        if nibble < 15 {
            return nibble as usize;
        }
        let mut rest = 0;
        let mut shift = 0;
        loop {
            let byte = packed[*position];
            *position += 1;
            rest |= ((byte & 0x7f) as usize) << shift;
            shift += 7;
            if byte & 0x80 == 0 {
                return 15 + rest;
            }
        }
    }
    let mut blocks = 0;
    let mut position = 0;
    while position < packed.len() {
        let header = packed[position];
        position += 1;
        position += length(header >> 4, packed, &mut position);
        length(header & 0x0f, packed, &mut position);
        blocks += 1;
    }
    blocks
}

/// Checks every rule on one input and prints what each writes.
///
/// # Arguments
///
/// * `input_name` - The input's name, for the table and the messages.
/// * `src` - The input.
/// * `rules` - The rules.
///
/// # Returns
///
/// Each rule's output, in the order of `rules`.
///
/// # Panics
///
/// Panics if the scalar or AVX-512 compressor writes other bytes than the public function,
/// or a decompressor does not give back the input: a benchmark of a wrong result means
/// nothing.
fn check_and_print_sizes(input_name: &str, src: &[u8], rules: &[Rule]) -> Vec<Vec<u8>> {
    let mut packed_by_rule = Vec::new();
    println!("{input_name}, {} bytes:", src.len());
    println!();
    println!("| rule | bytes written | of the input | blocks | input bytes per block |");
    println!("|---|---|---|---|---|");
    for &rule in rules {
        let what = format!("{} on {input_name}", rule.name());
        let mut packed = vec![0u8; zle::compress_bound(src.len())];
        let written = rule.compress_public(src, &mut packed);
        packed.truncate(written);
        let mut dst = vec![0u8; zle::compress_bound(src.len())];
        let written = rule.compress_scalar(src, &mut dst);
        assert_eq!(dst[..written], packed, "scalar {what}");
        let written = rule.compress_avx512(src, &mut dst);
        assert_eq!(dst[..written], packed, "avx512 {what}");
        // Into a buffer of garbage, so stale bytes can't pass for the input.
        let mut unpacked = vec![0xccu8; src.len()];
        assert_eq!(scalar_decompress(&packed, &mut unpacked), Ok(src.len()));
        assert_eq!(unpacked, src, "scalar decompress {what}");
        unpacked.fill(0xcc);
        assert_eq!(
            avx512_decompress_checked(&packed, &mut unpacked),
            Ok(src.len())
        );
        assert_eq!(unpacked, src, "avx512 decompress {what}");
        let blocks = block_count(&packed);
        println!(
            "| `{}` | {} | {:.1}% | {} | {:.0} |",
            rule.name(),
            packed.len(),
            packed.len() as f64 * 100.0 / src.len() as f64,
            blocks,
            src.len() as f64 / blocks as f64
        );
        packed_by_rule.push(packed);
    }
    println!();
    packed_by_rule
}

fn bench(c: &mut Criterion) {
    assert!(
        avx512_is_supported(),
        "this CPU has no AVX-512F, AVX-512BW and BMI2: run on an x86-64-v4 machine"
    );
    let account_like = SHAPES
        .iter()
        .find(|shape| shape.name == ACCOUNT_LIKE)
        .expect("benches/common has the account_like shape");
    let rules = Rule::all();
    // Each input with every rule's output.
    let inputs: Vec<Input> = [account_like, &MIXED_RUNS]
        .into_iter()
        .map(|shape| {
            let src = input(shape);
            let packed_by_rule = check_and_print_sizes(shape.name, &src, &rules);
            Input {
                name: shape.name,
                src,
                packed_by_rule,
            }
        })
        .collect();

    let mut dst = vec![0u8; zle::compress_bound(INPUT_LEN)];
    let compressors: [(&str, Compress); 2] = [
        ("compress_scalar", Rule::compress_scalar),
        ("compress_avx512", Rule::compress_avx512),
    ];
    for (group_name, compress) in compressors {
        let mut group = c.benchmark_group(group_name);
        group.throughput(Throughput::Bytes(INPUT_LEN as u64));
        for Input {
            name: input_name,
            src,
            ..
        } in &inputs
        {
            for &rule in &rules {
                group.bench_function(BenchmarkId::new(rule.name(), input_name), |b| {
                    b.iter(|| compress(rule, black_box(src), black_box(&mut dst)))
                });
            }
        }
        group.finish();
    }

    let mut unpacked = vec![0u8; INPUT_LEN];
    let decompressors: [(&str, Decompress); 2] = [
        ("decompress_scalar", scalar_decompress),
        ("decompress_avx512", avx512_decompress_checked),
    ];
    for (group_name, decompress) in decompressors {
        let mut group = c.benchmark_group(group_name);
        group.throughput(Throughput::Bytes(INPUT_LEN as u64));
        for Input {
            name: input_name,
            packed_by_rule,
            ..
        } in &inputs
        {
            for (rule, packed) in rules.iter().zip(packed_by_rule) {
                group.bench_function(BenchmarkId::new(rule.name(), input_name), |b| {
                    b.iter(|| decompress(black_box(packed), black_box(&mut unpacked)))
                });
            }
        }
        group.finish();
    }
}

/// One input, with every rule's output.
struct Input {
    /// The shape's name.
    name: &'static str,
    /// The input.
    src: Vec<u8>,
    /// Each rule's output, in the order of [`Rule::all`].
    packed_by_rule: Vec<Vec<u8>>,
}

/// A compressor under test, given the rule.
type Compress = fn(Rule, &[u8], &mut [u8]) -> usize;

/// A decompressor under test.
type Decompress = fn(&[u8], &mut [u8]) -> Result<usize, zle::DecompressError>;

fn avx512_decompress_checked(src: &[u8], dst: &mut [u8]) -> Result<usize, zle::DecompressError> {
    // SAFETY: `bench` asserted avx512_is_supported before running anything.
    unsafe { avx512_decompress(src, dst) }
}

criterion_group! {
    name = benches;
    // 64 benchmarks: shorter than criterion's 3 s and 5 s, still steady on a quiet server.
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3));
    targets = bench
}
criterion_main!(benches);
