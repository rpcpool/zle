//! What the tests share: the AVX-512 check, a seeded random generator, input generators, a
//! byte-at-a-time compressor and decompressor to check against, and a buffer that ends at
//! an unmapped page.

use crate::block::{block_len, write_block};
use crate::{compress_bound, escape_len};
use std::ffi::c_void;

/// The minimum runs the tests try: each side of every lane-width change (6 | 7, 14 | 15),
/// the smallest, the one excelsior uses (16), and some long ones.
pub const MIN_ZERO_RUNS: [usize; 12] = [3, 4, 6, 7, 8, 14, 15, 16, 17, 31, 64, 300];

/// Panics unless the CPU (or SDE) has every feature the AVX-512 code uses.
///
/// Running such a function on a CPU without AVX-512 is undefined behaviour (in practice,
/// SIGILL), so every test calling one calls this first.
///
/// # Panics
///
/// Panics if AVX-512F, AVX-512BW or BMI2 is missing.
#[cfg(target_arch = "x86_64")]
pub fn require_avx512() {
    assert!(
        is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("bmi2"),
        "this CPU has no AVX-512: run `cargo test`, which goes through Intel SDE (.cargo/config.toml)"
    );
}

/// A seeded xorshift64* generator: the same inputs on every run, no dependency.
pub struct Rng(u64);

impl Rng {
    /// Starts a generator.
    ///
    /// # Arguments
    ///
    /// * `seed` - Any value but 0.
    ///
    /// # Panics
    ///
    /// Panics if `seed` is 0, where xorshift is stuck.
    pub fn new(seed: u64) -> Self {
        assert_ne!(seed, 0, "xorshift is stuck at 0");
        Self(seed)
    }

    /// Returns the next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// Returns a number in `0..bound`.
    pub fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }

    /// Returns a non-zero byte.
    pub fn nonzero_byte(&mut self) -> u8 {
        1 + self.below(255) as u8
    }
}

/// Builds up to `max_len` bytes of non-zero stretches and zero runs, half of the runs around
/// the minimum (1 to `min_zero_run + 3` zeros), half long (up to `200 + 2 * min_zero_run`).
///
/// # Arguments
///
/// * `rng` - The generator.
/// * `max_len` - The most bytes; the length is random in `0..=max_len`.
/// * `min_zero_run` - The minimum run the input is meant for.
///
/// # Returns
///
/// The bytes.
pub fn sparse(rng: &mut Rng, max_len: usize, min_zero_run: usize) -> Vec<u8> {
    let len = rng.below(max_len + 1);
    let mut bytes = Vec::with_capacity(len);
    while bytes.len() < len {
        if rng.below(2) == 0 {
            let run_len = if rng.below(2) == 0 {
                1 + rng.below(min_zero_run + 3)
            } else {
                1 + rng.below(200 + 2 * min_zero_run)
            };
            bytes.extend(std::iter::repeat_n(0, run_len));
        } else {
            for _ in 0..1 + rng.below(40) {
                bytes.push(rng.nonzero_byte());
            }
        }
    }
    bytes.truncate(len);
    bytes
}

/// Compresses one byte at a time: slow, but simple enough to trust. Both compressors must
/// give exactly its bytes.
///
/// # Arguments
///
/// * `src` - The input.
/// * `min_zero_run` - The shortest run left out, at least 1.
///
/// # Returns
///
/// The compressed bytes.
pub fn reference_compressed(src: &[u8], min_zero_run: usize) -> Vec<u8> {
    let mut dst = vec![0u8; compress_bound(src.len())];
    let mut position = 0;
    let mut written = 0;
    while position < src.len() {
        let literal_start = position;
        let mut zero_len = 0;
        // Walk until `min_zero_run` zeros in a row, or the end.
        while position < src.len() && zero_len < min_zero_run {
            zero_len = if src[position] == 0 { zero_len + 1 } else { 0 };
            position += 1;
        }
        if zero_len < min_zero_run {
            // The end: fewer zeros stay in the literal.
            zero_len = 0;
        }
        while zero_len > 0 && position < src.len() && src[position] == 0 {
            zero_len += 1;
            position += 1;
        }
        let literal = &src[literal_start..position - zero_len];
        if written + block_len(literal.len(), zero_len) >= escape_len(src.len()) {
            let escaped = write_block(&mut dst, src, 0);
            dst.truncate(escaped);
            return dst;
        }
        written += write_block(&mut dst[written..], literal, zero_len);
    }
    dst.truncate(written);
    dst
}

/// Compresses by the aligned rule, one word at a time: slow, but simple enough to trust.
/// Both aligned compressors must give exactly its bytes.
///
/// A word is the 8 bytes at `8 * k`, for every whole word of the input. A run is a maximal
/// sequence of all-zero words, at least `min_zero_words` long, so it starts and ends on
/// word boundaries. Everything else is literal, the bytes after the last word included.
///
/// # Arguments
///
/// * `src` - The input.
/// * `min_zero_words` - The fewest zero words left out, at least 1.
///
/// # Returns
///
/// The compressed bytes: the blocks, or the escape if they are not shorter than it.
pub fn reference_compressed_aligned(src: &[u8], min_zero_words: usize) -> Vec<u8> {
    let word_count = src.len() / 8;
    let is_zero_word = |word_index: usize| {
        src[8 * word_index..8 * word_index + 8]
            .iter()
            .all(|&byte| byte == 0)
    };
    // Every block, as (literal, zero_len).
    let mut blocks: Vec<(&[u8], usize)> = Vec::new();
    let mut literal_start = 0;
    let mut word_index = 0;
    while word_index < word_count {
        if !is_zero_word(word_index) {
            word_index += 1;
            continue;
        }
        let run_start = word_index;
        while word_index < word_count && is_zero_word(word_index) {
            word_index += 1;
        }
        if word_index - run_start >= min_zero_words {
            blocks.push((
                &src[literal_start..8 * run_start],
                8 * (word_index - run_start),
            ));
            literal_start = 8 * word_index;
        }
    }
    if literal_start < src.len() {
        blocks.push((&src[literal_start..], 0));
    }

    let blocks_len: usize = blocks
        .iter()
        .map(|(literal, zero_len)| block_len(literal.len(), *zero_len))
        .sum();
    let mut dst = vec![0u8; compress_bound(src.len())];
    if blocks_len >= escape_len(src.len()) {
        let escaped = write_block(&mut dst, src, 0);
        dst.truncate(escaped);
        return dst;
    }
    let mut written = 0;
    for (literal, zero_len) in blocks {
        written += write_block(&mut dst[written..], literal, zero_len);
    }
    dst.truncate(written);
    dst
}

/// The inputs every aligned compressor is checked on against
/// [`reference_compressed_aligned`], with a name for each.
///
/// They cover: runs of exactly the minimum and one word fewer; zeros at every offset into a
/// word, which fill one whole word fewer than their length suggests; zeros around a run that
/// fill no word; trailing partial words; short inputs; all zeros; runs starting, ending and
/// crossing every 64-byte vector boundary; and long runs past the 256-byte skips.
///
/// # Arguments
///
/// * `min_zero_words` - The minimum the inputs are built around, at least 1.
///
/// # Returns
///
/// `(name, input)` pairs.
pub fn aligned_edge_cases(min_zero_words: usize) -> Vec<(String, Vec<u8>)> {
    let min_zero_len = 8 * min_zero_words;
    let mut cases = Vec::new();
    // `zero_len` zeros at `start`, between bytes of 0x5a, in an input of `len` bytes.
    let zeros_at = |len: usize, start: usize, zero_len: usize| {
        let mut src = vec![0x5au8; len];
        src[start..(start + zero_len).min(len)].fill(0);
        src
    };
    for zero_len in [
        min_zero_len - 8,
        min_zero_len - 1,
        min_zero_len,
        min_zero_len + 7,
        min_zero_len + 8,
    ] {
        for start in 0..=17 {
            let len = start + zero_len + 9;
            cases.push((
                format!("{zero_len} zeros at {start}"),
                zeros_at(len, start, zero_len),
            ));
            // The same run ending the input: its last word may be partial.
            cases.push((
                format!("{zero_len} zeros at {start}, at the end"),
                zeros_at(start + zero_len, start, zero_len),
            ));
        }
    }
    for len in [0, 1, 7, 8, 9, 15, 16, 17, 23, 24, 25, 100, 1000] {
        cases.push((format!("{len} zeros"), vec![0; len]));
        cases.push((format!("{len} bytes of 0x5a"), vec![0x5a; len]));
        // A partial word of zeros after the last whole word.
        let mut src = vec![0x5a; len];
        src[len - len % 8..].fill(0);
        cases.push((format!("{len} bytes, trailing partial word of zeros"), src));
    }
    for boundary in [64, 128, 192, 256, 320] {
        for zero_len in [min_zero_len, 64, 72, 120] {
            for offset in [-64, -24, -16, -8, -1, 0, 1, 8, 16] {
                let start = (boundary as isize + offset) as usize;
                cases.push((
                    format!("{zero_len} zeros at {start}, around byte {boundary}"),
                    zeros_at(boundary + 200, start, zero_len),
                ));
            }
        }
    }
    for zero_len in [255, 256, 257, 512, 600, 1024, 1100] {
        for start in [0, 3, 8, 64, 70] {
            cases.push((
                format!("{zero_len} zeros at {start}, long"),
                zeros_at(start + zero_len + 70, start, zero_len),
            ));
        }
    }
    cases
}

/// Decompresses a well-formed stream, one block at a time.
///
/// # Arguments
///
/// * `packed` - The blocks.
///
/// # Returns
///
/// The original bytes.
///
/// # Panics
///
/// Panics if `packed` is not a well-formed stream.
pub fn decompress(packed: &[u8]) -> Vec<u8> {
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
    let mut out = Vec::new();
    let mut position = 0;
    while position < packed.len() {
        let header = packed[position];
        position += 1;
        let literal_len = length(header >> 4, packed, &mut position);
        out.extend_from_slice(&packed[position..position + literal_len]);
        position += literal_len;
        let zero_len = length(header & 0x0f, packed, &mut position);
        out.resize(out.len() + zero_len, 0);
    }
    out
}

/// One page.
pub const PAGE: usize = 4096;

const PROT_NONE: i32 = 0;
const PROT_READ: i32 = 1;
const PROT_WRITE: i32 = 2;
const MAP_PRIVATE: i32 = 0x02;
const MAP_ANONYMOUS: i32 = 0x20;

unsafe extern "C" {
    fn mmap(
        addr: *mut c_void,
        len: usize,
        prot: i32,
        flags: i32,
        fd: i32,
        offset: i64,
    ) -> *mut c_void;
    fn mprotect(addr: *mut c_void, len: usize, prot: i32) -> i32;
    fn munmap(addr: *mut c_void, len: usize) -> i32;
}

/// A readable, writable page followed by an unmapped one (`PROT_NONE`): reading a single
/// byte past the end of [`Guarded::tail`] crashes the test with SIGSEGV.
///
/// It proves the AVX-512 code never reads past its input.
///
/// # Examples
///
/// ```ignore
/// let mut guarded = Guarded::new(0x5a);
/// let src = guarded.tail(100);
/// src[..10].fill(0);
/// compress_avx512(src, &mut dst); // crashes if it reads past src
/// ```
pub struct Guarded {
    base: *mut u8,
}

impl Guarded {
    /// Maps the two pages, the first one filled with `fill`.
    ///
    /// # Arguments
    ///
    /// * `fill` - The byte the readable page holds.
    ///
    /// # Panics
    ///
    /// Panics if `mmap` or `mprotect` fails.
    pub fn new(fill: u8) -> Self {
        // SAFETY: an anonymous private mapping at an address the kernel picks.
        let base = unsafe {
            mmap(
                std::ptr::null_mut(),
                2 * PAGE,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert_ne!(base as isize, -1, "mmap failed");
        // SAFETY: the second page belongs to the mapping just made.
        let protected = unsafe { mprotect(base.cast::<u8>().add(PAGE).cast(), PAGE, PROT_NONE) };
        assert_eq!(protected, 0, "mprotect failed");
        let base = base.cast::<u8>();
        // SAFETY: the first page is mapped, readable and writable.
        unsafe { std::ptr::write_bytes(base, fill, PAGE) };
        Self { base }
    }

    /// Returns the last `len` bytes before the unmapped page.
    ///
    /// # Arguments
    ///
    /// * `len` - How many bytes.
    ///
    /// # Panics
    ///
    /// Panics if `len` is over a page.
    pub fn tail(&mut self, len: usize) -> &mut [u8] {
        assert!(len <= PAGE);
        // SAFETY: the `len` bytes before the guard page are mapped and only lent here.
        unsafe { std::slice::from_raw_parts_mut(self.base.add(PAGE - len), len) }
    }
}

impl Drop for Guarded {
    fn drop(&mut self) {
        // SAFETY: unmaps the two pages `new` mapped; no slice of them outlives `self`.
        unsafe { munmap(self.base.cast(), 2 * PAGE) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t1_reference_compressed_on_known_examples() {
        assert_eq!(reference_compressed(&[], 3), []);
        assert_eq!(reference_compressed(&[1, 0, 0, 2], 3), [0x40, 1, 0, 0, 2]);
        assert_eq!(
            reference_compressed(&[1, 0, 0, 0, 2], 3),
            [0x13, 1, 0x10, 2]
        );
        assert_eq!(reference_compressed(&[0, 0, 0, 0, 7], 3), [0x04, 0x10, 7]);
        assert_eq!(
            reference_compressed(&[9, 0, 0], 3),
            [0x30, 9, 0, 0],
            "2 trailing zeros stay in the literal"
        );
        assert_eq!(reference_compressed(&[0; 100], 3), [0x0f, 85]);
    }

    #[test]
    fn t2_reference_compressed_at_a_minimum_of_16() {
        let run = |zero_len: usize| [&[1][..], &vec![0; zero_len], &[2]].concat();
        // 15 zeros stay in a literal of 17 bytes: nibble 15, extension 2.
        assert_eq!(
            reference_compressed(&run(15), 16),
            [&[0xf0, 2][..], &run(15)].concat()
        );
        // 16 zeros: Z = 16 = 15 + 1.
        assert_eq!(reference_compressed(&run(16), 16), [0x1f, 1, 1, 0x10, 2]);
        assert_eq!(reference_compressed(&[0; 16], 16), [0x0f, 1]);
    }

    #[test]
    fn t3_decompress_reads_back_the_reference() {
        let mut rng = Rng::new(1);
        for &min_zero_run in &MIN_ZERO_RUNS {
            for case in 0..100 {
                let src = sparse(&mut rng, 1000, min_zero_run);
                assert_eq!(
                    decompress(&reference_compressed(&src, min_zero_run)),
                    src,
                    "min {min_zero_run}, case {case}"
                );
            }
        }
    }

    #[test]
    fn t4_reference_compressed_aligned_on_the_plans_examples() {
        let (a, b) = (0x41u8, 0x42u8);
        // | A A A A A 0 0 0 | 0 x 8 | 0 x 8 | 0 B B B B B B B |: the zeros that fill no
        // word stay in the literals; L = 8 and Z = 16 = 15 + 1, then L = 8.
        let src = [&[a; 5][..], &[0; 3], &[0; 16], &[0], &[b; 7]].concat();
        let expected = [&[0x8f][..], &[a; 5], &[0; 3], &[0x01, 0x80, 0], &[b; 7]].concat();
        assert_eq!(reference_compressed_aligned(&src, 2), expected);
        // | A A A A 0 0 0 0 | 0 x 8 | 0 0 0 0 B B B B |: one zero word only, nothing left
        // out, and 24 bytes as one literal: L = 24 = 15 + 9.
        let src = [&[a; 4][..], &[0; 16], &[b; 4]].concat();
        assert_eq!(
            reference_compressed_aligned(&src, 2),
            [&[0xf0, 9][..], &src].concat()
        );
        // 100 zeros: 12 words, Z = 96 = 15 + 81, then the last 4 zeros as a literal.
        assert_eq!(
            reference_compressed_aligned(&[0; 100], 2),
            [0x0f, 81, 0x40, 0, 0, 0, 0]
        );
    }

    #[test]
    fn t5_reference_compressed_aligned_at_word_edges() {
        let (x, y) = (0x58u8, 0x59u8);
        // 23 zeros at offset 1: words 1 and 2 are left out, the 7 zeros of word 0 kept.
        let src = [&[x][..], &[0; 23], &[y; 8]].concat();
        let expected = [&[0x8f, x][..], &[0; 7], &[0x01, 0x80], &[y; 8]].concat();
        assert_eq!(reference_compressed_aligned(&src, 2), expected);
        // 16 zeros at offsets 1 to 7 of a word fill one word only: all kept.
        for offset in 1..8 {
            let src = [&vec![x; offset][..], &[0; 16], &vec![y; 8 - offset]].concat();
            assert_eq!(
                reference_compressed_aligned(&src, 2),
                [&[0xf0, 9][..], &src].concat(),
                "16 zeros at offset {offset}"
            );
        }
        // Exactly the minimum, alone and with a trailing partial word of zeros.
        assert_eq!(reference_compressed_aligned(&[0; 16], 2), [0x0f, 1]);
        assert_eq!(
            reference_compressed_aligned(&[0; 21], 2),
            [0x0f, 1, 0x50, 0, 0, 0, 0, 0]
        );
        // One word fewer than the minimum stays a literal; at a minimum of 1 it is a run.
        assert_eq!(
            reference_compressed_aligned(&[0; 8], 2),
            [0x80, 0, 0, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(reference_compressed_aligned(&[0; 8], 1), [0x08]);
        assert_eq!(reference_compressed_aligned(&[], 2), []);
    }

    #[test]
    fn t6_decompress_reads_back_the_aligned_reference() {
        let mut rng = Rng::new(6);
        for min_zero_words in [1, 2, 3, 5] {
            for case in 0..100 {
                let src = sparse(&mut rng, 1000, 8 * min_zero_words);
                assert_eq!(
                    decompress(&reference_compressed_aligned(&src, min_zero_words)),
                    src,
                    "{min_zero_words} words, case {case}"
                );
            }
            for (name, src) in aligned_edge_cases(min_zero_words) {
                assert_eq!(
                    decompress(&reference_compressed_aligned(&src, min_zero_words)),
                    src,
                    "{min_zero_words} words, {name}"
                );
            }
        }
    }
}
