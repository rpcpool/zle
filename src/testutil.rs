//! What the tests share: the AVX-512 check, a seeded random generator, input generators, a
//! byte-at-a-time compressor and decompressor to check against, and a buffer that ends at
//! an unmapped page.

use crate::block::{block_len, write_block};
use crate::{MIN_ZERO_RUN, compress_bound, escape_len};
use std::ffi::c_void;

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
        is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("bmi2"),
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

/// Builds up to `max_len` bytes of non-zero stretches and zero runs, half of the runs short
/// (1 to 5 zeros, around the 3-zero threshold), half long (up to 200).
///
/// # Arguments
///
/// * `rng` - The generator.
/// * `max_len` - The most bytes; the length is random in `0..=max_len`.
pub fn sparse(rng: &mut Rng, max_len: usize) -> Vec<u8> {
    let len = rng.below(max_len + 1);
    let mut bytes = Vec::with_capacity(len);
    while bytes.len() < len {
        if rng.below(2) == 0 {
            let run_len = if rng.below(2) == 0 { 1 + rng.below(5) } else { 1 + rng.below(200) };
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
///
/// # Returns
///
/// The compressed bytes.
pub fn reference_compressed(src: &[u8]) -> Vec<u8> {
    let mut dst = vec![0u8; compress_bound(src.len())];
    let mut position = 0;
    let mut written = 0;
    while position < src.len() {
        let literal_start = position;
        let mut zero_len = 0;
        // Walk until 3 zeros in a row, or the end.
        while position < src.len() && zero_len < MIN_ZERO_RUN {
            zero_len = if src[position] == 0 { zero_len + 1 } else { 0 };
            position += 1;
        }
        if zero_len < MIN_ZERO_RUN {
            // The end: 0 to 2 zeros stay in the literal.
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
    fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut c_void;
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
            mmap(std::ptr::null_mut(), 2 * PAGE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)
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
        assert_eq!(reference_compressed(&[]), []);
        assert_eq!(reference_compressed(&[1, 0, 0, 2]), [0x40, 1, 0, 0, 2]);
        assert_eq!(reference_compressed(&[1, 0, 0, 0, 2]), [0x13, 1, 0x10, 2]);
        assert_eq!(reference_compressed(&[0, 0, 0, 0, 7]), [0x04, 0x10, 7]);
        assert_eq!(reference_compressed(&[9, 0, 0]), [0x30, 9, 0, 0], "2 trailing zeros stay in the literal");
        assert_eq!(reference_compressed(&[0; 100]), [0x0f, 85]);
    }

    #[test]
    fn t2_decompress_reads_back_the_reference() {
        let mut rng = Rng::new(1);
        for case in 0..300 {
            let src = sparse(&mut rng, 1000);
            assert_eq!(decompress(&reference_compressed(&src)), src, "case {case}");
        }
    }
}
