//! Zero-length encoding (ZLE): every run of at least 3 zero bytes is left out of the
//! output and replaced by its length. The format is the one of
//! <https://github.com/riptl/zle>.
//!
//! The output is a sequence of blocks. A block is some literal bytes, copied as they are,
//! followed by a count of zeros left out:
//!
//! ```text
//! block  = header  [leb128(L - 15)]  L literal bytes  [leb128(Z - 15)]
//! header = min(L, 15) << 4 | min(Z, 15)
//!
//! input  01 00 00 00 02   →   13 01   10 02
//!                             └ L=1 Z=3  └ L=1 Z=0
//! ```
//!
//! If the blocks would not be shorter than the input, the output is the whole input as one
//! literal block instead, which bounds it at [`compress_bound`].
//!
//! [`compress`] and [`decompress`] use AVX-512 when the CPU has AVX-512F, AVX-512BW and BMI2, and a scalar
//! scalar code otherwise. Both write the same bytes. Which one is picked at compile time
//! when it can be: a build with those features enabled calls AVX-512 directly, a build for
//! another architecture calls the scalar compressor, and only an x86_64 build without them
//! checks the CPU at run time.
//!
//! # Examples
//!
//! ```
//! let src = [1, 0, 0, 0, 2];
//! let mut dst = vec![0u8; zle::compress_bound(src.len())];
//! let written = zle::compress(&src, &mut dst);
//! assert_eq!(dst[..written], [0x13, 1, 0x10, 2]);
//!
//! // The original length travels out of band: dst must hold at least that much.
//! let mut out = vec![0u8; src.len()];
//! assert_eq!(zle::decompress(&dst[..written], &mut out), Ok(src.len()));
//! assert_eq!(out, src);
//! ```

#[cfg(target_arch = "x86_64")]
mod avx512;
mod block;
mod decompress;
mod leb128;
mod scalar;

pub use decompress::DecompressError;

/// The implementations behind [`compress`] and [`decompress`], callable directly so they can
/// be benchmarked against each other. Not a stable API.
#[cfg(feature = "internals")]
#[doc(hidden)]
pub mod internals {
    use crate::DecompressError;

    /// The scalar compressor, on any CPU.
    ///
    /// # Arguments
    ///
    /// * `src` - The input.
    /// * `dst` - At least [`crate::compress_bound`] of the input.
    ///
    /// # Returns
    ///
    /// How many bytes were written.
    ///
    /// # Panics
    ///
    /// Panics if `dst` is shorter than [`crate::compress_bound`].
    pub fn scalar_compress(src: &[u8], dst: &mut [u8]) -> usize {
        crate::scalar::compress(src, dst)
    }

    /// The scalar decompressor, on any CPU.
    ///
    /// # Arguments
    ///
    /// * `src` - The blocks.
    /// * `dst` - Where the original bytes go.
    ///
    /// # Returns
    ///
    /// How many bytes were written.
    ///
    /// # Errors
    ///
    /// As [`crate::decompress`].
    pub fn scalar_decompress(src: &[u8], dst: &mut [u8]) -> Result<usize, DecompressError> {
        crate::decompress::decompress(src, dst)
    }

    /// Returns whether the CPU can run [`avx512_compress`] and [`avx512_decompress`].
    ///
    /// # Returns
    ///
    /// `true` if it has AVX-512F, AVX-512BW and BMI2.
    #[cfg(target_arch = "x86_64")]
    pub fn avx512_is_supported() -> bool {
        crate::avx512::is_supported()
    }

    /// The AVX-512 compressor.
    ///
    /// # Arguments
    ///
    /// * `src` - The input.
    /// * `dst` - At least [`crate::compress_bound`] of the input.
    ///
    /// # Returns
    ///
    /// How many bytes were written.
    ///
    /// # Panics
    ///
    /// Panics if `dst` is shorter than [`crate::compress_bound`].
    ///
    /// # Safety
    ///
    /// The CPU must have AVX-512F, AVX-512BW and BMI2: see [`avx512_is_supported`].
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn avx512_compress(src: &[u8], dst: &mut [u8]) -> usize {
        // SAFETY: the caller checked the CPU, as this function's contract requires.
        unsafe { crate::avx512::compress(src, dst) }
    }

    /// The AVX-512 decompressor.
    ///
    /// # Arguments
    ///
    /// * `src` - The blocks.
    /// * `dst` - Where the original bytes go.
    ///
    /// # Returns
    ///
    /// How many bytes were written.
    ///
    /// # Errors
    ///
    /// As [`crate::decompress`].
    ///
    /// # Safety
    ///
    /// The CPU must have AVX-512F, AVX-512BW and BMI2: see [`avx512_is_supported`].
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn avx512_decompress(src: &[u8], dst: &mut [u8]) -> Result<usize, DecompressError> {
        // SAFETY: the caller checked the CPU, as this function's contract requires.
        unsafe { crate::avx512::decompress(src, dst) }
    }
}

#[cfg(test)]
mod testutil;

/// The shortest zero run left out of the output. Shorter ones stay in the literal: a
/// block costs at least a header byte, so dropping 1 or 2 zeros saves nothing.
pub const MIN_ZERO_RUN: usize = 3;

/// Returns the most bytes [`compress`] writes.
///
/// # Arguments
///
/// * `src_len` - The input's length.
///
/// # Returns
///
/// `src_len + 8`: the input as one literal block, with a header byte and a length
/// extension of at most 7 bytes.
pub fn compress_bound(src_len: usize) -> usize {
    src_len + 8
}

/// Compresses bytes, with AVX-512 if the CPU has it.
///
/// # Arguments
///
/// * `src` - The input.
/// * `dst` - Where the blocks go, at least [`compress_bound`] of the input.
///
/// # Returns
///
/// How many bytes were written; 0 for an empty input.
///
/// # Panics
///
/// Panics if `dst` is shorter than [`compress_bound`] of `src.len()`.
pub fn compress(src: &[u8], dst: &mut [u8]) -> usize {
    assert!(
        dst.len() >= compress_bound(src.len()),
        "dst holds {} bytes, compress_bound is {}",
        dst.len(),
        compress_bound(src.len())
    );
    compress_dispatch(src, dst)
}

/// Built with AVX-512F, AVX-512BW and BMI2 enabled (`-C target-cpu` or
/// `-C target-feature`): every CPU this binary runs on has them, so no check is needed.
///
/// # Arguments
///
/// * `src` - The input.
/// * `dst` - Where the blocks go, at least [`compress_bound`] of the input.
///
/// # Returns
///
/// How many bytes were written.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "bmi2"
))]
#[inline(always)]
fn compress_dispatch(src: &[u8], dst: &mut [u8]) -> usize {
    // SAFETY: the cfg above means the whole build targets CPUs with every feature
    // avx512::compress enables.
    unsafe { avx512::compress(src, dst) }
}

/// x86_64 built without the AVX-512 features: checks the CPU at run time.
///
/// # Arguments
///
/// * `src` - The input.
/// * `dst` - Where the blocks go, at least [`compress_bound`] of the input.
///
/// # Returns
///
/// How many bytes were written.
#[cfg(all(
    target_arch = "x86_64",
    not(all(target_feature = "avx512f", target_feature = "avx512bw", target_feature = "bmi2"))
))]
#[inline(always)]
fn compress_dispatch(src: &[u8], dst: &mut [u8]) -> usize {
    if avx512::is_supported() {
        // SAFETY: is_supported checked the CPU has every feature avx512::compress enables.
        return unsafe { avx512::compress(src, dst) };
    }
    scalar::compress(src, dst)
}

/// Any other architecture: the scalar compressor.
///
/// # Arguments
///
/// * `src` - The input.
/// * `dst` - Where the blocks go, at least [`compress_bound`] of the input.
///
/// # Returns
///
/// How many bytes were written.
#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
fn compress_dispatch(src: &[u8], dst: &mut [u8]) -> usize {
    scalar::compress(src, dst)
}

/// Decompresses blocks written by [`compress`], with AVX-512 if the CPU has it.
///
/// # Arguments
///
/// * `src` - The blocks. Untrusted: a malformed stream is an error, never a panic.
/// * `dst` - Where the original bytes go, at least as long as the output. The format does not
///   store the original length, so the caller passes it along. Its old contents don't matter.
///
/// # Returns
///
/// How many bytes were written; 0 for an empty input.
///
/// # Errors
///
/// [`DecompressError::Truncated`] if `src` ends inside a block,
/// [`DecompressError::LengthOverflow`] if a length extension overflows 64 bits, and
/// [`DecompressError::DstTooSmall`] if the output does not fit in `dst`. On an error, `dst`
/// holds whatever was decoded before it.
pub fn decompress(src: &[u8], dst: &mut [u8]) -> Result<usize, DecompressError> {
    decompress_dispatch(src, dst)
}

/// Built with AVX-512F, AVX-512BW and BMI2 enabled: no check needed.
///
/// # Arguments
///
/// * `src` - The blocks.
/// * `dst` - Where the original bytes go.
///
/// # Returns
///
/// What [`decompress`] returns.
///
/// # Errors
///
/// What [`decompress`] returns.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "bmi2"
))]
#[inline(always)]
fn decompress_dispatch(src: &[u8], dst: &mut [u8]) -> Result<usize, DecompressError> {
    // SAFETY: the cfg above means the whole build targets CPUs with every feature
    // avx512::decompress enables.
    unsafe { avx512::decompress(src, dst) }
}

/// x86_64 built without the AVX-512 features: checks the CPU at run time.
///
/// # Arguments
///
/// * `src` - The blocks.
/// * `dst` - Where the original bytes go.
///
/// # Returns
///
/// What [`decompress`] returns.
///
/// # Errors
///
/// What [`decompress`] returns.
#[cfg(all(
    target_arch = "x86_64",
    not(all(target_feature = "avx512f", target_feature = "avx512bw", target_feature = "bmi2"))
))]
#[inline(always)]
fn decompress_dispatch(src: &[u8], dst: &mut [u8]) -> Result<usize, DecompressError> {
    if avx512::is_supported() {
        // SAFETY: is_supported checked the CPU has every feature avx512::decompress enables.
        return unsafe { avx512::decompress(src, dst) };
    }
    decompress::decompress(src, dst)
}

/// Any other architecture: the scalar decompressor.
///
/// # Arguments
///
/// * `src` - The blocks.
/// * `dst` - Where the original bytes go.
///
/// # Returns
///
/// What [`decompress`] returns.
///
/// # Errors
///
/// What [`decompress`] returns.
#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
fn decompress_dispatch(src: &[u8], dst: &mut [u8]) -> Result<usize, DecompressError> {
    decompress::decompress(src, dst)
}

/// Returns the size of the escape: the whole input as one literal block.
///
/// # Arguments
///
/// * `src_len` - The input's length.
///
/// # Returns
///
/// [`block::block_len`] of `src_len` literal bytes and no zeros. A compressor that would
/// write this many bytes or more writes the escape instead.
fn escape_len(src_len: usize) -> usize {
    block::block_len(src_len, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{Rng, decompress, sparse};

    fn compressed(src: &[u8]) -> Vec<u8> {
        let mut dst = vec![0u8; compress_bound(src.len())];
        let written = compress(src, &mut dst);
        dst.truncate(written);
        dst
    }

    #[test]
    fn t1_small_examples() {
        assert_eq!(compressed(&[]), []);
        assert_eq!(compressed(&[1, 2, 3]), [0x30, 1, 2, 3]);
        assert_eq!(compressed(&[1, 0, 0, 2]), [0x40, 1, 0, 0, 2]);
        assert_eq!(compressed(&[1, 0, 0, 0, 2]), [0x13, 1, 0x10, 2]);
        assert_eq!(compressed(&[0; 100]), [0x0f, 85]);
    }

    #[test]
    fn t2_round_trips() {
        let mut rng = Rng::new(2);
        for case in 0..300 {
            let src = sparse(&mut rng, 2000);
            let packed = compressed(&src);
            assert!(packed.len() <= escape_len(src.len()), "case {case}: longer than the escape");
            assert_eq!(decompress(&packed), src, "case {case}");
        }
    }

    #[test]
    fn t3_dst_of_exactly_compress_bound_is_enough() {
        // Incompressible input: the escape, the longest output.
        let src = vec![0x5a; 1000];
        let mut dst = vec![0u8; compress_bound(src.len())];
        assert_eq!(compress(&src, &mut dst), escape_len(src.len()));
    }

    #[test]
    #[should_panic(expected = "compress_bound")]
    fn t4_short_dst_panics() {
        let src = [1, 2, 3];
        let mut dst = vec![0u8; compress_bound(src.len()) - 1];
        compress(&src, &mut dst);
    }
}
