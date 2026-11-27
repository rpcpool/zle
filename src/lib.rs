//! Zero-length encoding (ZLE): leaves runs of zero bytes out of a byte buffer, and puts them
//! back. Lossless, no allocation, no dependencies. The format is the one of
//! <https://github.com/riptl/zle>.
//!
//! ```
//! let src = [1, 0, 0, 0, 2];
//! let mut dst = vec![0u8; zle::compress_bound(src.len())];
//! let written = zle::compress(&src, &mut dst);
//! assert_eq!(dst[..written], [0x13, 1, 0x10, 2]);
//!
//! // The format does not store the original length: send it along with the blocks.
//! let mut out = vec![0u8; src.len()];
//! assert_eq!(zle::decompress(&dst[..written], &mut out), Ok(src.len()));
//! assert_eq!(out, src);
//! ```
//!
//! # Which function to call
//!
//! | goal | call |
//! |---|---|
//! | smallest output | [`compress`]: every run of 3 or more zeros |
//! | fast compress and decompress, output a little larger | [`compress_with_min_run`]`(src, dst, 16)` |
//! | Solana account data, highest speed | [`compress_aligned`]`(src, dst, 2)` on a CPU with AVX-512 (see below) |
//! | decompress any of them | [`decompress`] |
//!
//! # Contract
//!
//! - **Compress into `dst` of at least [`compress_bound`]`(src.len())` bytes,** or the call
//!   panics. The return value is how many bytes of `dst` were written; send `&dst[..written]`.
//! - **Keep the original length yourself.** The output does not store it. Pass
//!   [`decompress`] a `dst` at least that long.
//! - **Check that [`decompress`] returned the original length.** A stream cut between two
//!   blocks decodes without error to a shorter output; a stream cut inside a block, or one
//!   that decodes to more than `dst` holds, is a [`DecompressError`].
//! - **[`decompress`] never panics on its input:** it treats it as untrusted.
//! - **Outputs can be concatenated.** Two compressed streams back to back decompress to the
//!   two inputs back to back, so a message can be compressed in parts into one buffer.
//! - **Every compressor's output decompresses with [`decompress`],** whatever the minimum.
//! - **The compressors panic** only on a short `dst` or a minimum below [`MIN_ZERO_RUN`] or
//!   [`MIN_ZERO_WORDS`].
//!
//! # Solana account data
//!
//! For the highest speed on account data, use [`compress_aligned`]`(src, dst, 2)` on a CPU
//! with AVX-512, ideally in a build with `RUSTFLAGS="-C target-cpu=x86-64-v4"`. Measured on
//! synthetic account-like data (dense bytes with short zero runs every few bytes, and runs of
//! 16 to 512 zeros) on a Xeon Platinum 8481C:
//!
//! | call | output | compress | decompress |
//! |---|---|---|---|
//! | [`compress_aligned`]`(src, dst, 2)` | 68.7% | 21.9 GiB/s | 24.7 GiB/s |
//! | [`compress_with_min_run`]`(src, dst, 16)` | 67.8% | 22.6 GiB/s | 24.1 GiB/s |
//! | [`compress`] | 59.8% | 1.2 GiB/s | 2.9 GiB/s |
//!
//! [`compress_aligned`] decompresses fastest, and without AVX-512 it also compresses fastest
//! (13.9 GiB/s against 9.3); with AVX-512 it compresses within 3% of
//! [`compress_with_min_run`]. Use [`compress`] only when the size matters more than speed.
//! More numbers in the [README](https://github.com/rpcpool/zle#readme).
//!
//! # CPU support
//!
//! **Run on a CPU with AVX-512, whichever function you pick.** Compress plus decompress is
//! faster with AVX-512 for every recommended call: 1.07× for [`compress`], 1.7× for
//! [`compress_with_min_run`]`(src, dst, 16)`, 1.3× for [`compress_aligned`]`(src, dst, 2)`.
//! The one step where it is not ahead: on data shaped like Solana accounts, [`compress`]
//! alone compresses about 10% slower with AVX-512, but its output decompresses 1.5× faster.
//! On data whose zeros come in fewer, longer runs, AVX-512 compresses faster too.
//!
//! Every function uses AVX-512 by itself when the CPU has AVX-512F, AVX-512BW and BMI2, and
//! scalar code otherwise; both write the same bytes. A build with those features enabled
//! (`RUSTFLAGS="-C target-cpu=x86-64-v4"`) calls AVX-512 directly. Other x86_64 builds check
//! the CPU once per call, and other architectures use the scalar code.
//!
//! # Format
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

#[cfg(target_arch = "x86_64")]
mod avx512;
mod block;
mod decompress;
mod leb128;
mod runs;
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
    /// * `min_zero_run` - As [`crate::compress_with_min_run`].
    ///
    /// # Returns
    ///
    /// How many bytes were written.
    ///
    /// # Panics
    ///
    /// Panics if `dst` is shorter than [`crate::compress_bound`], or `min_zero_run` is below
    /// [`crate::MIN_ZERO_RUN`].
    pub fn scalar_compress(src: &[u8], dst: &mut [u8], min_zero_run: usize) -> usize {
        crate::scalar::compress(src, dst, min_zero_run)
    }

    /// The scalar aligned compressor, on any CPU.
    ///
    /// # Arguments
    ///
    /// * `src` - The input.
    /// * `dst` - At least [`crate::compress_bound`] of the input.
    /// * `min_zero_words` - As [`crate::compress_aligned`].
    ///
    /// # Returns
    ///
    /// How many bytes were written.
    ///
    /// # Panics
    ///
    /// Panics if `dst` is shorter than [`crate::compress_bound`], or `min_zero_words` is
    /// below [`crate::MIN_ZERO_WORDS`].
    pub fn scalar_compress_aligned(src: &[u8], dst: &mut [u8], min_zero_words: usize) -> usize {
        crate::scalar::compress_aligned(src, dst, min_zero_words)
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
    /// * `min_zero_run` - As [`crate::compress_with_min_run`].
    ///
    /// # Returns
    ///
    /// How many bytes were written.
    ///
    /// # Panics
    ///
    /// Panics if `dst` is shorter than [`crate::compress_bound`], or `min_zero_run` is below
    /// [`crate::MIN_ZERO_RUN`].
    ///
    /// # Safety
    ///
    /// The CPU must have AVX-512F, AVX-512BW and BMI2: see [`avx512_is_supported`].
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn avx512_compress(src: &[u8], dst: &mut [u8], min_zero_run: usize) -> usize {
        // SAFETY: the caller checked the CPU, as this function's contract requires.
        unsafe { crate::avx512::compress(src, dst, min_zero_run) }
    }

    /// The AVX-512 aligned compressor.
    ///
    /// # Arguments
    ///
    /// * `src` - The input.
    /// * `dst` - At least [`crate::compress_bound`] of the input.
    /// * `min_zero_words` - As [`crate::compress_aligned`].
    ///
    /// # Returns
    ///
    /// How many bytes were written.
    ///
    /// # Panics
    ///
    /// Panics if `dst` is shorter than [`crate::compress_bound`], or `min_zero_words` is
    /// below [`crate::MIN_ZERO_WORDS`].
    ///
    /// # Safety
    ///
    /// The CPU must have AVX-512F, AVX-512BW and BMI2: see [`avx512_is_supported`].
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn avx512_compress_aligned(
        src: &[u8],
        dst: &mut [u8],
        min_zero_words: usize,
    ) -> usize {
        // SAFETY: the caller checked the CPU, as this function's contract requires.
        unsafe { crate::avx512::compress_aligned(src, dst, min_zero_words) }
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

/// The smallest `min_zero_run` [`compress_with_min_run`] accepts, and the one [`compress`]
/// uses: 3 zeros. A run of 1 or 2 zeros costs as much to encode as to keep.
pub const MIN_ZERO_RUN: usize = 3;

/// The smallest `min_zero_words` [`compress_aligned`] accepts: 2 words, 16 bytes. For runs
/// of one word, use [`compress_with_min_run`]`(src, dst, 8)`, which leaves out more.
pub const MIN_ZERO_WORDS: usize = 2;

/// Returns how large `dst` must be for any compressor of this crate.
///
/// Allocate `dst` with this size before calling [`compress`], [`compress_with_min_run`] or
/// [`compress_aligned`]; a shorter `dst` makes them panic. The output is never longer.
///
/// # Arguments
///
/// * `src_len` - The length of the input to compress.
///
/// # Returns
///
/// `src_len + 8`: the input as one literal block, with a header byte and a length extension
/// of at most 7 bytes.
///
/// # Examples
///
/// ```
/// let src = [0u8; 100];
/// let mut dst = vec![0u8; zle::compress_bound(src.len())];
/// let written = zle::compress(&src, &mut dst);
/// assert!(written <= zle::compress_bound(src.len()));
/// ```
pub fn compress_bound(src_len: usize) -> usize {
    src_len + 8
}

/// Compresses `src` into `dst` for the smallest output: every run of 3 or more zeros is left
/// out.
///
/// Use it when size matters more than speed. On data full of short zero runs it is the
/// slowest compressor here, and its output the slowest to decompress: see
/// [`compress_with_min_run`] and [`compress_aligned`]. It is
/// [`compress_with_min_run`]`(src, dst, `[`MIN_ZERO_RUN`]`)`.
///
/// # Arguments
///
/// * `src` - The bytes to compress.
/// * `dst` - Where the compressed bytes go: at least [`compress_bound`]`(src.len())` bytes.
///
/// # Returns
///
/// How many bytes were written to `dst`; 0 for an empty `src`. The output is
/// `&dst[..written]`.
///
/// # Panics
///
/// Panics if `dst` is shorter than [`compress_bound`]`(src.len())`.
///
/// # Examples
///
/// ```
/// let src = [1, 0, 0, 0, 2];
/// let mut dst = vec![0u8; zle::compress_bound(src.len())];
/// let written = zle::compress(&src, &mut dst);
/// assert_eq!(dst[..written], [0x13, 1, 0x10, 2]);
/// ```
pub fn compress(src: &[u8], dst: &mut [u8]) -> usize {
    compress_with_min_run(src, dst, MIN_ZERO_RUN)
}

/// Compresses `src` into `dst`, leaving out only runs of at least `min_zero_run` zeros.
///
/// Use it to trade a little size for speed: shorter runs stay in the output as they are,
/// so there are fewer, longer blocks. `min_zero_run = 16` is the usual choice. Compress speed
/// steps up at minimums of 7 and 15, and stays almost flat in between: prefer 16 to anything
/// from 7 to 14. For Solana account data, see [`compress_aligned`].
///
/// # Arguments
///
/// * `src` - The bytes to compress.
/// * `dst` - Where the compressed bytes go: at least [`compress_bound`]`(src.len())` bytes.
/// * `min_zero_run` - The shortest run of zeros left out, at least [`MIN_ZERO_RUN`].
///
/// # Returns
///
/// How many bytes were written to `dst`; 0 for an empty `src`. The output is
/// `&dst[..written]`.
///
/// # Panics
///
/// Panics if `dst` is shorter than [`compress_bound`]`(src.len())`, or `min_zero_run` is
/// below [`MIN_ZERO_RUN`].
///
/// # Examples
///
/// ```
/// // 1, 16 zeros, 2: at a minimum of 16, the zeros are left out.
/// let mut src = vec![1];
/// src.extend([0; 16]);
/// src.push(2);
/// let mut dst = vec![0u8; zle::compress_bound(src.len())];
/// let written = zle::compress_with_min_run(&src, &mut dst, 16);
/// assert_eq!(dst[..written], [0x1f, 1, 1, 0x10, 2]);
///
/// let mut out = vec![0u8; src.len()];
/// assert_eq!(zle::decompress(&dst[..written], &mut out), Ok(src.len()));
/// assert_eq!(out, src);
///
/// // At 17 they stay in the output: the input as one block, header and extension first.
/// assert_eq!(zle::compress_with_min_run(&src, &mut dst, 17), 2 + src.len());
/// ```
pub fn compress_with_min_run(src: &[u8], dst: &mut [u8], min_zero_run: usize) -> usize {
    assert!(
        dst.len() >= compress_bound(src.len()),
        "dst holds {} bytes, compress_bound is {}",
        dst.len(),
        compress_bound(src.len())
    );
    compress_dispatch(src, dst, min_zero_run)
}

/// Compresses `src` into `dst`, leaving out only runs of at least `min_zero_words` whole
/// 8-byte words of zeros, aligned to the start of `src`.
///
/// Use it for the highest speed on Solana account data: `min_zero_words = 2`, on a CPU with
/// AVX-512. Its output decompresses fastest, and without AVX-512 it is also the fastest to
/// compress; it is a little larger than [`compress_with_min_run`]`(src, dst, 16)`.
///
/// A word is the 8 bytes at `8 * k`, for every `k` with `8 * k + 8 <= src.len()`. A run
/// starts and ends on word boundaries: zeros next to it that do not fill a whole word stay in
/// the output, and so do the last `src.len() % 8` bytes.
///
/// # Arguments
///
/// * `src` - The bytes to compress.
/// * `dst` - Where the compressed bytes go: at least [`compress_bound`]`(src.len())` bytes.
/// * `min_zero_words` - The fewest whole zero words a run has, at least [`MIN_ZERO_WORDS`].
///
/// # Returns
///
/// How many bytes were written to `dst`; 0 for an empty `src`. The output is
/// `&dst[..written]`.
///
/// # Panics
///
/// Panics if `dst` is shorter than [`compress_bound`]`(src.len())`, or `min_zero_words` is
/// below [`MIN_ZERO_WORDS`].
///
/// # Examples
///
/// ```
/// // | 1 1 1 1 1 0 0 0 | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 | 0 2 2 2 2 2 2 2 |
/// let src = [&[1; 5][..], &[0; 20], &[2; 7]].concat();
/// let mut dst = vec![0u8; zle::compress_bound(src.len())];
///
/// // The two whole zero words are left out; the 4 zeros around them stay in the output.
/// let written = zle::compress_aligned(&src, &mut dst, 2);
/// assert_eq!(
///     dst[..written],
///     [0x8f, 1, 1, 1, 1, 1, 0, 0, 0, 0x01, 0x80, 0, 2, 2, 2, 2, 2, 2, 2]
/// );
///
/// let mut out = vec![0u8; src.len()];
/// assert_eq!(zle::decompress(&dst[..written], &mut out), Ok(src.len()));
/// assert_eq!(out, src);
/// ```
pub fn compress_aligned(src: &[u8], dst: &mut [u8], min_zero_words: usize) -> usize {
    assert!(
        dst.len() >= compress_bound(src.len()),
        "dst holds {} bytes, compress_bound is {}",
        dst.len(),
        compress_bound(src.len())
    );
    compress_aligned_dispatch(src, dst, min_zero_words)
}

/// Built with AVX-512F, AVX-512BW and BMI2 enabled (`-C target-cpu` or
/// `-C target-feature`): every CPU this binary runs on has them, so no check is needed.
///
/// # Arguments
///
/// * `src` - The input.
/// * `dst` - Where the blocks go, at least [`compress_bound`] of the input.
/// * `min_zero_run` - The shortest run left out.
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
fn compress_dispatch(src: &[u8], dst: &mut [u8], min_zero_run: usize) -> usize {
    // SAFETY: the cfg above means the whole build targets CPUs with every feature
    // avx512::compress enables.
    unsafe { avx512::compress(src, dst, min_zero_run) }
}

/// x86_64 built without the AVX-512 features: checks the CPU at run time.
///
/// # Arguments
///
/// * `src` - The input.
/// * `dst` - Where the blocks go, at least [`compress_bound`] of the input.
/// * `min_zero_run` - The shortest run left out.
///
/// # Returns
///
/// How many bytes were written.
#[cfg(all(
    target_arch = "x86_64",
    not(all(
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "bmi2"
    ))
))]
#[inline(always)]
fn compress_dispatch(src: &[u8], dst: &mut [u8], min_zero_run: usize) -> usize {
    if avx512::is_supported() {
        // SAFETY: is_supported checked the CPU has every feature avx512::compress enables.
        return unsafe { avx512::compress(src, dst, min_zero_run) };
    }
    scalar::compress(src, dst, min_zero_run)
}

/// Any other architecture: the scalar compressor.
///
/// # Arguments
///
/// * `src` - The input.
/// * `dst` - Where the blocks go, at least [`compress_bound`] of the input.
/// * `min_zero_run` - The shortest run left out.
///
/// # Returns
///
/// How many bytes were written.
#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
fn compress_dispatch(src: &[u8], dst: &mut [u8], min_zero_run: usize) -> usize {
    scalar::compress(src, dst, min_zero_run)
}

/// [`compress_aligned`], built with AVX-512F, AVX-512BW and BMI2 enabled: no check needed.
///
/// # Arguments
///
/// * `src` - The input.
/// * `dst` - Where the blocks go, at least [`compress_bound`] of the input.
/// * `min_zero_words` - The fewest zero words a run has.
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
fn compress_aligned_dispatch(src: &[u8], dst: &mut [u8], min_zero_words: usize) -> usize {
    // SAFETY: the cfg above means the whole build targets CPUs with every feature
    // avx512::compress_aligned enables.
    unsafe { avx512::compress_aligned(src, dst, min_zero_words) }
}

/// [`compress_aligned`], x86_64 built without the AVX-512 features: checks the CPU at run
/// time.
///
/// # Arguments
///
/// * `src` - The input.
/// * `dst` - Where the blocks go, at least [`compress_bound`] of the input.
/// * `min_zero_words` - The fewest zero words a run has.
///
/// # Returns
///
/// How many bytes were written.
#[cfg(all(
    target_arch = "x86_64",
    not(all(
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "bmi2"
    ))
))]
#[inline(always)]
fn compress_aligned_dispatch(src: &[u8], dst: &mut [u8], min_zero_words: usize) -> usize {
    if avx512::is_supported() {
        // SAFETY: is_supported checked the CPU has every feature avx512::compress_aligned
        // enables.
        return unsafe { avx512::compress_aligned(src, dst, min_zero_words) };
    }
    scalar::compress_aligned(src, dst, min_zero_words)
}

/// [`compress_aligned`] on any other architecture: the scalar compressor.
///
/// # Arguments
///
/// * `src` - The input.
/// * `dst` - Where the blocks go, at least [`compress_bound`] of the input.
/// * `min_zero_words` - The fewest zero words a run has.
///
/// # Returns
///
/// How many bytes were written.
#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
fn compress_aligned_dispatch(src: &[u8], dst: &mut [u8], min_zero_words: usize) -> usize {
    scalar::compress_aligned(src, dst, min_zero_words)
}

/// Decompresses `src` into `dst`: the output of any compressor of this crate.
///
/// `src` is untrusted: a malformed stream is an error, never a panic. The format does not
/// store the original length, so the caller keeps it, sizes `dst` with it, and checks the
/// returned length against it: a stream cut between two blocks decodes without error to a
/// shorter output.
///
/// # Arguments
///
/// * `src` - The compressed bytes, or several compressed streams back to back.
/// * `dst` - Where the original bytes go: at least the original length. Its old contents
///   don't matter.
///
/// # Returns
///
/// How many bytes were written to `dst`: the original length for a whole stream; 0 for an
/// empty `src`.
///
/// # Errors
///
/// * [`DecompressError::Truncated`] - `src` ends inside a block.
/// * [`DecompressError::LengthOverflow`] - a length in `src` overflows 64 bits.
/// * [`DecompressError::DstTooSmall`] - the output does not fit in `dst`.
///
/// On an error, `dst` holds whatever was decoded before it.
///
/// # Examples
///
/// ```
/// let src = [1, 0, 0, 0, 2];
/// let mut packed = vec![0u8; zle::compress_bound(src.len())];
/// let written = zle::compress(&src, &mut packed);
///
/// let mut out = vec![0u8; src.len()];
/// let out_len = zle::decompress(&packed[..written], &mut out)?;
/// assert_eq!(out_len, src.len(), "the stream was cut between two blocks");
/// assert_eq!(out, src);
/// # Ok::<(), zle::DecompressError>(())
/// ```
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
    not(all(
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "bmi2"
    ))
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
    use crate::testutil::{
        MIN_ZERO_RUNS, Rng, aligned_edge_cases, decompress, reference_compressed,
        reference_compressed_aligned, sparse,
    };

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
            let src = sparse(&mut rng, 2000, MIN_ZERO_RUN);
            let packed = compressed(&src);
            assert!(
                packed.len() <= escape_len(src.len()),
                "case {case}: longer than the escape"
            );
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

    #[test]
    #[should_panic(expected = "the smallest is 3")]
    fn t5_a_minimum_below_3_panics() {
        let src = [1, 0, 0, 2];
        let mut dst = vec![0u8; compress_bound(src.len())];
        compress_with_min_run(&src, &mut dst, 2);
    }

    #[test]
    fn t6_every_minimum_matches_the_reference_and_round_trips() {
        let mut rng = Rng::new(6);
        for &min_zero_run in &MIN_ZERO_RUNS {
            for case in 0..100 {
                let src = sparse(&mut rng, 2000, min_zero_run);
                let mut dst = vec![0u8; compress_bound(src.len())];
                let written = compress_with_min_run(&src, &mut dst, min_zero_run);
                assert_eq!(
                    dst[..written],
                    reference_compressed(&src, min_zero_run),
                    "min {min_zero_run}, case {case}"
                );
                assert_eq!(
                    decompress(&dst[..written]),
                    src,
                    "min {min_zero_run}, case {case}"
                );
            }
        }
    }

    fn compressed_aligned(src: &[u8], min_zero_words: usize) -> Vec<u8> {
        let mut dst = vec![0u8; compress_bound(src.len())];
        let written = compress_aligned(src, &mut dst, min_zero_words);
        dst.truncate(written);
        dst
    }

    #[test]
    fn t7_aligned_matches_the_reference_and_round_trips() {
        let mut rng = Rng::new(7);
        for min_zero_words in [MIN_ZERO_WORDS, 3, 5] {
            for case in 0..100 {
                let src = sparse(&mut rng, 2000, 8 * min_zero_words);
                let packed = compressed_aligned(&src, min_zero_words);
                assert_eq!(
                    packed,
                    reference_compressed_aligned(&src, min_zero_words),
                    "{min_zero_words} words, case {case}"
                );
                assert_eq!(
                    decompress(&packed),
                    src,
                    "{min_zero_words} words, case {case}"
                );
            }
            for (name, src) in aligned_edge_cases(min_zero_words) {
                assert_eq!(
                    compressed_aligned(&src, min_zero_words),
                    reference_compressed_aligned(&src, min_zero_words),
                    "{min_zero_words} words, {name}"
                );
            }
        }
    }

    #[test]
    fn t8_aligned_escape_when_blocks_save_nothing() {
        // Runs of 15 zeros fill at most one word each: nothing is left out, and the input
        // goes out as one literal block.
        let mut src = Vec::new();
        for offset in 0..8 {
            src.extend(std::iter::repeat_n(0x11, 9 + offset));
            src.extend([0; 15]);
        }
        let packed = compressed_aligned(&src, MIN_ZERO_WORDS);
        assert_eq!(packed.len(), escape_len(src.len()));
        assert_eq!(packed[packed.len() - src.len()..], src);
    }

    #[test]
    fn t9_aligned_dst_of_exactly_compress_bound_is_enough() {
        let src = vec![0x5a; 1000];
        let mut dst = vec![0u8; compress_bound(src.len())];
        assert_eq!(
            compress_aligned(&src, &mut dst, MIN_ZERO_WORDS),
            escape_len(src.len())
        );
    }

    #[test]
    #[should_panic(expected = "compress_bound")]
    fn t10_aligned_short_dst_panics() {
        let src = [0u8; 40];
        let mut dst = vec![0u8; compress_bound(src.len()) - 1];
        compress_aligned(&src, &mut dst, MIN_ZERO_WORDS);
    }

    #[test]
    #[should_panic(expected = "the smallest is 2")]
    fn t11_aligned_minimum_of_1_panics() {
        let src = [0u8; 40];
        let mut dst = vec![0u8; compress_bound(src.len())];
        compress_aligned(&src, &mut dst, 1);
    }
}
