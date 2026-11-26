//! The AVX-512 decompressor. Headers are parsed as in [`crate::decompress`]; what changes is
//! the copies. A literal or zero run of at most 64 bytes, the common case, is one masked
//! load and one masked store: no `memcpy`/`memset` call, and no branch on the length for the
//! predictor to miss. Longer ones still go to `memcpy`/`memset`, which are as fast as it gets
//! at those sizes.
//!
//! Masked-off lanes are neither read nor written, and never fault, so nothing is touched
//! past the end of `src` or `dst`: no slack is needed after either.

use std::arch::x86_64::*;

use crate::decompress::{DecompressError, fit, length};

/// The bytes one vector holds.
const VECTOR_LEN: usize = 64;

/// Decompresses blocks, copying short literals and zero runs with masked vector moves.
///
/// # Arguments
///
/// * `src` - The blocks. Untrusted.
/// * `dst` - Where the original bytes go.
///
/// # Returns
///
/// How many bytes were written; the same as [`crate::decompress::decompress`] returns.
///
/// # Errors
///
/// The same as [`crate::decompress::decompress`], with the same bytes in `dst` before it.
///
/// # Safety
///
/// The CPU must have AVX-512F, AVX-512BW and BMI2: see [`super::is_supported`].
#[target_feature(enable = "avx512f,avx512bw,bmi2")]
pub fn decompress(src: &[u8], dst: &mut [u8]) -> Result<usize, DecompressError> {
    let mut read = 0;
    let mut written = 0;
    while read < src.len() {
        let header = src[read];
        read += 1;

        let literal_len = length(header >> 4, src, &mut read)?;
        let literal_len = fit(literal_len, dst.len() - written)?;
        let literal = src.get(read..read + literal_len).ok_or(DecompressError::Truncated)?;
        copy(literal, &mut dst[written..written + literal_len]);
        read += literal_len;
        written += literal_len;

        let zero_len = length(header & 0x0f, src, &mut read)?;
        let zero_len = fit(zero_len, dst.len() - written)?;
        zero(&mut dst[written..written + zero_len]);
        written += zero_len;
    }
    Ok(written)
}

/// Copies `from` into `to`, one masked move if it fits a vector.
///
/// # Arguments
///
/// * `from` - The bytes.
/// * `to` - Where they go, the same length.
#[target_feature(enable = "avx512f,avx512bw,bmi2")]
#[inline]
fn copy(from: &[u8], to: &mut [u8]) {
    debug_assert_eq!(from.len(), to.len());
    if from.len() > VECTOR_LEN {
        to.copy_from_slice(from);
        return;
    }
    let mask = low_bits(from.len());
    // SAFETY: lanes the mask clears are not accessed and cannot fault; the others are the
    // first `len` bytes of `from` and of `to`, both in bounds.
    unsafe {
        let bytes = _mm512_maskz_loadu_epi8(mask, from.as_ptr().cast());
        _mm512_mask_storeu_epi8(to.as_mut_ptr().cast(), mask, bytes);
    }
}

/// Fills `to` with zeros, one masked store if it fits a vector.
///
/// # Arguments
///
/// * `to` - The bytes to zero.
#[target_feature(enable = "avx512f,avx512bw,bmi2")]
#[inline]
fn zero(to: &mut [u8]) {
    if to.len() > VECTOR_LEN {
        to.fill(0);
        return;
    }
    // SAFETY: lanes the mask clears are not written and cannot fault; the others are the
    // `len` bytes of `to`.
    unsafe { _mm512_mask_storeu_epi8(to.as_mut_ptr().cast(), low_bits(to.len()), _mm512_setzero_si512()) };
}

/// Returns a mask of the lowest `len` bits.
///
/// # Arguments
///
/// * `len` - How many bits, `0..=64`.
///
/// # Returns
///
/// `(1 << len) - 1`, all ones for 64.
#[target_feature(enable = "bmi2")]
#[inline]
fn low_bits(len: usize) -> u64 {
    // bzhi clears the bits from `len` up; for an index of 64 or more it clears none.
    _bzhi_u64(u64::MAX, len as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decompress::decompress as scalar_decompress;
    use crate::leb128;
    use crate::testutil::{Guarded, Rng, reference_compressed, require_avx512, sparse};

    /// Runs both decoders on `src` into `dst_len` bytes of garbage, and checks they agree on
    /// the result and on every byte of `dst`, including on an error.
    fn assert_matches_scalar(src: &[u8], dst_len: usize, what: &str) -> Result<Vec<u8>, DecompressError> {
        require_avx512();
        let mut scalar_dst = vec![0xcc; dst_len];
        let scalar_result = scalar_decompress(src, &mut scalar_dst);
        let mut dst = vec![0xcc; dst_len];
        // SAFETY: require_avx512 checked the CPU has AVX-512F, BW and BMI2.
        let result = unsafe { decompress(src, &mut dst) };
        assert_eq!(result, scalar_result, "{what}: result");
        assert_eq!(dst, scalar_dst, "{what}: dst");
        result.map(|written| dst[..written].to_vec())
    }

    #[test]
    fn t1_round_trips_and_matches_scalar() {
        let mut rng = Rng::new(4);
        for case in 0..500 {
            let src = sparse(&mut rng, 2000);
            let packed = reference_compressed(&src);
            assert_eq!(assert_matches_scalar(&packed, src.len(), &format!("case {case}")), Ok(src));
        }
    }

    #[test]
    fn t2_every_length_around_the_vector_size() {
        // Literals and zero runs of 0 to 130 bytes: the masked path, exactly 64, and memcpy.
        for literal_len in 0..=130 {
            for zero_len in [0, 3, 63, 64, 65, 130] {
                let mut src = vec![0x77u8; literal_len];
                src.extend(std::iter::repeat_n(0, zero_len));
                src.push(1);
                let packed = reference_compressed(&src);
                let what = format!("{literal_len} literal, {zero_len} zeros");
                assert_eq!(assert_matches_scalar(&packed, src.len(), &what), Ok(src));
            }
        }
    }

    #[test]
    fn t3_errors_match_scalar() {
        let mut overflow = vec![0x0f];
        overflow.extend([0xff; 9]);
        overflow.push(0x02);
        let mut huge = vec![0x0f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let len = leb128::put(u64::MAX - 15, &mut huge[1..]);
        huge.truncate(1 + len);
        for (src, dst_len, expected) in [
            (vec![0x30, 1, 2], 10, DecompressError::Truncated),
            (vec![0xf0], 100, DecompressError::Truncated),
            (vec![0x0f], 100, DecompressError::Truncated),
            (vec![0x30, 1, 2, 3], 2, DecompressError::DstTooSmall),
            (vec![0x05], 4, DecompressError::DstTooSmall),
            // A valid block first: dst must hold its bytes on the error, as with scalar.
            (vec![0x23, 9, 9, 0x05], 7, DecompressError::DstTooSmall),
            (overflow, 10, DecompressError::LengthOverflow),
            (huge, 10, DecompressError::DstTooSmall),
        ] {
            assert_eq!(assert_matches_scalar(&src, dst_len, &format!("{src:02x?}")), Err(expected));
        }
    }

    #[test]
    fn t4_cut_at_every_byte_matches_scalar() {
        let mut src = vec![5u8; 100];
        src.extend([0; 300]);
        src.extend([6u8; 20]);
        src.extend([0; 10]);
        let packed = reference_compressed(&src);
        for cut in 0..packed.len() {
            let _ = assert_matches_scalar(&packed[..cut], src.len(), &format!("cut at {cut}"));
        }
    }

    #[test]
    fn t5_never_touches_past_src_or_dst() {
        // src and dst both end right before an unmapped page: one byte read or written past
        // either is SIGSEGV. Every literal and zero-run length up to 70 ends each buffer.
        require_avx512();
        let mut guarded_src = Guarded::new(0);
        let mut guarded_dst = Guarded::new(0);
        for literal_len in 0..=70 {
            for zero_len in [0, 3, 40, 64, 70] {
                let mut original = vec![0x42u8; literal_len];
                original.extend(std::iter::repeat_n(0, zero_len));
                let packed = reference_compressed(&original);
                let src = guarded_src.tail(packed.len());
                src.copy_from_slice(&packed);
                let dst = guarded_dst.tail(original.len());
                // SAFETY: require_avx512 checked the CPU has AVX-512F, BW and BMI2.
                let result = unsafe { decompress(src, dst) };
                assert_eq!(result, Ok(original.len()), "{literal_len} literal, {zero_len} zeros");
                assert_eq!(dst, &original[..]);
            }
        }
    }
}
