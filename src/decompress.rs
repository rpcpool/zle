//! The decompressor: reads blocks one at a time, copying each literal and writing its
//! zeros. The input is untrusted, so every length is checked against what is left of the
//! input and of the output before anything is copied.

use std::fmt;

use crate::leb128;

/// The largest length a header nibble holds; 15 means an extension follows.
const NIBBLE_MAX: u8 = 15;

/// Why [`decompress`] rejected its input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecompressError {
    /// The input ends inside a block: in a length extension or in the literal.
    Truncated,
    /// A length extension does not fit in a `u64`.
    LengthOverflow,
    /// The output would be longer than `dst`.
    DstTooSmall,
}

impl fmt::Display for DecompressError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::Truncated => "compressed input ends inside a block",
            Self::LengthOverflow => "a length extension overflows 64 bits",
            Self::DstTooSmall => "decompressed output is longer than dst",
        };
        f.write_str(message)
    }
}

impl std::error::Error for DecompressError {}

/// Decompresses blocks written by [`crate::compress`], copying with `memcpy` and `memset`.
///
/// # Arguments
///
/// * `src` - The blocks.
/// * `dst` - Where the original bytes go. Its old contents don't matter: zeros are
///   written, not assumed.
///
/// # Returns
///
/// How many bytes were written; 0 for an empty input.
///
/// # Errors
///
/// [`DecompressError::Truncated`] if `src` ends inside a block,
/// [`DecompressError::LengthOverflow`] if a length extension overflows 64 bits, and
/// [`DecompressError::DstTooSmall`] if the output does not fit in `dst`. On an error,
/// `dst` holds whatever was decoded before it.
pub fn decompress(src: &[u8], dst: &mut [u8]) -> Result<usize, DecompressError> {
    let mut read = 0;
    let mut written = 0;
    while read < src.len() {
        let header = src[read];
        read += 1;

        let literal_len = length(header >> 4, src, &mut read)?;
        let literal_len = fit(literal_len, dst.len() - written)?;
        let literal = src.get(read..read + literal_len).ok_or(DecompressError::Truncated)?;
        dst[written..written + literal_len].copy_from_slice(literal);
        read += literal_len;
        written += literal_len;

        let zero_len = length(header & 0x0f, src, &mut read)?;
        let zero_len = fit(zero_len, dst.len() - written)?;
        dst[written..written + zero_len].fill(0);
        written += zero_len;
    }
    Ok(written)
}

/// Reads one length: the nibble itself below 15, else 15 plus the extension at `read`.
///
/// # Arguments
///
/// * `nibble` - The header nibble, 0 to 15.
/// * `src` - The blocks.
/// * `read` - Where the extension starts; moved past it.
///
/// # Returns
///
/// The length, L or Z.
///
/// # Errors
///
/// [`DecompressError::Truncated`] if `src` ends inside the extension,
/// [`DecompressError::LengthOverflow`] if 15 plus the extension overflows a `u64`.
pub(crate) fn length(nibble: u8, src: &[u8], read: &mut usize) -> Result<u64, DecompressError> {
    if nibble < NIBBLE_MAX {
        return Ok(nibble as u64);
    }
    let extension = leb128::get(&src[*read..]).map_err(|error| match error {
        leb128::GetError::Truncated => DecompressError::Truncated,
        leb128::GetError::Overflow => DecompressError::LengthOverflow,
    })?;
    *read += extension.len;
    extension.value.checked_add(NIBBLE_MAX as u64).ok_or(DecompressError::LengthOverflow)
}

/// Checks a length against the room left in `dst`.
///
/// # Arguments
///
/// * `len` - The length read from the input.
/// * `room` - Bytes left in `dst`.
///
/// # Returns
///
/// `len` as a `usize`.
///
/// # Errors
///
/// [`DecompressError::DstTooSmall`] if `len` is more than `room`.
pub(crate) fn fit(len: u64, room: usize) -> Result<usize, DecompressError> {
    // Comparing as u64 first: on a 32-bit target, `len as usize` could wrap.
    if len > room as u64 { Err(DecompressError::DstTooSmall) } else { Ok(len as usize) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{Rng, sparse};
    use crate::{compress, compress_bound};

    fn decompressed(src: &[u8], dst_len: usize) -> Result<Vec<u8>, DecompressError> {
        // Garbage in dst: the decompressor must write the zeros itself.
        let mut dst = vec![0xcc; dst_len];
        let written = decompress(src, &mut dst)?;
        dst.truncate(written);
        Ok(dst)
    }

    fn compressed(src: &[u8]) -> Vec<u8> {
        let mut dst = vec![0u8; compress_bound(src.len())];
        let written = compress(src, &mut dst);
        dst.truncate(written);
        dst
    }

    #[test]
    fn t1_small_examples() {
        assert_eq!(decompressed(&[], 0), Ok(vec![]));
        assert_eq!(decompressed(&[0x13, 1, 0x10, 2], 5), Ok(vec![1, 0, 0, 0, 2]));
        assert_eq!(decompressed(&[0x40, 1, 0, 0, 2], 4), Ok(vec![1, 0, 0, 2]));
        assert_eq!(decompressed(&[0x0f, 85], 100), Ok(vec![0; 100]));
    }

    #[test]
    fn t2_round_trips() {
        let mut rng = Rng::new(3);
        for case in 0..300 {
            let src = sparse(&mut rng, 2000);
            assert_eq!(decompressed(&compressed(&src), src.len()), Ok(src), "case {case}");
        }
    }

    #[test]
    fn t3_dst_of_exactly_the_output_is_enough_one_less_is_not() {
        // A literal, a zero run and a long literal with its extension.
        let mut src = vec![7u8; 3];
        src.extend([0; 40]);
        src.extend([9; 30]);
        let packed = compressed(&src);
        assert_eq!(decompressed(&packed, src.len()), Ok(src.clone()));
        assert_eq!(decompressed(&packed, src.len() + 10), Ok(src.clone()), "spare room is fine");
        assert_eq!(decompressed(&packed, src.len() - 1), Err(DecompressError::DstTooSmall));
    }

    #[test]
    fn t4_dst_too_small_in_literal_and_in_zeros() {
        assert_eq!(decompressed(&[0x30, 1, 2, 3], 2), Err(DecompressError::DstTooSmall));
        assert_eq!(decompressed(&[0x05], 4), Err(DecompressError::DstTooSmall));
    }

    #[test]
    fn t5_cut_at_every_byte_is_truncated_or_a_shorter_valid_stream() {
        // Long literal and long zero run: both extensions and the literal get cut.
        let mut src = vec![5u8; 200];
        src.extend([0; 300]);
        src.push(1);
        let packed = compressed(&src);
        for cut in 0..packed.len() {
            let result = decompressed(&packed[..cut], src.len());
            // A cut on a block boundary is a valid, shorter stream; anywhere else, truncated.
            match result {
                Ok(prefix) => assert!(src.starts_with(&prefix), "cut at {cut}"),
                Err(error) => assert_eq!(error, DecompressError::Truncated, "cut at {cut}"),
            }
        }
    }

    #[test]
    fn t6_truncated_literal_and_extensions() {
        assert_eq!(decompressed(&[0x30, 1, 2], 10), Err(DecompressError::Truncated), "literal");
        assert_eq!(decompressed(&[0xf0], 100), Err(DecompressError::Truncated), "L extension");
        assert_eq!(decompressed(&[0xf0, 0x80], 100), Err(DecompressError::Truncated));
        assert_eq!(decompressed(&[0x0f], 100), Err(DecompressError::Truncated), "Z extension");
    }

    #[test]
    fn t7_length_overflow() {
        // Extension of 10 bytes with bit 64 set.
        let mut src = vec![0x0f];
        src.extend([0xff; 9]);
        src.push(0x02);
        assert_eq!(decompressed(&src, 10), Err(DecompressError::LengthOverflow));
        // Extension of u64::MAX: 15 + u64::MAX overflows.
        let mut src = vec![0x0f];
        src.extend([0xff; 9]);
        src.push(0x01);
        assert_eq!(decompressed(&src, 10), Err(DecompressError::LengthOverflow));
    }

    #[test]
    fn t8_huge_length_is_dst_too_small_not_a_panic() {
        // Z = 15 + (u64::MAX - 15): fits a u64, not any dst.
        let mut extension = [0u8; 10];
        let len = leb128::put(u64::MAX - 15, &mut extension);
        let mut src = vec![0x0f];
        src.extend(&extension[..len]);
        assert_eq!(decompressed(&src, 10), Err(DecompressError::DstTooSmall));
    }

    #[test]
    fn t9_accepts_blocks_the_compressor_never_writes() {
        // Runs of 1 zero, an empty block and a 15 nibble with a 0 extension.
        assert_eq!(decompressed(&[0x11, 7, 0x00, 0x0f, 0x00], 20), Ok([vec![7, 0], vec![0; 15]].concat()));
    }
}
