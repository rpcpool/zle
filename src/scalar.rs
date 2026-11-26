//! The compressor without SIMD: it scans 8 bytes at a time in a `u64`.
//!
//! Every run of at least [`MIN_ZERO_RUN`] zeros ends a block, whose literal is everything
//! since the previous run. Whatever follows the last run goes out as a last block with no
//! zeros. Before writing a block, if the output would reach [`escape_len`] bytes or more,
//! everything is thrown away and the input is written as one literal block instead.

use crate::block::{block_len, write_block};
use crate::{MIN_ZERO_RUN, escape_len};

/// Compresses bytes, scanning them 8 at a time in a `u64`.
///
/// # Arguments
///
/// * `src` - The input.
/// * `dst` - Where the blocks go, at least [`crate::compress_bound`] of the input.
///
/// # Returns
///
/// How many bytes were written; 0 for an empty input.
///
/// # Panics
///
/// Panics if `dst` is shorter than [`crate::compress_bound`].
pub fn compress(src: &[u8], dst: &mut [u8]) -> usize {
    let src_len = src.len();
    // The current literal is everything from `literal_start` up to the next run.
    let mut literal_start = 0;
    // Where to look for the next zero byte.
    let mut position = 0;
    let mut written = 0;

    loop {
        let zeros_start = next_zero(src, position);
        if zeros_start == src_len {
            break;
        }
        let zeros_end = end_of_zeros(src, zeros_start);
        let zero_len = zeros_end - zeros_start;
        if zero_len >= MIN_ZERO_RUN {
            // A run: it ends the block whose literal is everything since the previous run.
            let literal = &src[literal_start..zeros_start];
            if written + block_len(literal.len(), zero_len) >= escape_len(src_len) {
                return write_block(dst, src, 0);
            }
            written += write_block(&mut dst[written..], literal, zero_len);
            literal_start = zeros_end;
        }
        // 1 or 2 zeros stay in the literal: only move past them.
        position = zeros_end;
    }

    // Whatever follows the last run is a last block with no zeros.
    if literal_start < src_len {
        let literal = &src[literal_start..];
        if written + block_len(literal.len(), 0) >= escape_len(src_len) {
            return write_block(dst, src, 0);
        }
        written += write_block(&mut dst[written..], literal, 0);
    }
    written
}

/// Reads 8 bytes as a `u64`.
///
/// # Arguments
///
/// * `src` - The bytes.
/// * `position` - Where the 8 bytes start.
///
/// # Returns
///
/// The word, little-endian: byte `position` is its lowest byte.
///
/// # Panics
///
/// Panics if fewer than 8 bytes follow `position`.
fn load_word(src: &[u8], position: usize) -> u64 {
    u64::from_le_bytes(src[position..position + 8].try_into().unwrap())
}

/// Returns which bytes of a word are zero, exactly.
///
/// # Arguments
///
/// * `word` - 8 bytes.
///
/// # Returns
///
/// The high bit of each byte set where that byte of `word` is 0, every other bit clear.
fn zero_bytes(word: u64) -> u64 {
    const LOW_7_BITS: u64 = 0x7f7f_7f7f_7f7f_7f7f;
    // (byte & 0x7f) + 0x7f sets a byte's high bit unless its low 7 bits are 0; it is at most
    // 0xfe, so no carry reaches the next byte. ORing in the byte itself covers its own high
    // bit. A byte is zero exactly where neither sets the high bit.
    !(((word & LOW_7_BITS) + LOW_7_BITS) | word | LOW_7_BITS)
}

/// Returns where the next zero byte is.
///
/// # Arguments
///
/// * `src` - The bytes.
/// * `position` - Where to start looking.
///
/// # Returns
///
/// The index of the first zero byte at or after `position`, or `src.len()` if there is none.
fn next_zero(src: &[u8], mut position: usize) -> usize {
    while position + 8 <= src.len() {
        let zeros = zero_bytes(load_word(src, position));
        if zeros != 0 {
            // Each byte is 8 bits, and the first zero byte's high bit is the lowest set bit.
            return position + zeros.trailing_zeros() as usize / 8;
        }
        position += 8;
    }
    while position < src.len() && src[position] != 0 {
        position += 1;
    }
    position
}

/// Returns where a stretch of zero bytes ends.
///
/// # Arguments
///
/// * `src` - The bytes.
/// * `position` - Where the zeros start.
///
/// # Returns
///
/// The index of the first non-zero byte at or after `position`, or `src.len()` if there
/// is none.
fn end_of_zeros(src: &[u8], mut position: usize) -> usize {
    while position + 8 <= src.len() {
        let word = load_word(src, position);
        if word != 0 {
            // The word's low zero bits, 8 per byte, are the zero bytes before the first
            // non-zero one.
            return position + word.trailing_zeros() as usize / 8;
        }
        position += 8;
    }
    while position < src.len() && src[position] == 0 {
        position += 1;
    }
    position
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compress_bound;
    use crate::testutil::{Rng, decompress, reference_compressed, sparse};

    fn compressed(src: &[u8]) -> Vec<u8> {
        let mut dst = vec![0u8; compress_bound(src.len())];
        let written = compress(src, &mut dst);
        dst.truncate(written);
        dst
    }

    #[test]
    fn t1_empty_input_writes_nothing() {
        assert_eq!(compressed(&[]), []);
    }

    #[test]
    fn t2_no_zeros_is_one_literal_block() {
        assert_eq!(compressed(&[1, 2, 3]), [0x30, 1, 2, 3]);
    }

    #[test]
    fn t3_two_zeros_stay_in_the_literal() {
        assert_eq!(compressed(&[1, 0, 0, 2]), [0x40, 1, 0, 0, 2]);
    }

    #[test]
    fn t4_three_zeros_end_a_block() {
        assert_eq!(compressed(&[1, 0, 0, 0, 2]), [0x13, 1, 0x10, 2]);
    }

    #[test]
    fn t5_leading_zeros_are_a_block_without_literal() {
        assert_eq!(compressed(&[0, 0, 0, 0, 7]), [0x04, 0x10, 7]);
    }

    #[test]
    fn t6_trailing_zeros_need_no_last_block() {
        let mut src = vec![9];
        src.extend([0; 10]);
        assert_eq!(compressed(&src), [0x1a, 9]);
    }

    #[test]
    fn t7_all_zeros() {
        // Z = 100 = 15 + 85.
        assert_eq!(compressed(&[0; 100]), [0x0f, 85]);
    }

    #[test]
    fn t8_escape_when_blocks_save_nothing() {
        // 143 literal bytes cost 1 + 2 + 143 (an extension of 2 bytes), exactly what the
        // 3 zeros after them save. The output reaches escape_len, so it escapes.
        let mut src = vec![0x11; 143];
        src.extend([0; 3]);
        src.extend([0x22; 143]);
        let out = compressed(&src);
        assert_eq!(out.len(), escape_len(src.len()));
        assert_eq!(out[..3], [0xf0, 0x92, 0x02], "one literal block of 289 bytes: 289 - 15 = 274");
        assert_eq!(out[3..], src);
    }

    #[test]
    fn t9_one_byte_less_does_not_escape() {
        // 142 literal bytes have a 1-byte extension: the run now saves one byte.
        let mut src = vec![0x11; 142];
        src.extend([0; 3]);
        src.extend([0x22; 143]);
        let out = compressed(&src);
        assert_eq!(out.len(), escape_len(src.len()) - 1);
        assert_eq!(out[..2], [0xf3, 127]);
    }

    #[test]
    fn t10_matches_the_reference_and_round_trips() {
        let mut rng = Rng::new(10);
        for case in 0..500 {
            let src = sparse(&mut rng, 2000);
            let packed = compressed(&src);
            assert_eq!(packed, reference_compressed(&src), "case {case}");
            assert_eq!(decompress(&packed), src, "case {case}");
        }
    }

    #[test]
    fn t11_matches_the_reference_on_word_edges() {
        for len in [1, 2, 7, 8, 9, 15, 16, 17, 24] {
            for start in 0..len {
                for zeros in [1, 2, 3, 4, 8, 9, 16] {
                    let mut src = vec![0x5a; len];
                    src[start..(start + zeros).min(len)].fill(0);
                    assert_eq!(compressed(&src), reference_compressed(&src), "len {len}, {zeros} zeros at {start}");
                }
            }
        }
    }

    #[test]
    fn t12_zero_bytes_flags_exactly_the_zero_bytes() {
        // Every byte value at every position, with the other bytes zero, then non-zero: a
        // borrow or carry between bytes would flag a wrong neighbour.
        for filler in [0x00u8, 0x01, 0x80, 0xff] {
            for byte_index in 0..8 {
                for value in 0..=255u8 {
                    let mut bytes = [filler; 8];
                    bytes[byte_index] = value;
                    let expected = (0..8)
                        .filter(|&index| bytes[index] == 0)
                        .fold(0u64, |mask, index| mask | 0x80 << (index * 8));
                    assert_eq!(zero_bytes(u64::from_le_bytes(bytes)), expected, "{bytes:02x?}");
                }
            }
        }
    }

    #[test]
    fn t13_next_zero_and_end_of_zeros_match_a_byte_scan() {
        let mut rng = Rng::new(13);
        for case in 0..300 {
            let src = sparse(&mut rng, 40);
            for position in 0..=src.len() {
                let next = (position..src.len()).find(|&index| src[index] == 0).unwrap_or(src.len());
                assert_eq!(next_zero(&src, position), next, "case {case}, next zero from {position}");
                let end = (position..src.len()).find(|&index| src[index] != 0).unwrap_or(src.len());
                assert_eq!(end_of_zeros(&src, position), end, "case {case}, end of zeros from {position}");
            }
        }
    }
}
