//! One ZLE block: a literal copied as it is, then a count of zeros left out.
//!
//! ```text
//! block  = header  [leb128(L - 15)]  L literal bytes  [leb128(Z - 15)]
//! header = min(L, 15) << 4 | min(Z, 15)
//! ```
//!
//! A nibble of 15 means "15 or more": the rest, `L - 15` or `Z - 15`, follows as LEB128,
//! the literal's before the literal, the zeros' after it. A nibble of 15 always has its
//! extension, even when the rest is 0.

use crate::leb128;

/// The largest length a header nibble holds; from there on, the nibble is 15 and an
/// extension holds the rest.
const NIBBLE_MAX: usize = 15;

/// Returns a block's header byte.
///
/// # Arguments
///
/// * `literal_len` - The literal's length, L.
/// * `zero_len` - The zeros after it, Z.
///
/// # Returns
///
/// `min(L, 15)` in the high nibble, `min(Z, 15)` in the low one.
fn header(literal_len: usize, zero_len: usize) -> u8 {
    (literal_len.min(NIBBLE_MAX) << 4 | zero_len.min(NIBBLE_MAX)) as u8
}

/// Returns how many bytes a length's extension takes.
///
/// # Arguments
///
/// * `len` - L or Z.
///
/// # Returns
///
/// 0 below 15 (the nibble holds it), else the LEB128 length of `len - 15`.
fn extension_len(len: usize) -> usize {
    if len < NIBBLE_MAX { 0 } else { leb128::len((len - NIBBLE_MAX) as u64) }
}

/// Returns how many bytes [`write_block`] writes.
///
/// # Arguments
///
/// * `literal_len` - L.
/// * `zero_len` - Z.
///
/// # Returns
///
/// The header, both extensions and the literal.
pub fn block_len(literal_len: usize, zero_len: usize) -> usize {
    1 + extension_len(literal_len) + literal_len + extension_len(zero_len)
}

/// Writes one block.
///
/// # Arguments
///
/// * `out` - Where it goes, at least [`block_len`] bytes.
/// * `literal` - The literal bytes.
/// * `zero_len` - The zeros that follow them.
///
/// # Returns
///
/// How many bytes were written: [`block_len`].
///
/// # Panics
///
/// Panics if `out` is shorter than [`block_len`].
pub fn write_block(out: &mut [u8], literal: &[u8], zero_len: usize) -> usize {
    out[0] = header(literal.len(), zero_len);
    let mut written = 1;
    if literal.len() >= NIBBLE_MAX {
        written += leb128::put((literal.len() - NIBBLE_MAX) as u64, &mut out[written..]);
    }
    out[written..written + literal.len()].copy_from_slice(literal);
    written += literal.len();
    if zero_len >= NIBBLE_MAX {
        written += leb128::put((zero_len - NIBBLE_MAX) as u64, &mut out[written..]);
    }
    written
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(literal: &[u8], zero_len: usize) -> Vec<u8> {
        let mut out = vec![0u8; literal.len() + 32];
        let written = write_block(&mut out, literal, zero_len);
        out.truncate(written);
        out
    }

    #[test]
    fn t1_header_packs_two_nibbles() {
        assert_eq!(header(2, 5), 0x25);
        assert_eq!(header(1, 0), 0x10, "a literal with no zeros after it");
        assert_eq!(header(0, 3), 0x03, "zeros with no literal before them");
    }

    #[test]
    fn t2_header_saturates_at_15() {
        assert_eq!(header(15, 0), 0xf0);
        assert_eq!(header(200, 40), 0xff, "15 means: 15 or more, see the extension");
    }

    #[test]
    fn t3_extension_only_from_15() {
        assert_eq!(extension_len(14), 0);
        assert_eq!(extension_len(15), 1, "15 has an extension of 0, one byte");
        assert_eq!(extension_len(15 + 127), 1);
        assert_eq!(extension_len(15 + 128), 2);
    }

    #[test]
    fn t4_small_block() {
        assert_eq!(block(&[0x41, 0x42], 5), [0x25, 0x41, 0x42]);
        assert_eq!(block(&[], 3), [0x03]);
    }

    #[test]
    fn t5_long_literal_has_its_extension_before_the_literal() {
        let literal = [0xaa; 20];
        let mut expected = vec![0xf3, 20 - 15];
        expected.extend(literal);
        assert_eq!(block(&literal, 3), expected);
    }

    #[test]
    fn t6_long_zero_run_has_its_extension_after_the_literal() {
        // 300 - 15 = 285 = 0x9d 0x02 in LEB128.
        assert_eq!(block(&[0xaa], 300), [0x1f, 0xaa, 0x9d, 0x02]);
    }

    #[test]
    fn t7_exactly_15_still_writes_an_extension_of_0() {
        let literal = [0x11; 15];
        let mut expected = vec![0xff, 0x00];
        expected.extend(literal);
        expected.push(0x00);
        assert_eq!(block(&literal, 15), expected);
    }

    #[test]
    fn t8_block_len_is_what_write_block_writes() {
        for literal_len in [0, 1, 14, 15, 16, 142, 143, 200] {
            for zero_len in [0, 3, 14, 15, 142, 143, 100_000] {
                let literal = vec![0x55; literal_len];
                let mut out = vec![0u8; block_len(literal_len, zero_len)];
                assert_eq!(
                    write_block(&mut out, &literal, zero_len),
                    out.len(),
                    "L = {literal_len}, Z = {zero_len}: a buffer of exactly block_len is enough"
                );
            }
        }
    }
}
