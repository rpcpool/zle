//! LEB128, the variable-length integers ZLE writes for lengths of 15 or more.
//!
//! A number is cut into 7-bit groups, lowest first, one group per byte. The top bit of a
//! byte says whether another byte follows (1) or this is the last (0):
//!
//! ```text
//! 300 = 0b10_0101100  →  [1|0101100] [0|0000010]  =  AC 02
//! ```

/// Returns how many bytes [`put`] writes for a value.
///
/// # Arguments
///
/// * `value` - The value.
///
/// # Returns
///
/// 1 for 0..=127, 2 for 128..=16_383, and so on: one byte per started 7 bits.
pub fn len(value: u64) -> usize {
    // A value's significant bits, at least 1 so that 0 takes a byte, rounded up to groups of 7.
    let bits = (u64::BITS - value.leading_zeros()).max(1) as usize;
    bits.div_ceil(7)
}

/// Writes a value as LEB128, the shortest way.
///
/// # Arguments
///
/// * `value` - The value.
/// * `out` - Where it goes, at least [`len`] of `value` bytes.
///
/// # Returns
///
/// How many bytes were written: [`len`] of `value`.
///
/// # Panics
///
/// Panics if `out` is shorter than [`len`] of `value`.
pub fn put(value: u64, out: &mut [u8]) -> usize {
    let mut written = 0;
    let mut rest = value;
    loop {
        let group = (rest & 0x7f) as u8;
        rest >>= 7;
        if rest == 0 {
            out[written] = group;
            return written + 1;
        }
        // The top bit says another byte follows.
        out[written] = group | 0x80;
        written += 1;
    }
}

/// A value [`get`] read.
#[derive(Debug, PartialEq, Eq)]
pub struct Decoded {
    /// The value.
    pub value: u64,
    /// How many bytes it took.
    pub len: usize,
}

/// Why [`get`] could not read a value.
#[derive(Debug, PartialEq, Eq)]
pub enum GetError {
    /// The input ended before a byte with the top bit clear.
    Truncated,
    /// The value does not fit in a `u64`.
    Overflow,
}

/// Reads a LEB128 value from the start of `input`.
///
/// # Arguments
///
/// * `input` - The bytes; the value starts at the first one, anything after it is ignored.
///
/// # Returns
///
/// The value and how many bytes it took.
///
/// # Errors
///
/// [`GetError::Truncated`] if `input` ends inside the value, [`GetError::Overflow`] if the
/// value has bits past the 64th.
pub fn get(input: &[u8]) -> Result<Decoded, GetError> {
    let mut value = 0u64;
    for (byte_index, &byte) in input.iter().enumerate() {
        let group = (byte & 0x7f) as u64;
        let shift = 7 * byte_index as u32;
        // The 10th byte holds bit 63 only; any higher bit, or an 11th byte, overflows.
        if shift >= u64::BITS || (shift == 63 && group > 1) {
            return Err(GetError::Overflow);
        }
        value |= group << shift;
        // The top bit says another byte follows.
        if byte & 0x80 == 0 {
            return Ok(Decoded {
                value,
                len: byte_index + 1,
            });
        }
    }
    Err(GetError::Truncated)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encoded(value: u64) -> Vec<u8> {
        let mut out = [0u8; 16];
        let written = put(value, &mut out);
        out[..written].to_vec()
    }

    #[test]
    fn t1_small_values_take_one_byte() {
        assert_eq!(encoded(0), [0x00]);
        assert_eq!(encoded(5), [0x05]);
        assert_eq!(encoded(127), [0x7f], "127 is the largest value 7 bits hold");
    }

    #[test]
    fn t2_300_is_ac_02() {
        assert_eq!(encoded(300), [0xac, 0x02]);
        assert_eq!(encoded(128), [0x80, 0x01]);
    }

    #[test]
    fn t3_len_grows_every_7_bits_and_matches_put() {
        for (value, bytes) in [
            (0, 1),
            (127, 1),
            (128, 2),
            (16_383, 2),
            (16_384, 3),
            ((1 << 21) - 1, 3),
            (1 << 21, 4),
            ((1 << 49) - 1, 7),
            (u64::MAX, 10),
        ] {
            assert_eq!(len(value), bytes, "len({value})");
            assert_eq!(
                encoded(value).len(),
                bytes,
                "put({value}) must write len({value}) bytes"
            );
        }
    }

    #[test]
    fn t4_put_writes_exactly_len_bytes() {
        // A buffer of exactly len bytes: one byte too many written would panic.
        for value in [0, 1, 127, 128, 300, 16_384, u64::MAX] {
            let mut out = vec![0u8; len(value)];
            assert_eq!(put(value, &mut out), out.len(), "{value}");
        }
    }

    #[test]
    fn t5_get_reads_what_put_wrote() {
        for value in [
            0,
            1,
            127,
            128,
            300,
            16_383,
            16_384,
            1 << 49,
            u64::MAX - 1,
            u64::MAX,
        ] {
            let mut bytes = encoded(value);
            let len = bytes.len();
            bytes.push(0xee); // a trailing byte get must not read
            assert_eq!(get(&bytes), Ok(Decoded { value, len }), "{value}");
        }
    }

    #[test]
    fn t6_get_cut_at_every_byte_is_truncated() {
        let bytes = encoded(u64::MAX);
        for cut in 0..bytes.len() {
            assert_eq!(get(&bytes[..cut]), Err(GetError::Truncated), "cut at {cut}");
        }
    }

    #[test]
    fn t7_get_overflow() {
        // 10th byte with bit 64 set.
        let mut bytes = vec![0xff; 9];
        bytes.push(0x02);
        assert_eq!(get(&bytes), Err(GetError::Overflow));
        // An 11th byte, even one of zeros.
        let mut bytes = vec![0x80; 10];
        bytes.push(0x00);
        assert_eq!(get(&bytes), Err(GetError::Overflow));
    }

    #[test]
    fn t8_get_accepts_non_shortest_form() {
        // 0 written in two bytes: legal LEB128, put never writes it.
        assert_eq!(get(&[0x80, 0x00]), Ok(Decoded { value: 0, len: 2 }));
    }
}
