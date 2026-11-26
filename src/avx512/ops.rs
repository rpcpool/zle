//! The vector operations the AVX-512 compressor is built from.

use std::arch::x86_64::*;

/// Returns which of at most 64 bytes are zero, reading none past the slice.
///
/// # Arguments
///
/// * `bytes` - At most 64 bytes.
///
/// # Returns
///
/// A mask with bit `i` set where `bytes[i] == 0`; bits from `bytes.len()` up are clear.
///
/// # Panics
///
/// Panics if `bytes` is longer than 64.
#[target_feature(enable = "avx512f,avx512bw,bmi2")]
pub fn zero_mask_tail(bytes: &[u8]) -> u64 {
    assert!(bytes.len() <= 64);
    // The first `bytes.len()` lanes. Unlike `(1 << n) - 1`, bzhi works for n = 64.
    let in_slice = _bzhi_u64(u64::MAX, bytes.len() as u32);
    // SAFETY: a masked load reads only the selected lanes, all inside the slice; the others
    // are never touched, so they can't fault.
    let loaded = unsafe { _mm512_maskz_loadu_epi8(in_slice, bytes.as_ptr().cast()) };
    // The unselected lanes were loaded as zeros: clear their bits.
    _mm512_cmpeq_epi8_mask(loaded, _mm512_setzero_si512()) & in_slice
}

/// Returns whether 256 bytes are all zero.
///
/// # Arguments
///
/// * `block` - The 256 bytes.
///
/// # Returns
///
/// `true` if every byte is 0.
#[target_feature(enable = "avx512f,avx512bw")]
pub fn is_zero_256(block: &[u8; 256]) -> bool {
    let vectors: [__m512i; 4] = std::array::from_fn(|vector_index| {
        // SAFETY: the block holds 256 bytes, so the 64 at vector_index * 64 are readable.
        unsafe { _mm512_loadu_si512(block.as_ptr().add(vector_index * 64).cast()) }
    });
    // A byte of the OR is zero only where it is zero in all four.
    let ored = or3(vectors[0], vectors[1], _mm512_or_si512(vectors[2], vectors[3]));
    _mm512_test_epi8_mask(ored, ored) == 0
}

/// Returns whether a run of 3 zero bytes starts in the first 256 bytes of a window.
///
/// # Arguments
///
/// * `window` - 320 bytes: the 256 to search, then 64 more for runs that start at the end.
///
/// # Returns
///
/// `true` if some `i < 256` has `window[i..i + 3]` all zero.
#[target_feature(enable = "avx512f,avx512bw")]
pub fn has_zero_run_256(window: &[u8; 320]) -> bool {
    // The 5 vectors of the window: 4 to search, and the 5th for runs starting at 254, 255.
    let vectors: [__m512i; 5] = std::array::from_fn(|vector_index| {
        // SAFETY: the window holds 320 bytes, so the 64 at vector_index * 64 are readable.
        unsafe { _mm512_loadu_si512(window.as_ptr().add(vector_index * 64).cast()) }
    });
    // Byte i is zero where bytes i, i+1 and i+2 of the window are: a run starts at i.
    let starts = |vector_index: usize| {
        let (lo, hi) = (vectors[vector_index], vectors[vector_index + 1]);
        or3(lo, shift_bytes::<1>(lo, hi), shift_bytes::<2>(lo, hi))
    };
    // The unsigned minimum per byte is zero where either is zero.
    let merged = _mm512_min_epu8(_mm512_min_epu8(starts(0), starts(1)), _mm512_min_epu8(starts(2), starts(3)));
    _mm512_testn_epi8_mask(merged, merged) != 0
}

/// Returns `a | b | c`, in one instruction.
///
/// # Arguments
///
/// * `a`, `b`, `c` - The vectors.
///
/// # Returns
///
/// Their bitwise OR.
#[target_feature(enable = "avx512f")]
fn or3(a: __m512i, b: __m512i, c: __m512i) -> __m512i {
    // vpternlogd computes, per bit, the function whose truth table is the immediate: bit
    // (a << 2 | b << 1 | c) of it is the result. OR is 1 everywhere but 0 0 0: 0xFE.
    _mm512_ternarylogic_epi32::<0xFE>(a, b, c)
}

/// Returns bytes `K..K + 64` of the 128 bytes `lo` then `hi`.
///
/// # Arguments
///
/// * `K` - The shift, in bytes, `0..=15`.
/// * `lo` - Bytes 0 to 63.
/// * `hi` - Bytes 64 to 127.
///
/// # Returns
///
/// `lo` shifted down by `K` bytes, with the first `K` bytes of `hi` shifted in at the top.
#[target_feature(enable = "avx512f,avx512bw")]
fn shift_bytes<const K: i32>(lo: __m512i, hi: __m512i) -> __m512i {
    // vpalignr shifts within each 128-bit lane only. So first shift by 4 elements of 4
    // bytes, one whole lane (valignd crosses lanes): lane n of `next` is lane n + 1 of `lo`,
    // and its lane 3 is lane 0 of `hi`.
    let next = _mm512_alignr_epi32::<4>(hi, lo);
    // Per lane, `next` then `lo` shifted down by K bytes: the lane's own bytes from K on,
    // then the first K bytes of the following lane.
    _mm512_alignr_epi8::<K>(next, lo)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{Guarded, Rng, require_avx512};

    fn vector(bytes: [u8; 64]) -> __m512i {
        // SAFETY: both types are 64 bytes, and every bit pattern is valid for each.
        unsafe { std::mem::transmute(bytes) }
    }

    fn bytes_of(vector: __m512i) -> [u8; 64] {
        // SAFETY: both types are 64 bytes, and every bit pattern is valid for each.
        unsafe { std::mem::transmute(vector) }
    }

    fn window_with_zeros(zeros: std::ops::Range<usize>) -> [u8; 320] {
        let mut window = [0x5au8; 320];
        window[zeros].fill(0);
        window
    }

    fn has_run(window: &[u8; 320]) -> bool {
        require_avx512();
        // SAFETY: require_avx512 checked the CPU has AVX-512F and BW.
        unsafe { has_zero_run_256(window) }
    }

    #[test]
    fn t1_zero_mask_tail_clears_bits_past_the_slice() {
        require_avx512();
        let zeros = [0u8; 64];
        for len in 0..64 {
            // SAFETY: require_avx512 checked the CPU has AVX-512F, BW and BMI2.
            assert_eq!(unsafe { zero_mask_tail(&zeros[..len]) }, (1u64 << len) - 1, "{len} zero bytes");
        }
        let mut bytes = [0u8; 64];
        bytes[63] = 1;
        // SAFETY: as above.
        assert_eq!(unsafe { zero_mask_tail(&bytes) }, u64::MAX >> 1);
    }

    #[test]
    fn t2_zero_mask_tail_never_reads_past_the_slice() {
        require_avx512();
        // The slice ends right before an unmapped page: one byte read past it is SIGSEGV.
        let mut guarded = Guarded::new(0);
        for len in 0..=64 {
            let expected = if len == 64 { u64::MAX } else { (1u64 << len) - 1 };
            // SAFETY: require_avx512 checked the CPU has AVX-512F, BW and BMI2.
            assert_eq!(unsafe { zero_mask_tail(guarded.tail(len)) }, expected, "{len} bytes at a page's end");
        }
    }

    #[test]
    fn t3_is_zero_256_sees_one_nonzero_byte_anywhere() {
        require_avx512();
        // SAFETY: require_avx512 checked the CPU has AVX-512F and BW.
        assert!(unsafe { is_zero_256(&[0; 256]) });
        for byte_index in 0..256 {
            for value in [0x01, 0x80, 0xff] {
                let mut block = [0u8; 256];
                block[byte_index] = value;
                // SAFETY: as above.
                assert!(!unsafe { is_zero_256(&block) }, "byte {byte_index} = {value:#04x}");
            }
        }
    }

    #[test]
    fn t4_has_zero_run_256_finds_a_run_starting_anywhere() {
        assert!(!has_run(&[0x5a; 320]));
        for start in 0..256 {
            assert!(has_run(&window_with_zeros(start..start + 3)), "run at {start}");
        }
        assert!(has_run(&window_with_zeros(254..257)), "254, 255 and 256");
        assert!(has_run(&window_with_zeros(255..258)), "255, 256 and 257");
    }

    #[test]
    fn t5_has_zero_run_256_ignores_pairs_and_later_runs() {
        let pairs: [u8; 320] = std::array::from_fn(|index| if index % 3 == 2 { 1 } else { 0 });
        assert!(!has_run(&pairs), "00 00 01 repeated: never 3 zeros in a row");
        assert!(!has_run(&window_with_zeros(256..320)), "a run starting at 256 is the next block's");
        assert!(!has_run(&window_with_zeros(255..257)), "two zeros only");
    }

    #[test]
    fn t6_or3_ors_three_vectors() {
        require_avx512();
        let mut rng = Rng::new(6);
        let mut random = || std::array::from_fn::<u8, 64, _>(|_| if rng.below(4) == 0 { rng.nonzero_byte() } else { 0 });
        for _ in 0..50 {
            let (a, b, c) = (random(), random(), random());
            // SAFETY: require_avx512 checked the CPU has AVX-512F.
            let ored = bytes_of(unsafe { or3(vector(a), vector(b), vector(c)) });
            let expected: [u8; 64] = std::array::from_fn(|index| a[index] | b[index] | c[index]);
            assert_eq!(ored, expected);
        }
    }

    #[test]
    fn t7_shift_bytes_crosses_the_lanes() {
        require_avx512();
        let lo = vector(std::array::from_fn(|index| index as u8));
        let hi = vector(std::array::from_fn(|index| 64 + index as u8));
        let expected = |shift: u8| -> [u8; 64] { std::array::from_fn(|index| shift + index as u8) };
        // SAFETY: require_avx512 checked the CPU has AVX-512F and BW.
        unsafe {
            assert_eq!(bytes_of(shift_bytes::<0>(lo, hi)), expected(0));
            assert_eq!(bytes_of(shift_bytes::<1>(lo, hi)), expected(1), "byte 15 comes from the next lane");
            assert_eq!(bytes_of(shift_bytes::<2>(lo, hi)), expected(2));
            assert_eq!(bytes_of(shift_bytes::<15>(lo, hi)), expected(15));
        }
    }
}
