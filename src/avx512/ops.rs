//! The vector operations the AVX-512 compressor is built from.
//!
//! [`or3`] and [`is_zero_256`] are derived from Richard Patel's zle
//! (<https://github.com/riptl/zle>, `ZLE_OR3V` and `zle_nz_256` in `lib/zle.c`), Copyright
//! (c) 2026 Richard Patel, licensed under the Apache License, Version 2.0 (`LICENSE-APACHE`
//! at the crate's root). They were translated to Rust and modified for this crate.

use std::arch::x86_64::*;

/// Loads at most 64 bytes, reading none past the slice.
///
/// # Arguments
///
/// * `bytes` - At most 64 bytes.
///
/// # Returns
///
/// A vector holding `bytes`, then zeros from byte `bytes.len()` up.
///
/// # Panics
///
/// Panics if `bytes` is longer than 64.
#[target_feature(enable = "avx512f,avx512bw,bmi2")]
pub fn load_tail(bytes: &[u8]) -> __m512i {
    assert!(bytes.len() <= 64);
    // The first `bytes.len()` lanes. Unlike `(1 << n) - 1`, bzhi works for n = 64.
    let in_slice = _bzhi_u64(u64::MAX, bytes.len() as u32);
    // SAFETY: a masked load reads only the selected lanes, all inside the slice; the others
    // are never touched, so they can't fault.
    unsafe { _mm512_maskz_loadu_epi8(in_slice, bytes.as_ptr().cast()) }
}

/// Returns which `LANE`-byte lanes of a vector are zero.
///
/// # Arguments
///
/// * `LANE` - The lane width in bytes: 2, 4 or 8.
/// * `vector` - 64 bytes, `64 / LANE` lanes.
///
/// # Returns
///
/// A mask with bit `i` set where lane `i` (bytes `i * LANE` to `i * LANE + LANE - 1`) is
/// zero; bits from `64 / LANE` up are clear.
#[target_feature(enable = "avx512f,avx512bw")]
pub fn zero_lane_mask<const LANE: usize>(vector: __m512i) -> u64 {
    let zero = _mm512_setzero_si512();
    // LANE is a constant: only one arm is compiled into each copy.
    match LANE {
        2 => _mm512_cmpeq_epi16_mask(vector, zero) as u64,
        4 => _mm512_cmpeq_epi32_mask(vector, zero) as u64,
        8 => _mm512_cmpeq_epi64_mask(vector, zero) as u64,
        _ => unreachable!("lanes of {LANE} bytes"),
    }
}

/// Returns whether 256 bytes hold an aligned zero lane.
///
/// # Arguments
///
/// * `LANE` - The lane width in bytes: 2, 4 or 8.
/// * `block` - The 256 bytes; their lanes are aligned to the block's start.
///
/// # Returns
///
/// `true` if some `LANE` bytes at a multiple of `LANE` are all zero.
#[target_feature(enable = "avx512f,avx512bw")]
pub fn has_zero_lane_256<const LANE: usize>(block: &[u8; 256]) -> bool {
    let vectors: [__m512i; 4] = std::array::from_fn(|vector_index| {
        // SAFETY: the block holds 256 bytes, so the 64 at vector_index * 64 are readable.
        unsafe { _mm512_loadu_si512(block.as_ptr().add(vector_index * 64).cast()) }
    });
    // The unsigned minimum per lane is zero where any of the four is zero.
    let min = |a: __m512i, b: __m512i| match LANE {
        2 => _mm512_min_epu16(a, b),
        4 => _mm512_min_epu32(a, b),
        8 => _mm512_min_epu64(a, b),
        _ => unreachable!("lanes of {LANE} bytes"),
    };
    let merged = min(min(vectors[0], vectors[1]), min(vectors[2], vectors[3]));
    zero_lane_mask::<LANE>(merged) != 0
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
    let ored = or3(
        vectors[0],
        vectors[1],
        _mm512_or_si512(vectors[2], vectors[3]),
    );
    _mm512_test_epi8_mask(ored, ored) == 0
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

    /// The lanes of `bytes` that are all zero, one at a time.
    fn zero_lanes_bytewise<const LANE: usize>(bytes: &[u8]) -> u64 {
        bytes
            .chunks_exact(LANE)
            .enumerate()
            .filter(|(_, lane)| lane.iter().all(|&byte| byte == 0))
            .fold(0, |mask, (lane_index, _)| mask | 1 << lane_index)
    }

    #[test]
    fn t1_load_tail_zeroes_past_the_slice() {
        require_avx512();
        let bytes: [u8; 64] = std::array::from_fn(|index| 1 + index as u8);
        for len in 0..=64 {
            // SAFETY: require_avx512 checked the CPU has AVX-512F, BW and BMI2.
            let loaded = bytes_of(unsafe { load_tail(&bytes[..len]) });
            let expected: [u8; 64] =
                std::array::from_fn(|index| if index < len { bytes[index] } else { 0 });
            assert_eq!(loaded, expected, "{len} bytes");
        }
    }

    #[test]
    fn t2_load_tail_never_reads_past_the_slice() {
        require_avx512();
        // The slice ends right before an unmapped page: one byte read past it is SIGSEGV.
        let mut guarded = Guarded::new(0x5a);
        for len in 0..=64 {
            // SAFETY: require_avx512 checked the CPU has AVX-512F, BW and BMI2.
            let loaded = bytes_of(unsafe { load_tail(guarded.tail(len)) });
            assert!(
                loaded[..len].iter().all(|&byte| byte == 0x5a),
                "{len} bytes at a page's end"
            );
        }
    }

    fn check_zero_lane_mask<const LANE: usize>(rng: &mut Rng) {
        // Each lane zero or not, and a non-zero lane with one non-zero byte at any place.
        for case in 0..200 {
            let bytes: [u8; 64] = std::array::from_fn(|_| {
                if rng.below(LANE + 1) == 0 {
                    rng.nonzero_byte()
                } else {
                    0
                }
            });
            // SAFETY: the callers ran require_avx512, which checked AVX-512F and BW.
            let mask = unsafe { zero_lane_mask::<LANE>(vector(bytes)) };
            assert_eq!(
                mask,
                zero_lanes_bytewise::<LANE>(&bytes),
                "lanes of {LANE}, case {case}: {bytes:02x?}"
            );
        }
    }

    #[test]
    fn t3_zero_lane_mask_flags_exactly_the_zero_lanes() {
        require_avx512();
        let mut rng = Rng::new(3);
        check_zero_lane_mask::<2>(&mut rng);
        check_zero_lane_mask::<4>(&mut rng);
        check_zero_lane_mask::<8>(&mut rng);
    }

    fn check_has_zero_lane_256<const LANE: usize>() {
        // SAFETY (every call below): the callers ran require_avx512, which checked
        // AVX-512F and BW.
        assert!(
            !unsafe { has_zero_lane_256::<LANE>(&[0x5a; 256]) },
            "lanes of {LANE}: no zero"
        );
        // With 0xff around it, a zero lane is the unsigned minimum but not the signed one.
        for filler in [0x5a, 0xff] {
            for lane_start in (0..256).step_by(LANE) {
                let mut block = [filler; 256];
                block[lane_start..lane_start + LANE].fill(0);
                assert!(
                    unsafe { has_zero_lane_256::<LANE>(&block) },
                    "lanes of {LANE}: zero lane at {lane_start} among {filler:#04x}"
                );
            }
        }
        // LANE zeros straddling two lanes, at every misaligned start: no lane is zero.
        for zeros_start in (0..256 - LANE).filter(|start| start % LANE != 0) {
            let mut block = [0x5au8; 256];
            block[zeros_start..zeros_start + LANE].fill(0);
            assert!(
                !unsafe { has_zero_lane_256::<LANE>(&block) },
                "lanes of {LANE}: zeros at {zeros_start}"
            );
        }
        // 0x80 and 0x01 in each byte of a lane: the minimum must be unsigned and per lane.
        for byte_in_lane in 0..LANE {
            for value in [0x01, 0x80, 0xff] {
                let mut block = [0x5au8; 256];
                block[64..64 + LANE].fill(0);
                block[64 + byte_in_lane] = value;
                assert!(
                    !unsafe { has_zero_lane_256::<LANE>(&block) },
                    "lanes of {LANE}: {value:#04x} at {byte_in_lane}"
                );
            }
        }
    }

    #[test]
    fn t4_has_zero_lane_256_finds_only_aligned_zero_lanes() {
        require_avx512();
        check_has_zero_lane_256::<2>();
        check_has_zero_lane_256::<4>();
        check_has_zero_lane_256::<8>();
    }

    #[test]
    fn t5_is_zero_256_sees_one_nonzero_byte_anywhere() {
        require_avx512();
        // SAFETY: require_avx512 checked the CPU has AVX-512F and BW.
        assert!(unsafe { is_zero_256(&[0; 256]) });
        for byte_index in 0..256 {
            for value in [0x01, 0x80, 0xff] {
                let mut block = [0u8; 256];
                block[byte_index] = value;
                // SAFETY: as above.
                assert!(
                    !unsafe { is_zero_256(&block) },
                    "byte {byte_index} = {value:#04x}"
                );
            }
        }
    }

    #[test]
    fn t6_or3_ors_three_vectors() {
        require_avx512();
        let mut rng = Rng::new(6);
        let mut random = || {
            std::array::from_fn::<u8, 64, _>(|_| {
                if rng.below(4) == 0 {
                    rng.nonzero_byte()
                } else {
                    0
                }
            })
        };
        for _ in 0..50 {
            let (a, b, c) = (random(), random(), random());
            // SAFETY: require_avx512 checked the CPU has AVX-512F.
            let ored = bytes_of(unsafe { or3(vector(a), vector(b), vector(c)) });
            let expected: [u8; 64] = std::array::from_fn(|index| a[index] | b[index] | c[index]);
            assert_eq!(ored, expected);
        }
    }
}
