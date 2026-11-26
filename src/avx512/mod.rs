//! The AVX-512 compressor: it walks the input 64 bytes at a time with a zero mask, and
//! skips 256 bytes at a time inside long zero runs and long literals. It writes exactly
//! the bytes of [`crate::scalar::compress`].

mod decompress;
mod ops;

pub use decompress::decompress;

use std::arch::x86_64::*;

use crate::block::{block_len, write_block};
use crate::{MIN_ZERO_RUN, escape_len, scalar};
use ops::{has_zero_run_256, is_zero_256, zero_mask_tail};

/// Returns whether the CPU has every feature [`compress`] and [`decompress`] enable.
///
/// # Returns
///
/// `true` if it has AVX-512F, AVX-512BW and BMI2.
pub fn is_supported() -> bool {
    // std caches the CPUID results: each check is a load and a test.
    is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("bmi2")
}

/// Compresses bytes, 64 at a time with AVX-512.
///
/// # Arguments
///
/// * `src` - The input.
/// * `dst` - Where the blocks go, at least [`crate::compress_bound`] of the input.
///
/// # Returns
///
/// How many bytes were written; the same bytes as [`crate::scalar::compress`].
///
/// # Panics
///
/// Panics if `dst` is shorter than [`crate::compress_bound`].
///
/// # Safety
///
/// The CPU must have AVX-512F, AVX-512BW and BMI2: see [`is_supported`].
#[target_feature(enable = "avx512f,avx512bw,bmi2")]
pub fn compress(src: &[u8], dst: &mut [u8]) -> usize {
    let src_len = src.len();
    if src_len < 64 {
        // Not one whole word: the vector setup doesn't pay.
        return scalar::compress(src, dst);
    }

    // The end of the last whole 64-byte word.
    let full = src_len / 64 * 64;
    let zero = _mm512_setzero_si512();

    // Where the current literal starts in `src`, and how many bytes it holds so far.
    let mut literal_start = 0;
    let mut literal_len: usize = 0;
    // Zeros seen right after the literal, not yet known to be a run (3 or more).
    let mut zero_len: usize = 0;
    let mut written = 0;

    // Offset in `src` of the next word to load.
    let mut next_word = 0;
    // The current word's mask: bit i set where byte i is zero, shifted so that bit 0 is the
    // next byte to look at.
    let mut zero_mask: u64 = 0;
    // How many bytes of the current word are still to look at.
    let mut remaining_bytes = 0;

    loop {
        // Refill: when the current word is used up, load the next one.
        if remaining_bytes == 0 {
            if next_word == src_len {
                break;
            }
            if next_word < full {
                // SAFETY: next_word + 64 <= full <= src_len, so the 64 bytes are in `src`.
                let word = unsafe { _mm512_loadu_si512(src.as_ptr().add(next_word).cast()) };
                zero_mask = _mm512_cmpeq_epi8_mask(word, zero);
                remaining_bytes = 64;
                next_word += 64;
            } else {
                // The last partial word, 1 to 63 bytes: a masked load reads none past the
                // end, and the mask's bits past the end are clear, so the stretches below
                // stop at the input's end.
                zero_mask = zero_mask_tail(&src[full..]);
                remaining_bytes = src_len - full;
                next_word = src_len;
            }
        }

        // Take one stretch: the zeros at the bottom of the mask, or the literal bytes up to
        // the next place a run may start.
        let is_zero_stretch = zero_mask & 1 == 1;
        let stretch = if is_zero_stretch {
            // The bits above the word are 0 after the shifts, so this stops at its end.
            zero_mask.trailing_ones() as usize
        } else {
            // Starts: bit i set where bytes i, i+1 and i+2 are zero. Stretches of 1 or 2
            // zeros have no start, so they stay inside this literal stretch.
            //
            // The bytes past the word's end aren't loaded yet: count them as zeros. Then
            // zeros touching the end are flagged as a start and taken as a zero stretch on
            // the next pass, so they can join zeros at the start of the next word. If that
            // run turns out shorter than 3, it is put back in the literal when it ends.
            let in_word = _bzhi_u64(u64::MAX, remaining_bytes as u32);
            let guessed = zero_mask | !in_word;
            let starts = guessed & ((guessed >> 1) | (1 << 63)) & ((guessed >> 2) | (3 << 62)) & in_word;
            // Byte 0 is non-zero, so the first start is at 1 or later. trailing_zeros is 64
            // when there is no start: cap it at what remains.
            (starts.trailing_zeros() as usize).min(remaining_bytes)
        };
        // A stretch can be the whole word, and a shift by 64 overflows: hence checked_shr.
        zero_mask = zero_mask.checked_shr(stretch as u32).unwrap_or(0);
        remaining_bytes -= stretch;

        if is_zero_stretch {
            // `stretch` more zeros follow the literal. Whether they are a run is decided
            // when they end, so only count them here.
            zero_len += stretch;

            // The whole word was zeros: a long run is likely, so skip ahead 256 bytes at a
            // time while they are all zero. `remaining_bytes` stays 0, so the refill then
            // loads the first word after them.
            if stretch == 64 {
                while next_word + 256 <= full {
                    let block: &[u8; 256] = src[next_word..next_word + 256].try_into().unwrap();
                    if !is_zero_256(block) {
                        break;
                    }
                    zero_len += 256;
                    next_word += 256;
                }
            }
        } else {
            // `stretch` non-zero bytes: the zeros before them are complete.
            if zero_len >= MIN_ZERO_RUN {
                // A run: it ends the block.
                let literal_end = literal_start + literal_len;
                if written + block_len(literal_len, zero_len) >= escape_len(src_len) {
                    return write_block(dst, src, 0);
                }
                written += write_block(&mut dst[written..], &src[literal_start..literal_end], zero_len);
                literal_start = literal_end + zero_len;
                literal_len = 0;
            } else {
                // 1 or 2 zeros are not a run: they stay in the literal.
                literal_len += zero_len;
            }
            zero_len = 0;
            literal_len += stretch;

            // The whole word was literal, with no run starting in it: likely more literal
            // follows, so skip ahead 256 bytes at a time while no run starts there. The
            // check reads 64 bytes past the 256, for runs starting at their last 2 bytes.
            // A word of 64 literal bytes ends with a non-zero byte (a zero there is flagged
            // as a possible start), so no open zeros are skipped. `remaining_bytes` stays 0,
            // so the refill then loads the first word after them.
            if stretch == 64 {
                while next_word + 320 <= src_len {
                    let window: &[u8; 320] = src[next_word..next_word + 320].try_into().unwrap();
                    if has_zero_run_256(window) {
                        break;
                    }
                    literal_len += 256;
                    next_word += 256;
                }
            }
        }
    }

    // The input ended: no non-zero stretch will close what is still open, so close it here.
    if zero_len < MIN_ZERO_RUN {
        // 1 or 2 zeros at the very end are not a run: they stay in the literal.
        literal_len += zero_len;
        zero_len = 0;
    }
    if literal_len > 0 || zero_len > 0 {
        // The last block: the literal, then the zeros that end the input (0 if none).
        if written + block_len(literal_len, zero_len) >= escape_len(src_len) {
            return write_block(dst, src, 0);
        }
        written += write_block(&mut dst[written..], &src[literal_start..literal_start + literal_len], zero_len);
    }
    written
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compress_bound;
    use crate::testutil::{Guarded, Rng, decompress, reference_compressed, require_avx512, sparse};

    fn compressed(src: &[u8]) -> Vec<u8> {
        require_avx512();
        let mut dst = vec![0u8; compress_bound(src.len())];
        // SAFETY: require_avx512 checked the CPU has AVX-512F, BW and BMI2.
        let written = unsafe { compress(src, &mut dst) };
        dst.truncate(written);
        dst
    }

    fn assert_matches_reference(src: &[u8], what: &str) {
        assert_eq!(compressed(src), reference_compressed(src), "{what}");
    }

    #[test]
    fn t1_matches_the_reference_and_round_trips() {
        // Lengths up to 2000: below 64, partial last words, runs crossing words, and long
        // runs that take the 256-byte skips.
        let mut rng = Rng::new(1);
        for case in 0..500 {
            let src = sparse(&mut rng, 2000);
            let packed = compressed(&src);
            assert_eq!(packed, reference_compressed(&src), "case {case}, {} bytes", src.len());
            assert_eq!(decompress(&packed), src, "case {case}");
        }
    }

    #[test]
    fn t2_matches_the_reference_on_word_edges() {
        // A run of `zeros` zeros starting at `start`, between non-zero bytes, for starts and
        // lengths around the 64-byte word boundaries.
        for len in [63, 64, 65, 127, 128, 129, 200] {
            for start in 0..len {
                for zeros in [1, 2, 3, 4, 64, 65] {
                    let mut src = vec![0x5a; len];
                    src[start..(start + zeros).min(len)].fill(0);
                    assert_matches_reference(&src, &format!("len {len}, {zeros} zeros at {start}"));
                }
            }
        }
    }

    #[test]
    fn t3_matches_the_reference_on_long_zero_runs() {
        // All zeros, and long runs ending before, at and after a 256-byte skip's edge.
        for len in [64, 256, 320, 1000, 1024, 1100] {
            assert_matches_reference(&vec![0u8; len], &format!("{len} zeros"));
        }
        for zeros in [255, 256, 257, 319, 320, 321, 600] {
            let mut src = vec![0x11; 70];
            src.extend(std::iter::repeat_n(0, zeros));
            src.extend([0x22; 70]);
            assert_matches_reference(&src, &format!("{zeros} zeros between literals"));
        }
    }

    #[test]
    fn t4_matches_the_reference_on_long_literals() {
        // A long literal with scattered pairs of zeros (never a run), and one run of 3 at
        // every position across several 256-byte skips, including their last bytes.
        let literal: Vec<u8> = (0..1400).map(|index| if index % 7 < 2 { 0 } else { 0x33 }).collect();
        assert_matches_reference(&literal, "no run at all");
        for start in 0..900 {
            let mut src = literal.clone();
            src[start..start + 3].fill(0);
            assert_matches_reference(&src, &format!("run at {start}"));
        }
    }

    #[test]
    fn t5_matches_the_reference_on_the_escape() {
        let mut src = vec![0x11; 143];
        src.extend([0; 3]);
        src.extend([0x22; 143]);
        let packed = compressed(&src);
        assert_eq!(packed.len(), escape_len(src.len()));
        assert_eq!(packed, reference_compressed(&src));
    }

    #[test]
    fn t6_never_reads_past_the_input() {
        // The input ends right before an unmapped page: one byte read past it is SIGSEGV.
        // Literals and zeros at the end, for every length of the last partial word.
        for fill in [0x00, 0x5a] {
            let mut guarded = Guarded::new(fill);
            for len in 64..=400 {
                let src = guarded.tail(len);
                src[len / 2..len / 2 + 5].fill(fill ^ 0x5a);
                let src = &*src;
                assert_matches_reference(src, &format!("{len} bytes of {fill:#04x} at a page's end"));
                // Put the page back for the next length.
                guarded.tail(len).fill(fill);
            }
        }
    }
}
