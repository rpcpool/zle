//! The compressor without SIMD: it scans 8 bytes at a time in a `u64`.
//!
//! It looks for aligned zero lanes as wide as the minimum run allows (see [`crate::runs`]),
//! several per `u64`, and [`crate::runs::compress`] extends each one to a run and writes
//! the blocks. With a minimum of 15 or more, a lane is the whole `u64`: one compare per 8
//! bytes, however many isolated zeros the literals hold.
//!
//! [`compress_aligned`] looks for whole zero words instead, and ends a run at the first
//! non-zero word: no looking back, no byte-exact end.

use crate::runs::{self, LaneWidth};

/// Compresses bytes, scanning them 8 at a time in a `u64`.
///
/// # Arguments
///
/// * `src` - The input.
/// * `dst` - Where the blocks go, at least [`crate::compress_bound`] of the input.
/// * `min_zero_run` - The shortest run left out, at least [`crate::MIN_ZERO_RUN`].
///
/// # Returns
///
/// How many bytes were written; 0 for an empty input.
///
/// # Panics
///
/// Panics if `dst` is shorter than [`crate::compress_bound`], or `min_zero_run` is below
/// [`crate::MIN_ZERO_RUN`].
pub fn compress(src: &[u8], dst: &mut [u8], min_zero_run: usize) -> usize {
    match LaneWidth::for_min_zero_run(min_zero_run) {
        LaneWidth::Two => runs::compress::<2>(
            src,
            dst,
            min_zero_run,
            find_zero_lane::<2>,
            runs::zeros_before_lane::<2>,
            end_of_zeros,
        ),
        LaneWidth::Four => runs::compress::<4>(
            src,
            dst,
            min_zero_run,
            find_zero_lane::<4>,
            runs::zeros_before_lane::<4>,
            end_of_zeros,
        ),
        LaneWidth::Eight => runs::compress::<8>(
            src,
            dst,
            min_zero_run,
            find_zero_lane::<8>,
            runs::zeros_before_lane::<8>,
            end_of_zeros,
        ),
    }
}

/// Compresses bytes by the aligned rule of [`crate::compress_aligned`], a word at a time.
///
/// # Arguments
///
/// * `src` - The input.
/// * `dst` - Where the blocks go, at least [`crate::compress_bound`] of the input.
/// * `min_zero_words` - The fewest zero words a run has, at least [`crate::MIN_ZERO_WORDS`].
///
/// # Returns
///
/// How many bytes were written; 0 for an empty input.
///
/// # Panics
///
/// Panics if `dst` is shorter than [`crate::compress_bound`], or `min_zero_words` is below
/// [`crate::MIN_ZERO_WORDS`].
pub fn compress_aligned(src: &[u8], dst: &mut [u8], min_zero_words: usize) -> usize {
    runs::compress::<8>(
        src,
        dst,
        runs::min_zero_run_of_words(min_zero_words),
        find_zero_word,
        runs::zeros_before_word,
        end_of_zero_words,
    )
}

/// Returns where the next zero word is.
///
/// # Arguments
///
/// * `src` - The bytes.
/// * `position` - Where to start looking: a multiple of 8, possibly past the end.
///
/// # Returns
///
/// The first multiple of 8 at or after `position` where 8 zero bytes start, all inside
/// `src`; `src.len()` if there is none.
fn find_zero_word(src: &[u8], position: usize) -> usize {
    let Some(rest) = src.get(position..) else {
        return src.len();
    };
    // Whole words as arrays: no bounds check, and no check that `position + 8` wraps, per
    // word, which an indexed loop keeps.
    let (words, _) = rest.as_chunks::<8>();
    match words.iter().position(|word| u64::from_ne_bytes(*word) == 0) {
        Some(word_index) => position + 8 * word_index,
        None => src.len(),
    }
}

/// Returns where a run of zero words ends.
///
/// # Arguments
///
/// * `src` - The bytes.
/// * `position` - Where the zero words start: a multiple of 8, at most `src.len()`.
///
/// # Returns
///
/// The start of the first non-zero word at or after `position`, or the end of the last
/// whole word if they are all zero.
fn end_of_zero_words(src: &[u8], position: usize) -> usize {
    let (words, _) = src[position..].as_chunks::<8>();
    let zero_words = words
        .iter()
        .position(|word| u64::from_ne_bytes(*word) != 0)
        .unwrap_or(words.len());
    position + 8 * zero_words
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

/// Returns which `LANE`-byte lanes of a word are zero, exactly.
///
/// # Arguments
///
/// * `LANE` - The lane width in bytes: 2, 4 or 8.
/// * `word` - 8 bytes, `8 / LANE` lanes.
///
/// # Returns
///
/// The high bit of each lane set where that lane of `word` is 0, every other bit clear.
#[inline(always)]
fn zero_lanes<const LANE: usize>(word: u64) -> u64 {
    if LANE == 8 {
        return if word == 0 { 1 << 63 } else { 0 };
    }
    // Every bit of every lane but its high bit: 0x7fff repeated for lanes of 2.
    let lane_bits = 8 * LANE as u32;
    let low_bits = (u64::MAX / ((1 << lane_bits) - 1)) * ((1 << (lane_bits - 1)) - 1);
    // (lane & low_bits) + low_bits sets a lane's high bit unless its low bits are 0; it is
    // at most 2 * low_bits, so no carry reaches the next lane. ORing in the lane itself
    // covers its own high bit. A lane is zero exactly where neither sets the high bit.
    !(((word & low_bits) + low_bits) | word | low_bits)
}

/// Returns where the next aligned zero lane is.
///
/// # Arguments
///
/// * `LANE` - The lane width in bytes: 2, 4 or 8.
/// * `src` - The bytes.
/// * `position` - Where to start looking: a multiple of `LANE`, possibly past the end.
///
/// # Returns
///
/// The first multiple of `LANE` at or after `position` where `LANE` zero bytes start, all
/// inside `src`; `src.len()` if there is none.
fn find_zero_lane<const LANE: usize>(src: &[u8], mut position: usize) -> usize {
    // `position` stays a multiple of LANE, so every lane of each word is aligned too.
    while position + 8 <= src.len() {
        let lanes = zero_lanes::<LANE>(load_word(src, position));
        if lanes != 0 {
            // The first zero lane's high bit is the lowest set bit.
            return position + lanes.trailing_zeros() as usize / (8 * LANE) * LANE;
        }
        position += 8;
    }
    // Fewer than 8 bytes left: the whole lanes among them, one at a time.
    while position + LANE <= src.len() {
        if src[position..position + LANE].iter().all(|&byte| byte == 0) {
            return position;
        }
        position += LANE;
    }
    src.len()
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
    use crate::testutil::{
        MIN_ZERO_RUNS, Rng, aligned_edge_cases, decompress, reference_compressed,
        reference_compressed_aligned, sparse,
    };
    use crate::{MIN_ZERO_RUN, compress_bound, escape_len};

    fn compressed(src: &[u8], min_zero_run: usize) -> Vec<u8> {
        let mut dst = vec![0u8; compress_bound(src.len())];
        let written = compress(src, &mut dst, min_zero_run);
        dst.truncate(written);
        dst
    }

    #[test]
    fn t1_empty_input_writes_nothing() {
        for &min_zero_run in &MIN_ZERO_RUNS {
            assert_eq!(compressed(&[], min_zero_run), [], "min {min_zero_run}");
        }
    }

    #[test]
    fn t2_no_zeros_is_one_literal_block() {
        assert_eq!(compressed(&[1, 2, 3], MIN_ZERO_RUN), [0x30, 1, 2, 3]);
    }

    #[test]
    fn t3_two_zeros_stay_in_the_literal() {
        assert_eq!(compressed(&[1, 0, 0, 2], MIN_ZERO_RUN), [0x40, 1, 0, 0, 2]);
    }

    #[test]
    fn t4_three_zeros_end_a_block() {
        assert_eq!(
            compressed(&[1, 0, 0, 0, 2], MIN_ZERO_RUN),
            [0x13, 1, 0x10, 2]
        );
    }

    #[test]
    fn t5_leading_zeros_are_a_block_without_literal() {
        assert_eq!(compressed(&[0, 0, 0, 0, 7], MIN_ZERO_RUN), [0x04, 0x10, 7]);
    }

    #[test]
    fn t6_trailing_zeros_need_no_last_block() {
        let mut src = vec![9];
        src.extend([0; 10]);
        assert_eq!(compressed(&src, MIN_ZERO_RUN), [0x1a, 9]);
    }

    #[test]
    fn t7_all_zeros() {
        // Z = 100 = 15 + 85.
        for &min_zero_run in &MIN_ZERO_RUNS[..10] {
            assert_eq!(
                compressed(&[0; 100], min_zero_run),
                [0x0f, 85],
                "min {min_zero_run}"
            );
        }
    }

    #[test]
    fn t8_escape_when_blocks_save_nothing() {
        // 143 literal bytes cost 1 + 2 + 143 (an extension of 2 bytes), exactly what the
        // 3 zeros after them save. The output reaches escape_len, so it escapes.
        let mut src = vec![0x11; 143];
        src.extend([0; 3]);
        src.extend([0x22; 143]);
        let out = compressed(&src, MIN_ZERO_RUN);
        assert_eq!(out.len(), escape_len(src.len()));
        assert_eq!(
            out[..3],
            [0xf0, 0x92, 0x02],
            "one literal block of 289 bytes: 289 - 15 = 274"
        );
        assert_eq!(out[3..], src);
    }

    #[test]
    fn t9_one_byte_less_does_not_escape() {
        // 142 literal bytes have a 1-byte extension: the run now saves one byte.
        let mut src = vec![0x11; 142];
        src.extend([0; 3]);
        src.extend([0x22; 143]);
        let out = compressed(&src, MIN_ZERO_RUN);
        assert_eq!(out.len(), escape_len(src.len()) - 1);
        assert_eq!(out[..2], [0xf3, 127]);
    }

    #[test]
    fn t10_matches_the_reference_and_round_trips() {
        let mut rng = Rng::new(10);
        for &min_zero_run in &MIN_ZERO_RUNS {
            for case in 0..200 {
                let src = sparse(&mut rng, 2000, min_zero_run);
                let packed = compressed(&src, min_zero_run);
                assert_eq!(
                    packed,
                    reference_compressed(&src, min_zero_run),
                    "min {min_zero_run}, case {case}"
                );
                assert_eq!(decompress(&packed), src, "min {min_zero_run}, case {case}");
            }
        }
    }

    #[test]
    fn t11_matches_the_reference_on_word_edges() {
        // Runs around each minimum, starting at every offset of inputs that end in partial
        // words, so the lane scan's tail loop is crossed too.
        for &min_zero_run in &MIN_ZERO_RUNS[..9] {
            // Tails of every length after the last whole word, among them whole lanes only
            // (4, 6, 12, 14, 20): a run whose one zero lane is the input's last.
            for len in [1, 2, 4, 6, 7, 8, 9, 12, 14, 15, 16, 17, 20, 24, 33] {
                for start in 0..len {
                    for zeros in [
                        1,
                        2,
                        min_zero_run - 1,
                        min_zero_run,
                        min_zero_run + 1,
                        2 * min_zero_run,
                    ] {
                        let mut src = vec![0x5a; len];
                        src[start..(start + zeros).min(len)].fill(0);
                        assert_eq!(
                            compressed(&src, min_zero_run),
                            reference_compressed(&src, min_zero_run),
                            "min {min_zero_run}, len {len}, {zeros} zeros at {start}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "the smallest is 3")]
    fn t12_a_minimum_below_3_panics() {
        compressed(&[1, 0, 0, 2], 2);
    }

    /// Checks [`zero_lanes`] for one lane width against a lane-by-lane comparison.
    fn check_zero_lanes<const LANE: usize>() {
        // Every lane takes values that probe each carry and borrow, with the other lanes
        // zero, then non-zero: a carry between lanes would flag a wrong neighbour.
        let probes: Vec<u64> = vec![
            0,
            1,
            0x7f,
            0x80,
            0xff,
            0x100,
            0x7fff,
            0x8000,
            0xffff,
            0x8000_0000,
            0xffff_ffff,
            u64::MAX,
        ];
        let lane_mask = if LANE == 8 {
            u64::MAX
        } else {
            (1u64 << (8 * LANE)) - 1
        };
        for filler in [0u64, 1, 0x8000_0000_0000_0000 >> (64 - 8 * LANE)] {
            for lane_index in 0..8 / LANE {
                for &probe in &probes {
                    let shift = 8 * LANE * lane_index;
                    let filled =
                        (0..8 / LANE).fold(0u64, |word, index| word | filler << (8 * LANE * index));
                    let word = filled & !(lane_mask << shift) | (probe & lane_mask) << shift;
                    let expected = (0..8 / LANE)
                        .filter(|&index| (word >> (8 * LANE * index)) & lane_mask == 0)
                        .fold(0u64, |mask, index| {
                            mask | 1 << (8 * LANE * index + 8 * LANE - 1)
                        });
                    assert_eq!(
                        zero_lanes::<LANE>(word),
                        expected,
                        "lanes of {LANE}, word {word:#018x}"
                    );
                }
            }
        }
    }

    #[test]
    fn t13_zero_lanes_flags_exactly_the_zero_lanes() {
        check_zero_lanes::<2>();
        check_zero_lanes::<4>();
        check_zero_lanes::<8>();
    }

    /// Checks [`find_zero_lane`] for one lane width against a lane-by-lane scan.
    fn check_find_zero_lane<const LANE: usize>(rng: &mut Rng) {
        for case in 0..300 {
            let src = sparse(rng, 40, 2 * LANE);
            for position in (0..=src.len() + LANE).step_by(LANE) {
                let expected = (position..src.len())
                    .step_by(LANE)
                    .find(|&lane| {
                        lane + LANE <= src.len()
                            && src[lane..lane + LANE].iter().all(|&byte| byte == 0)
                    })
                    .unwrap_or(src.len());
                assert_eq!(
                    find_zero_lane::<LANE>(&src, position),
                    expected,
                    "lanes of {LANE}, case {case}, from {position}"
                );
            }
        }
    }

    #[test]
    fn t14_find_zero_lane_matches_a_lane_scan() {
        let mut rng = Rng::new(14);
        check_find_zero_lane::<2>(&mut rng);
        check_find_zero_lane::<4>(&mut rng);
        check_find_zero_lane::<8>(&mut rng);
    }

    #[test]
    fn t15_end_of_zeros_matches_a_byte_scan() {
        let mut rng = Rng::new(15);
        for case in 0..300 {
            let src = sparse(&mut rng, 40, MIN_ZERO_RUN);
            for position in 0..=src.len() {
                let end = (position..src.len())
                    .find(|&index| src[index] != 0)
                    .unwrap_or(src.len());
                assert_eq!(
                    end_of_zeros(&src, position),
                    end,
                    "case {case}, end of zeros from {position}"
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
    fn t16_aligned_matches_the_reference_on_edge_cases() {
        for min_zero_words in [2, 3, 5] {
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
    fn t17_aligned_matches_the_reference_and_round_trips() {
        let mut rng = Rng::new(17);
        for min_zero_words in [2, 3, 5, 40] {
            for case in 0..200 {
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
        }
    }

    #[test]
    fn t18_find_zero_word_and_end_of_zero_words_match_a_word_scan() {
        let mut rng = Rng::new(18);
        for case in 0..300 {
            let src = sparse(&mut rng, 60, 16);
            let is_zero_word = |word: usize| src[word..word + 8].iter().all(|&byte| byte == 0);
            let last_word_end = src.len() / 8 * 8;
            for position in (0..=src.len() + 8).step_by(8) {
                let next = (position..last_word_end)
                    .step_by(8)
                    .find(|&word| is_zero_word(word))
                    .unwrap_or(src.len());
                assert_eq!(
                    find_zero_word(&src, position),
                    next,
                    "case {case}, from {position}"
                );
                if position <= src.len() {
                    let end = (position..last_word_end)
                        .step_by(8)
                        .find(|&word| !is_zero_word(word))
                        .unwrap_or(last_word_end.max(position));
                    assert_eq!(
                        end_of_zero_words(&src, position),
                        end,
                        "case {case}, end from {position}"
                    );
                }
            }
        }
    }
}
