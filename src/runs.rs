//! Turns the zero runs a scanner finds into blocks: the part of compression the scalar and
//! AVX-512 compressors share.
//!
//! A run of at least `2 * W - 1` zeros always covers a whole aligned `W`-byte lane (a
//! `W`-byte word starting at a multiple of `W` in the input): the first multiple of `W` at
//! or after the run's start is at most `W - 1` bytes in, and `W` more zeros follow it. So
//! instead of stopping at every zero byte, a scanner looks only for zero lanes, as wide as
//! the minimum run allows, and [`compress`] extends each one to the run's real ends.
//!
//! The aligned rule of [`crate::compress_aligned`] is the same loop with lanes of 8 and no
//! extending: a run is whole zero words, so it starts at its first zero word and ends at the
//! first non-zero one.
//!
//! ```text
//! min_zero_run = 7, lanes of 4:   | 5a 00 00 00 | 00 00 00 5a | 00 00 5a 5a |
//!                                    └─ run of 6 ─┘ no zero lane: stays literal
//! min_zero_run = 3, lanes of 2:   | 5a 00 | 00 00 | 00 00 | 00 5a | 00 00 | 5a 5a |
//!                                    └── zero lane at 2: run 1..7 ──┘ └ lane at 8: run of 2, literal
//! ```

use crate::block::{block_len, write_block};
use crate::{MIN_ZERO_RUN, MIN_ZERO_WORDS, escape_len};

/// How wide the aligned lanes are that a scanner looks for, for one minimum run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaneWidth {
    /// 2-byte lanes, for minimum runs of 3 to 6.
    Two,
    /// 4-byte lanes, for minimum runs of 7 to 14.
    Four,
    /// 8-byte lanes, for minimum runs of 15 and more.
    Eight,
}

impl LaneWidth {
    /// Returns the widest lane every run of at least `min_zero_run` zeros covers.
    ///
    /// # Arguments
    ///
    /// * `min_zero_run` - The shortest run left out of the output.
    ///
    /// # Returns
    ///
    /// The widest `W` of 2, 4 and 8 with `2 * W - 1 <= min_zero_run`.
    ///
    /// # Panics
    ///
    /// Panics if `min_zero_run` is below [`MIN_ZERO_RUN`].
    pub fn for_min_zero_run(min_zero_run: usize) -> Self {
        assert!(
            min_zero_run >= MIN_ZERO_RUN,
            "min_zero_run is {min_zero_run}, the smallest is {MIN_ZERO_RUN}: shorter runs save nothing"
        );
        match min_zero_run {
            ..=6 => Self::Two,
            7..=14 => Self::Four,
            _ => Self::Eight,
        }
    }
}

/// Returns the shortest run in bytes for a minimum in words, for the aligned rule.
///
/// # Arguments
///
/// * `min_zero_words` - The fewest whole zero words left out.
///
/// # Returns
///
/// `8 * min_zero_words`, or `usize::MAX` if that overflows: no run is that long.
///
/// # Panics
///
/// Panics if `min_zero_words` is below [`MIN_ZERO_WORDS`].
pub fn min_zero_run_of_words(min_zero_words: usize) -> usize {
    assert!(
        min_zero_words >= MIN_ZERO_WORDS,
        "min_zero_words is {min_zero_words}, the smallest is {MIN_ZERO_WORDS}: \
         for runs of one word, compress_with_min_run at 8 leaves out more"
    );
    min_zero_words.saturating_mul(8)
}

/// Compresses bytes, with the scanner given as three functions.
///
/// The scanner decides what a run is: where its lanes are, where a run starts relative to
/// its first zero lane, and where it ends. This loop decides which runs are long enough,
/// writes the blocks and falls back to the escape.
///
/// Inlined into each caller, so that a scanner built from `#[target_feature]` functions is
/// inlined too, with the caller's features.
///
/// # Arguments
///
/// * `LANE` - The lane width in bytes: 2, 4 or 8, with `2 * LANE - 1 <= min_zero_run`.
/// * `src` - The input.
/// * `dst` - Where the blocks go, at least [`crate::compress_bound`] of the input.
/// * `min_zero_run` - The shortest run left out; shorter ones stay in the literal.
/// * `find_zero_lane` - Given `src` and a multiple of `LANE`, returns where the first
///   `LANE` zero bytes start at a multiple of `LANE` at or after it, wholly inside `src`;
///   `src.len()` if there are none. It may be passed a start past the end.
/// * `zeros_before_lane` - Given `src`, a zero lane and where the current literal starts,
///   returns how many bytes before the lane belong to its run: [`zeros_before_lane`] for
///   runs of exact length, [`zeros_before_word`] for runs of whole words.
/// * `end_of_zeros` - Given `src` and the end of a zero lane, returns where the run ends:
///   a non-zero byte, or `src.len()`, for runs of exact length; the first non-zero word, or
///   the end of the last whole word, for runs of whole words.
///
/// # Returns
///
/// How many bytes were written: the blocks of every run of at least `min_zero_run` zeros,
/// or the escape if they would not be shorter.
///
/// # Panics
///
/// Panics if `dst` is shorter than [`crate::compress_bound`].
#[inline(always)]
pub fn compress<const LANE: usize>(
    src: &[u8],
    dst: &mut [u8],
    min_zero_run: usize,
    find_zero_lane: impl Fn(&[u8], usize) -> usize,
    zeros_before_lane: impl Fn(&[u8], usize, usize) -> usize,
    end_of_zeros: impl Fn(&[u8], usize) -> usize,
) -> usize {
    debug_assert!(
        2 * LANE - 1 <= min_zero_run,
        "a run of {min_zero_run} may cover no lane of {LANE}"
    );
    let src_len = src.len();
    // The current literal is everything from `literal_start` up to the next run.
    let mut literal_start = 0;
    // Where to look for the next zero lane: always a multiple of LANE.
    let mut lane_start = 0;
    let mut written = 0;

    loop {
        let lane = find_zero_lane(src, lane_start);
        if lane == src_len {
            break;
        }
        let zeros_start = lane - zeros_before_lane(src, lane, literal_start);
        let zeros_end = end_of_zeros(src, lane + LANE);
        let zero_len = zeros_end - zeros_start;
        if zero_len >= min_zero_run {
            // A run: it ends the block whose literal is everything since the previous run.
            let literal = &src[literal_start..zeros_start];
            if written + block_len(literal.len(), zero_len) >= escape_len(src_len) {
                return write_block(dst, src, 0);
            }
            written += write_block(&mut dst[written..], literal, zero_len);
            literal_start = zeros_end;
        }
        // A shorter run stays in the literal: only move past it. The lane holding
        // `zeros_end` holds a non-zero byte (or is partial, or lies past the end), so look
        // from the next.
        lane_start = (zeros_end / LANE + 1) * LANE;
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

/// Returns how many zeros come right before a zero lane: where its run really starts.
///
/// There are at most `LANE - 1`. The lane before this one held a non-zero byte, or was
/// skipped because it holds the end of the previous run, which is non-zero; within the
/// first lane of the input, the zeros stop at its start.
///
/// # Arguments
///
/// * `LANE` - The lane width in bytes.
/// * `src` - The input.
/// * `lane` - Where the zero lane starts.
/// * `literal_start` - Where the current literal starts: the zeros never reach before it.
///
/// # Returns
///
/// The count of zero bytes just before `lane`, at most `LANE - 1`.
#[inline(always)]
pub fn zeros_before_lane<const LANE: usize>(
    src: &[u8],
    lane: usize,
    literal_start: usize,
) -> usize {
    if lane < 8 {
        // At the input's start: too few bytes for a word, so one byte at a time.
        return (literal_start..lane)
            .rev()
            .take_while(|&index| src[index] == 0)
            .count();
    }
    // Without a branch per byte: with one per run, mispredicted half the time on tiny
    // blocks, it would cost more than the scan. Byte `lane - 1` is the word's highest, so
    // its leading zero bits, 8 per byte, count the zeros just before the lane.
    let before = u64::from_le_bytes(src[lane - 8..lane].try_into().unwrap());
    let zero_count = (before.leading_zeros() / 8) as usize;
    debug_assert!(
        zero_count < LANE,
        "{zero_count} zeros before a lane of {LANE}"
    );
    debug_assert!(
        zero_count <= lane - literal_start,
        "zeros before the literal's start"
    );
    zero_count
}

/// Returns how many zeros before a zero word belong to its run, for runs of whole words:
/// none, since the run starts at its first zero word.
///
/// # Arguments
///
/// * `_src` - The input.
/// * `_word` - Where the zero word starts.
/// * `_literal_start` - Where the current literal starts.
///
/// # Returns
///
/// 0.
#[inline(always)]
pub fn zeros_before_word(_src: &[u8], _word: usize, _literal_start: usize) -> usize {
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compress_bound;
    use crate::testutil::{MIN_ZERO_RUNS, Rng, reference_compressed, sparse};

    /// The slowest correct scanner: one byte at a time.
    fn find_zero_lane_bytewise<const LANE: usize>(src: &[u8], lane_start: usize) -> usize {
        (lane_start..src.len())
            .step_by(LANE)
            .find(|&lane| {
                lane + LANE <= src.len() && src[lane..lane + LANE].iter().all(|&byte| byte == 0)
            })
            .unwrap_or(src.len())
    }

    fn end_of_zeros_bytewise(src: &[u8], position: usize) -> usize {
        (position..src.len())
            .find(|&index| src[index] != 0)
            .unwrap_or(src.len())
    }

    fn compressed_bytewise(src: &[u8], min_zero_run: usize) -> Vec<u8> {
        let mut dst = vec![0u8; compress_bound(src.len())];
        let written = match LaneWidth::for_min_zero_run(min_zero_run) {
            LaneWidth::Two => compress::<2>(
                src,
                &mut dst,
                min_zero_run,
                find_zero_lane_bytewise::<2>,
                zeros_before_lane::<2>,
                end_of_zeros_bytewise,
            ),
            LaneWidth::Four => compress::<4>(
                src,
                &mut dst,
                min_zero_run,
                find_zero_lane_bytewise::<4>,
                zeros_before_lane::<4>,
                end_of_zeros_bytewise,
            ),
            LaneWidth::Eight => compress::<8>(
                src,
                &mut dst,
                min_zero_run,
                find_zero_lane_bytewise::<8>,
                zeros_before_lane::<8>,
                end_of_zeros_bytewise,
            ),
        };
        dst.truncate(written);
        dst
    }

    /// Checks [`zeros_before_lane`] for one lane width: every count it can see, at lanes
    /// near the input's start (the byte loop) and further in (the word).
    fn check_zeros_before_lane<const LANE: usize>() {
        for lane in (LANE..40).step_by(LANE) {
            for zero_count in 0..LANE.min(lane) {
                for literal_start in [0, lane - zero_count] {
                    let mut src = vec![0x5au8; lane + LANE + 1];
                    src[lane - zero_count..lane + LANE].fill(0);
                    assert_eq!(
                        zeros_before_lane::<LANE>(&src, lane, literal_start),
                        zero_count,
                        "lanes of {LANE}, lane at {lane}, literal from {literal_start}"
                    );
                }
            }
        }
        // A lane at the input's start has nothing before it.
        assert_eq!(zeros_before_lane::<LANE>(&[0; 16], 0, 0), 0);
    }

    #[test]
    fn t7_zeros_before_lane_counts_the_zeros_just_before() {
        check_zeros_before_lane::<2>();
        check_zeros_before_lane::<4>();
        check_zeros_before_lane::<8>();
    }

    #[test]
    fn t1_lane_width_is_the_widest_every_run_covers() {
        for min_zero_run in MIN_ZERO_RUN..=64 {
            let lane = match LaneWidth::for_min_zero_run(min_zero_run) {
                LaneWidth::Two => 2,
                LaneWidth::Four => 4,
                LaneWidth::Eight => 8,
            };
            assert!(
                2 * lane - 1 <= min_zero_run,
                "{min_zero_run}: a run may miss a lane of {lane}"
            );
            assert!(
                lane == 8 || 2 * (2 * lane) - 1 > min_zero_run,
                "{min_zero_run}: a lane of {} fits",
                2 * lane
            );
        }
        assert_eq!(LaneWidth::for_min_zero_run(usize::MAX), LaneWidth::Eight);
    }

    #[test]
    #[should_panic(expected = "the smallest is 3")]
    fn t2_a_minimum_below_3_panics() {
        LaneWidth::for_min_zero_run(2);
    }

    #[test]
    fn t3_matches_the_reference_for_every_minimum() {
        let mut rng = Rng::new(3);
        for &min_zero_run in &MIN_ZERO_RUNS {
            for case in 0..200 {
                let src = sparse(&mut rng, 1000, min_zero_run);
                assert_eq!(
                    compressed_bytewise(&src, min_zero_run),
                    reference_compressed(&src, min_zero_run),
                    "min {min_zero_run}, case {case}"
                );
            }
        }
    }

    #[test]
    fn t4_a_run_of_exactly_the_minimum_at_every_offset() {
        // The run starts at every offset into a lane, and is exactly long enough or one
        // short: the shortest run that counts has the least room around its lane.
        for &min_zero_run in &MIN_ZERO_RUNS {
            for zero_len in [min_zero_run - 1, min_zero_run] {
                for start in 0..20 {
                    let mut src = vec![0x5a; start + zero_len + 20];
                    src[start..start + zero_len].fill(0);
                    assert_eq!(
                        compressed_bytewise(&src, min_zero_run),
                        reference_compressed(&src, min_zero_run),
                        "min {min_zero_run}, {zero_len} zeros at {start}"
                    );
                }
            }
        }
    }

    #[test]
    fn t5_runs_at_the_input_edges() {
        // Leading and trailing runs, alone or with a literal, around the minimum.
        for &min_zero_run in &MIN_ZERO_RUNS {
            for zero_len in [
                1,
                min_zero_run - 1,
                min_zero_run,
                min_zero_run + 1,
                2 * min_zero_run + 7,
            ] {
                let zeros = vec![0u8; zero_len];
                let inputs = [
                    zeros.clone(),
                    [&zeros[..], &[9]].concat(),
                    [&[9], &zeros[..]].concat(),
                ];
                for src in inputs {
                    assert_eq!(
                        compressed_bytewise(&src, min_zero_run),
                        reference_compressed(&src, min_zero_run),
                        "min {min_zero_run}, {src:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn t6_runs_one_short_of_the_minimum_give_the_escape() {
        // Runs of 15 at a minimum of 16: each covers a zero lane of 8 or not depending on
        // where it starts, and none counts, so the output is the input as one block.
        let mut src = Vec::new();
        for start_in_lane in 0..8 {
            src.extend(std::iter::repeat_n(0x11, 9 + start_in_lane));
            src.extend([0; 15]);
        }
        let packed = compressed_bytewise(&src, 16);
        assert_eq!(packed, reference_compressed(&src, 16));
        assert_eq!(packed.len(), escape_len(src.len()));
    }

    #[test]
    fn t8_min_zero_run_of_words_is_8_bytes_a_word() {
        assert_eq!(min_zero_run_of_words(2), 16);
        assert_eq!(min_zero_run_of_words(5), 40);
        assert_eq!(min_zero_run_of_words(usize::MAX), usize::MAX, "saturates");
    }

    #[test]
    #[should_panic(expected = "the smallest is 2")]
    fn t9_min_zero_run_of_words_below_2_panics() {
        min_zero_run_of_words(1);
    }
}
