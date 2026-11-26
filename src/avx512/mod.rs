//! The AVX-512 compressor: it looks for aligned zero lanes 64 bytes at a time with one
//! compare per vector, and skips 256 bytes at a time inside long zero runs and long
//! literals. It writes exactly the bytes of [`crate::scalar::compress`], and
//! [`compress_aligned`] those of [`crate::scalar::compress_aligned`].

mod decompress;
mod ops;

pub use decompress::decompress;

use std::arch::x86_64::*;

use crate::runs::{self, LaneWidth};
use ops::{has_zero_lane_256, is_zero_256, load_tail, zero_lane_mask};

/// Returns whether the CPU has every feature [`compress`] and [`decompress`] enable.
///
/// # Returns
///
/// `true` if it has AVX-512F, AVX-512BW and BMI2.
pub fn is_supported() -> bool {
    // std caches the CPUID results: each check is a load and a test.
    is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("bmi2")
}

/// Compresses bytes, 64 at a time with AVX-512.
///
/// # Arguments
///
/// * `src` - The input.
/// * `dst` - Where the blocks go, at least [`crate::compress_bound`] of the input.
/// * `min_zero_run` - The shortest run left out, at least [`crate::MIN_ZERO_RUN`].
///
/// # Returns
///
/// How many bytes were written; the same bytes as [`crate::scalar::compress`].
///
/// # Panics
///
/// Panics if `dst` is shorter than [`crate::compress_bound`], or `min_zero_run` is below
/// [`crate::MIN_ZERO_RUN`].
///
/// # Safety
///
/// The CPU must have AVX-512F, AVX-512BW and BMI2: see [`is_supported`].
#[target_feature(enable = "avx512f,avx512bw,bmi2")]
pub fn compress(src: &[u8], dst: &mut [u8], min_zero_run: usize) -> usize {
    // The closures inherit this function's target features, so the scanners inline into
    // runs::compress, which inlines here.
    match LaneWidth::for_min_zero_run(min_zero_run) {
        LaneWidth::Two => runs::compress::<2>(
            src,
            dst,
            min_zero_run,
            |src, position| find_zero_lane::<2>(src, position),
            runs::zeros_before_lane::<2>,
            |src, position| end_of_zeros(src, position),
        ),
        LaneWidth::Four => runs::compress::<4>(
            src,
            dst,
            min_zero_run,
            |src, position| find_zero_lane::<4>(src, position),
            runs::zeros_before_lane::<4>,
            |src, position| end_of_zeros(src, position),
        ),
        LaneWidth::Eight => runs::compress::<8>(
            src,
            dst,
            min_zero_run,
            |src, position| find_zero_lane::<8>(src, position),
            runs::zeros_before_lane::<8>,
            |src, position| end_of_zeros(src, position),
        ),
    }
}

/// Compresses bytes by the aligned rule of [`crate::compress_aligned`], 64 at a time with
/// AVX-512.
///
/// # Arguments
///
/// * `src` - The input.
/// * `dst` - Where the blocks go, at least [`crate::compress_bound`] of the input.
/// * `min_zero_words` - The fewest zero words a run has, at least [`crate::MIN_ZERO_WORDS`].
///
/// # Returns
///
/// How many bytes were written; the same bytes as [`crate::scalar::compress_aligned`].
///
/// # Panics
///
/// Panics if `dst` is shorter than [`crate::compress_bound`], or `min_zero_words` is below
/// [`crate::MIN_ZERO_WORDS`].
///
/// # Safety
///
/// The CPU must have AVX-512F, AVX-512BW and BMI2: see [`is_supported`].
#[target_feature(enable = "avx512f,avx512bw,bmi2")]
pub fn compress_aligned(src: &[u8], dst: &mut [u8], min_zero_words: usize) -> usize {
    // A zero lane of 8 is a zero word: find_zero_lane::<8> finds the next one.
    runs::compress::<8>(
        src,
        dst,
        runs::min_zero_run_of_words(min_zero_words),
        |src, position| find_zero_lane::<8>(src, position),
        runs::zeros_before_word,
        |src, position| end_of_zero_words(src, position),
    )
}

/// Returns where the next aligned zero lane is, 64 bytes at a time.
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
#[target_feature(enable = "avx512f,avx512bw,bmi2")]
fn find_zero_lane<const LANE: usize>(src: &[u8], mut position: usize) -> usize {
    let src_len = src.len();
    // `position` stays a multiple of LANE, so the lanes of every vector are aligned too.
    while position + 64 <= src_len {
        // SAFETY: position + 64 <= src_len, so the 64 bytes are in `src`.
        let vector = unsafe { _mm512_loadu_si512(src.as_ptr().add(position).cast()) };
        let lanes = zero_lane_mask::<LANE>(vector);
        if lanes != 0 {
            return position + lanes.trailing_zeros() as usize * LANE;
        }
        position += 64;
        // A whole vector without a zero lane: likely a long literal, so skip ahead 256
        // bytes at a time while none holds one.
        while position + 256 <= src_len {
            let block: &[u8; 256] = src[position..position + 256].try_into().unwrap();
            if has_zero_lane_256::<LANE>(block) {
                break;
            }
            position += 256;
        }
    }
    if position >= src_len {
        return src_len;
    }
    // The last 1 to 63 bytes: the bytes past the end load as zeros, so keep only the lanes
    // wholly inside `src`.
    let whole_lanes = (src_len - position) / LANE;
    let lanes = zero_lane_mask::<LANE>(load_tail(&src[position..]))
        & _bzhi_u64(u64::MAX, whole_lanes as u32);
    if lanes == 0 {
        return src_len;
    }
    position + lanes.trailing_zeros() as usize * LANE
}

/// Returns where a stretch of zero bytes ends, 64 bytes at a time.
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
#[target_feature(enable = "avx512f,avx512bw,bmi2")]
fn end_of_zeros(src: &[u8], mut position: usize) -> usize {
    let src_len = src.len();
    while position + 64 <= src_len {
        // SAFETY: position + 64 <= src_len, so the 64 bytes are in `src`.
        let vector = unsafe { _mm512_loadu_si512(src.as_ptr().add(position).cast()) };
        let nonzero = _mm512_test_epi8_mask(vector, vector);
        if nonzero != 0 {
            return position + nonzero.trailing_zeros() as usize;
        }
        position += 64;
        // A whole vector of zeros: a long run is likely, so skip ahead 256 bytes at a time
        // while they are all zero.
        while position + 256 <= src_len {
            let block: &[u8; 256] = src[position..position + 256].try_into().unwrap();
            if !is_zero_256(block) {
                break;
            }
            position += 256;
        }
    }
    if position >= src_len {
        return src_len;
    }
    // The last 1 to 63 bytes: the bytes past the end load as zeros, so they never count as
    // the non-zero byte that ends the run.
    let vector = load_tail(&src[position..]);
    let nonzero = _mm512_test_epi8_mask(vector, vector);
    if nonzero == 0 {
        return src_len;
    }
    position + nonzero.trailing_zeros() as usize
}

/// Returns where a run of zero words ends, 64 bytes at a time.
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
#[target_feature(enable = "avx512f,avx512bw,bmi2")]
fn end_of_zero_words(src: &[u8], mut position: usize) -> usize {
    let src_len = src.len();
    while position + 64 <= src_len {
        // SAFETY: position + 64 <= src_len, so the 64 bytes are in `src`.
        let vector = unsafe { _mm512_loadu_si512(src.as_ptr().add(position).cast()) };
        // 8 bits, one per word: the first clear one is the first non-zero word.
        let nonzero_words = !zero_lane_mask::<8>(vector) & 0xff;
        if nonzero_words != 0 {
            return position + nonzero_words.trailing_zeros() as usize * 8;
        }
        position += 64;
        // A whole vector of zero words: a long run is likely, so skip ahead 256 bytes at a
        // time while they are all zero.
        while position + 256 <= src_len {
            let block: &[u8; 256] = src[position..position + 256].try_into().unwrap();
            if !is_zero_256(block) {
                break;
            }
            position += 256;
        }
    }
    // The last 0 to 63 bytes: only their whole words count. The bytes past the end load as
    // zeros, so keep the words wholly inside `src`, and end at the last of them if all zero.
    let whole_words = (src_len - position) / 8;
    let in_src = _bzhi_u64(u64::MAX, whole_words as u32);
    let nonzero_words = !zero_lane_mask::<8>(load_tail(&src[position..])) & in_src;
    if nonzero_words == 0 {
        return position + 8 * whole_words;
    }
    position + nonzero_words.trailing_zeros() as usize * 8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{
        Guarded, MIN_ZERO_RUNS, Rng, aligned_edge_cases, decompress, reference_compressed,
        reference_compressed_aligned, require_avx512, sparse,
    };
    use crate::{MIN_ZERO_RUN, compress_bound, escape_len};

    fn compressed(src: &[u8], min_zero_run: usize) -> Vec<u8> {
        require_avx512();
        let mut dst = vec![0u8; compress_bound(src.len())];
        // SAFETY: require_avx512 checked the CPU has AVX-512F, BW and BMI2.
        let written = unsafe { compress(src, &mut dst, min_zero_run) };
        dst.truncate(written);
        dst
    }

    fn assert_matches_reference(src: &[u8], min_zero_run: usize, what: &str) {
        assert_eq!(
            compressed(src, min_zero_run),
            reference_compressed(src, min_zero_run),
            "min {min_zero_run}, {what}"
        );
    }

    #[test]
    fn t1_matches_the_reference_and_round_trips() {
        // Lengths up to 2000: below 64, partial last vectors, runs crossing vectors, and long
        // runs that take the 256-byte skips.
        let mut rng = Rng::new(1);
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
    fn t2_matches_the_reference_on_vector_edges() {
        // A run of `zeros` zeros starting at `start`, between non-zero bytes, for starts and
        // lengths around the 64-byte vector boundaries.
        for &min_zero_run in &[3, 6, 7, 14, 15, 16] {
            // Tails after the last whole vector of 1 byte, of whole lanes (66, 68, 72, 120)
            // and of 63 bytes.
            for len in [1, 15, 63, 64, 65, 66, 68, 72, 120, 127, 128, 129, 200] {
                for start in 0..len {
                    for zeros in [1, 2, min_zero_run - 1, min_zero_run, 64, 65] {
                        let mut src = vec![0x5a; len];
                        src[start..(start + zeros).min(len)].fill(0);
                        assert_matches_reference(
                            &src,
                            min_zero_run,
                            &format!("len {len}, {zeros} zeros at {start}"),
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn t3_matches_the_reference_on_long_zero_runs() {
        // All zeros, and long runs ending before, at and after a 256-byte skip's edge.
        for &min_zero_run in &MIN_ZERO_RUNS {
            for len in [64, 256, 320, 1000, 1024, 1100] {
                assert_matches_reference(&vec![0u8; len], min_zero_run, &format!("{len} zeros"));
            }
            for zeros in [255, 256, 257, 319, 320, 321, 600] {
                let mut src = vec![0x11; 70];
                src.extend(std::iter::repeat_n(0, zeros));
                src.extend([0x22; 70]);
                assert_matches_reference(
                    &src,
                    min_zero_run,
                    &format!("{zeros} zeros between literals"),
                );
            }
        }
    }

    #[test]
    fn t4_matches_the_reference_on_long_literals() {
        // A long literal with runs one short of the minimum every few bytes (never a run,
        // but often a zero lane), and one run of exactly the minimum at every position
        // across several 256-byte skips, including their last bytes.
        for &min_zero_run in &[3, 7, 15, 16] {
            let period = min_zero_run + 5;
            let literal: Vec<u8> = (0..1400)
                .map(|index| {
                    if index % period < min_zero_run - 1 {
                        0
                    } else {
                        0x33
                    }
                })
                .collect();
            assert_matches_reference(&literal, min_zero_run, "no run at all");
            let dense = vec![0x33u8; 1400];
            for start in 0..900 {
                for base in [&literal, &dense] {
                    let mut src = base.clone();
                    src[start..start + min_zero_run].fill(0);
                    assert_matches_reference(&src, min_zero_run, &format!("run at {start}"));
                }
            }
        }
    }

    #[test]
    fn t5_matches_the_reference_on_the_escape() {
        let mut src = vec![0x11; 143];
        src.extend([0; 3]);
        src.extend([0x22; 143]);
        let packed = compressed(&src, MIN_ZERO_RUN);
        assert_eq!(packed.len(), escape_len(src.len()));
        assert_eq!(packed, reference_compressed(&src, MIN_ZERO_RUN));
    }

    #[test]
    fn t6_never_reads_past_the_input() {
        // The input ends right before an unmapped page: one byte read past it is SIGSEGV.
        // Literals and zeros at the end, for every length of the last partial vector.
        for &min_zero_run in &[3, 7, 16] {
            for fill in [0x00, 0x5a] {
                let mut guarded = Guarded::new(fill);
                for len in 0..=400 {
                    let src = guarded.tail(len);
                    if len >= 10 {
                        src[len / 2..len / 2 + 5].fill(fill ^ 0x5a);
                    }
                    let src = &*src;
                    assert_matches_reference(
                        src,
                        min_zero_run,
                        &format!("{len} bytes of {fill:#04x} at a page's end"),
                    );
                    // Put the page back for the next length.
                    guarded.tail(len).fill(fill);
                }
            }
        }
    }

    fn check_find_zero_lane<const LANE: usize>(rng: &mut Rng) {
        for case in 0..100 {
            let src = sparse(rng, 700, 2 * LANE);
            for position in (0..=src.len() + LANE).step_by(LANE) {
                let expected = (position..src.len())
                    .step_by(LANE)
                    .find(|&lane| {
                        lane + LANE <= src.len()
                            && src[lane..lane + LANE].iter().all(|&byte| byte == 0)
                    })
                    .unwrap_or(src.len());
                // SAFETY: the caller ran require_avx512.
                let found = unsafe { find_zero_lane::<LANE>(&src, position) };
                assert_eq!(
                    found, expected,
                    "lanes of {LANE}, case {case}, from {position}"
                );
            }
        }
    }

    #[test]
    fn t7_find_zero_lane_matches_a_lane_scan() {
        require_avx512();
        let mut rng = Rng::new(7);
        check_find_zero_lane::<2>(&mut rng);
        check_find_zero_lane::<4>(&mut rng);
        check_find_zero_lane::<8>(&mut rng);
    }

    #[test]
    fn t8_end_of_zeros_matches_a_byte_scan() {
        require_avx512();
        let mut rng = Rng::new(8);
        for case in 0..100 {
            let src = sparse(&mut rng, 700, 64);
            for position in 0..=src.len() {
                let end = (position..src.len())
                    .find(|&index| src[index] != 0)
                    .unwrap_or(src.len());
                // SAFETY: require_avx512 checked the CPU has AVX-512F, BW and BMI2.
                assert_eq!(
                    unsafe { end_of_zeros(&src, position) },
                    end,
                    "case {case}, from {position}"
                );
            }
        }
    }

    #[test]
    fn t9_scanners_never_read_past_the_input() {
        require_avx512();
        for fill in [0x00, 0x5a] {
            let mut guarded = Guarded::new(fill);
            for len in 0..=200 {
                let src = &*guarded.tail(len);
                for position in (0..=len).step_by(8) {
                    // SAFETY: require_avx512 checked the CPU has AVX-512F, BW and BMI2.
                    unsafe {
                        find_zero_lane::<2>(src, position);
                        find_zero_lane::<4>(src, position);
                        find_zero_lane::<8>(src, position);
                        end_of_zeros(src, position);
                    }
                }
            }
        }
    }

    fn compressed_aligned(src: &[u8], min_zero_words: usize) -> Vec<u8> {
        require_avx512();
        let mut dst = vec![0u8; compress_bound(src.len())];
        // SAFETY: require_avx512 checked the CPU has AVX-512F, BW and BMI2.
        let written = unsafe { compress_aligned(src, &mut dst, min_zero_words) };
        dst.truncate(written);
        dst
    }

    #[test]
    fn t10_aligned_matches_the_reference_on_edge_cases() {
        // Among them, runs starting, ending and crossing each 64-byte vector boundary, and
        // long runs that take the 256-byte skip.
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
    fn t11_aligned_matches_the_reference_and_round_trips() {
        let mut rng = Rng::new(11);
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
    fn t12_aligned_never_reads_past_the_input() {
        // Zero words, partial words and literals at the end of a page, for every length of
        // the last partial vector.
        for fill in [0x00, 0x5a] {
            let mut guarded = Guarded::new(fill);
            for len in 0..=400 {
                let src = guarded.tail(len);
                if len >= 40 {
                    src[len / 2..len / 2 + 5].fill(fill ^ 0x5a);
                }
                let src = &*src;
                assert_eq!(
                    compressed_aligned(src, 2),
                    reference_compressed_aligned(src, 2),
                    "{len} bytes of {fill:#04x} at a page's end"
                );
                guarded.tail(len).fill(fill);
            }
        }
    }

    #[test]
    fn t13_end_of_zero_words_matches_a_word_scan() {
        require_avx512();
        let mut rng = Rng::new(13);
        for case in 0..100 {
            let src = sparse(&mut rng, 700, 64);
            let is_zero_word = |word: usize| src[word..word + 8].iter().all(|&byte| byte == 0);
            let last_word_end = src.len() / 8 * 8;
            for position in (0..=src.len()).step_by(8) {
                let end = (position..last_word_end)
                    .step_by(8)
                    .find(|&word| !is_zero_word(word))
                    .unwrap_or(last_word_end);
                // SAFETY: require_avx512 checked the CPU has AVX-512F, BW and BMI2.
                let found = unsafe { end_of_zero_words(&src, position) };
                assert_eq!(found, end, "case {case}, from {position}");
            }
        }
        // Past the end of a page: the scan must stop at the last whole word.
        let mut guarded = Guarded::new(0);
        for len in 0..=200 {
            let src = &*guarded.tail(len);
            for position in (0..=len).step_by(8) {
                // SAFETY: as above.
                let found = unsafe { end_of_zero_words(src, position) };
                assert_eq!(
                    found,
                    len / 8 * 8,
                    "{len} zeros at a page's end, from {position}"
                );
            }
        }
    }
}
