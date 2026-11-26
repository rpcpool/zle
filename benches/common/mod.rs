//! The inputs both benchmarks run on: 256 KiB of alternating literals and zero runs, whose
//! lengths are drawn from a range per [`Shape`]. A literal is non-zero bytes, or, like real
//! account data, non-zero bytes with short zero runs among them.

/// Uncompressed size of every input: fits in L2, so the benchmark measures the code rather
/// than memory bandwidth.
pub const INPUT_LEN: usize = 256 * 1024;

/// A seeded xorshift64* generator: the same inputs on every run.
struct Rng(u64);

impl Rng {
    /// Returns the next 64 random bits.
    ///
    /// # Returns
    ///
    /// The next value of the sequence.
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// Returns a value in `low..=high`.
    ///
    /// # Arguments
    ///
    /// * `low` - The smallest value.
    /// * `high` - The largest value.
    ///
    /// # Returns
    ///
    /// A value in `low..=high`, close to uniform.
    fn between(&mut self, low: usize, high: usize) -> usize {
        low + (self.next_u64() % (high - low + 1) as u64) as usize
    }
}

/// The shape of one input: alternating non-zero literals and zero runs, each length drawn
/// uniformly from its range.
pub struct Shape {
    /// The benchmark's name for it.
    pub name: &'static str,
    /// The shortest and longest literal.
    pub literal_len: (usize, usize),
    /// The shortest and longest zero run; 0 for none.
    pub zero_len: (usize, usize),
    /// Inside a literal, the shortest and longest stretch of non-zero bytes between two
    /// inner zero runs; 0 for a literal of non-zero bytes only.
    pub inner_gap: (usize, usize),
    /// Inside a literal, the shortest and longest inner zero run.
    pub inner_zero_len: (usize, usize),
}

pub const SHAPES: [Shape; 6] = [
    // Many tiny blocks: header parsing and call overhead dominate.
    Shape {
        name: "tiny_blocks",
        literal_len: (1, 8),
        zero_len: (3, 8),
        ..NO_INNER_ZEROS
    },
    // Blocks of a few dozen bytes: each copy fits one 64-byte vector.
    Shape {
        name: "short_blocks",
        literal_len: (8, 48),
        zero_len: (3, 48),
        ..NO_INNER_ZEROS
    },
    // Blocks of kilobytes: memcpy and memset speed.
    Shape {
        name: "long_blocks",
        literal_len: (512, 4096),
        zero_len: (512, 4096),
        ..NO_INNER_ZEROS
    },
    // No zero runs: the compressor writes the escape, one literal block.
    Shape {
        name: "incompressible",
        literal_len: (INPUT_LEN, INPUT_LEN),
        zero_len: (0, 0),
        ..NO_INNER_ZEROS
    },
    // Only zeros: one block with a long zero run.
    Shape {
        name: "all_zeros",
        literal_len: (0, 0),
        zero_len: (INPUT_LEN, INPUT_LEN),
        ..NO_INNER_ZEROS
    },
    // Like Solana account data: dense bytes with 1 to 6 zeros every 1 to 24 bytes (small
    // integers, padding), and runs of 16 to 512 zeros in between (unused space). With the
    // default minimum of 3 most inner runs end a block; with a minimum of 16 none do.
    Shape {
        name: "account_like",
        literal_len: (64, 1024),
        zero_len: (16, 512),
        inner_gap: (1, 24),
        inner_zero_len: (1, 6),
    },
];

/// The fields of a [`Shape`] whose literals hold no zero byte.
const NO_INNER_ZEROS: Shape = Shape {
    name: "",
    literal_len: (0, 0),
    zero_len: (0, 0),
    inner_gap: (0, 0),
    inner_zero_len: (0, 0),
};

/// Builds an input of [`INPUT_LEN`] bytes of the given shape.
///
/// # Arguments
///
/// * `shape` - The literal and zero-run lengths.
///
/// # Returns
///
/// The input; its literals hold no zero byte unless the shape has inner zero runs.
pub fn input(shape: &Shape) -> Vec<u8> {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut bytes = Vec::with_capacity(INPUT_LEN);
    while bytes.len() < INPUT_LEN {
        let literal_len = rng.between(shape.literal_len.0, shape.literal_len.1);
        let literal_end = bytes.len() + literal_len;
        // Count down to the next inner zero run; none if the shape has no inner zeros.
        let mut until_inner_zeros = next_inner_gap(&mut rng, shape);
        while bytes.len() < literal_end {
            if until_inner_zeros == 0 {
                let inner_zero_len = rng.between(shape.inner_zero_len.0, shape.inner_zero_len.1);
                // Cut at the literal's end, so the zero run after it keeps its drawn length.
                bytes.resize((bytes.len() + inner_zero_len).min(literal_end), 0);
                until_inner_zeros = next_inner_gap(&mut rng, shape);
                continue;
            }
            // `| 1` keeps the byte non-zero, so runs are exactly the drawn lengths.
            bytes.push(rng.next_u64() as u8 | 1);
            until_inner_zeros -= 1;
        }
        let zero_len = rng.between(shape.zero_len.0, shape.zero_len.1);
        bytes.resize(bytes.len() + zero_len, 0);
    }
    bytes.truncate(INPUT_LEN);
    bytes
}

/// Draws how many non-zero bytes come before the next inner zero run.
///
/// # Arguments
///
/// * `rng` - The generator.
/// * `shape` - The shape, whose [`Shape::inner_gap`] is the range.
///
/// # Returns
///
/// A length in [`Shape::inner_gap`], or `usize::MAX` (never) if the shape has no inner zeros.
fn next_inner_gap(rng: &mut Rng, shape: &Shape) -> usize {
    if shape.inner_gap == (0, 0) {
        return usize::MAX;
    }
    rng.between(shape.inner_gap.0, shape.inner_gap.1)
}

/// The minimum runs both benchmarks measure: the default, and the one excelsior uses.
pub const MIN_ZERO_RUNS: [usize; 2] = [zle::MIN_ZERO_RUN, 16];

/// Compresses an input into a fresh buffer, with the crate's dispatch.
///
/// # Arguments
///
/// * `src` - The input.
/// * `min_zero_run` - The shortest run left out.
///
/// # Returns
///
/// The compressed bytes.
pub fn compressed(src: &[u8], min_zero_run: usize) -> Vec<u8> {
    let mut dst = vec![0u8; zle::compress_bound(src.len())];
    let written = zle::compress_with_min_run(src, &mut dst, min_zero_run);
    dst.truncate(written);
    dst
}

/// Names a benchmark group after what it measures and the minimum run.
///
/// # Arguments
///
/// * `what` - `compress`, `decompress`, or an implementation's variant of them.
/// * `min_zero_run` - The shortest run left out.
///
/// # Returns
///
/// `what` alone for the default minimum, else `what` then `_min_run_` and the minimum.
pub fn group_name(what: &str, min_zero_run: usize) -> String {
    if min_zero_run == zle::MIN_ZERO_RUN {
        what.to_string()
    } else {
        format!("{what}_min_run_{min_zero_run}")
    }
}
