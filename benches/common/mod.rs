//! The inputs both benchmarks run on: 256 KiB of alternating non-zero literals and zero
//! runs, whose lengths are drawn from a range per [`Shape`].

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
}

pub const SHAPES: [Shape; 5] = [
    // Many tiny blocks: header parsing and call overhead dominate.
    Shape { name: "tiny_blocks", literal_len: (1, 8), zero_len: (3, 8) },
    // Blocks of a few dozen bytes: each copy fits one 64-byte vector.
    Shape { name: "short_blocks", literal_len: (8, 48), zero_len: (3, 48) },
    // Blocks of kilobytes: memcpy and memset speed.
    Shape { name: "long_blocks", literal_len: (512, 4096), zero_len: (512, 4096) },
    // No zero runs: the compressor writes the escape, one literal block.
    Shape { name: "incompressible", literal_len: (INPUT_LEN, INPUT_LEN), zero_len: (0, 0) },
    // Only zeros: one block with a long zero run.
    Shape { name: "all_zeros", literal_len: (0, 0), zero_len: (INPUT_LEN, INPUT_LEN) },
];

/// Builds an input of [`INPUT_LEN`] bytes of the given shape.
///
/// # Arguments
///
/// * `shape` - The literal and zero-run lengths.
///
/// # Returns
///
/// The input; its literals hold no zero byte.
pub fn input(shape: &Shape) -> Vec<u8> {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut bytes = Vec::with_capacity(INPUT_LEN);
    while bytes.len() < INPUT_LEN {
        let literal_len = rng.between(shape.literal_len.0, shape.literal_len.1);
        // `| 1` keeps literal bytes non-zero, so runs are exactly the drawn lengths.
        bytes.extend((0..literal_len).map(|_| rng.next_u64() as u8 | 1));
        let zero_len = rng.between(shape.zero_len.0, shape.zero_len.1);
        bytes.resize(bytes.len() + zero_len, 0);
    }
    bytes.truncate(INPUT_LEN);
    bytes
}

/// Compresses an input into a fresh buffer, with the crate's dispatch.
///
/// # Arguments
///
/// * `src` - The input.
///
/// # Returns
///
/// The compressed bytes.
pub fn compressed(src: &[u8]) -> Vec<u8> {
    let mut dst = vec![0u8; zle::compress_bound(src.len())];
    let written = zle::compress(src, &mut dst);
    dst.truncate(written);
    dst
}
