# zle

Zero-length encoding: removes runs of zero bytes from a buffer and puts them back.
Lossless, no allocation, no dependencies, with an AVX-512 path on x86_64. The format is the
one of <https://github.com/riptl/zle>.

## Usage

```rust
// 100 bytes with zero runs of 5, 12 and 40 bytes.
let mut src = vec![0xaa; 100];
src[10..15].fill(0);
src[30..42].fill(0);
src[50..90].fill(0);
let mut dst = vec![0u8; zle::compress_bound(src.len())];

// Every run of 3 or more zeros is left out: the smallest output.
assert_eq!(zle::compress(&src, &mut dst), 49);

// Only runs of at least N zeros are left out; shorter ones are kept as they are.
assert_eq!(zle::compress_with_min_run(&src, &mut dst, 8), 53); // keeps the 5
assert_eq!(zle::compress_with_min_run(&src, &mut dst, 16), 64); // keeps the 5 and 12

// Only whole 8-byte zero words aligned to the start, at least 2 in a row: of the 40
// zeros, the 32 in words 56..88 are left out, the 8 around them kept.
let written = zle::compress_aligned(&src, &mut dst, 2);
assert_eq!(written, 72);

// The output does not store the original length: keep it, and check it on the way back.
let mut out = vec![0u8; src.len()];
assert_eq!(zle::decompress(&dst[..written], &mut out), Ok(src.len()));
assert_eq!(out, src);
```

`decompress` reads the output of any of them, and never panics on bad input: it returns a
`DecompressError`.

## Format

The output is a sequence of blocks: some bytes copied as they are (the literal), then a
count of zeros left out.

```text
stream      = block*
block       = header [literal_ext] literal [zero_ext]
header      = min(L, 15) << 4 | min(Z, 15)    one byte
literal_ext = leb128(L - 15)                 only if L >= 15
literal     = L bytes, copied as they are
zero_ext    = leb128(Z - 15)                 only if Z >= 15

L = literal length, Z = number of zeros after it

input  01 00 00 00 02   →   13 01   10 02
                            └ L=1 Z=3  └ L=1 Z=0
```

If the blocks would not be shorter than the input, the output is the input as one literal
block, so it is never longer than `compress_bound(src.len())` = `src.len() + 8`.

## How to pick

1. **Smallest output:** `compress`.
2. **Fast, for a slightly larger output:** `compress_with_min_run(src, dst, 16)`.
3. **Solana account data, highest speed:** `compress_aligned(src, dst, 2)`.

**Run on a CPU with AVX-512, whichever you pick.** Compress plus decompress is faster with
AVX-512 for all three: 1.07× for `compress`, 1.7× for `compress_with_min_run(.., 16)`, 1.3×
for `compress_aligned(.., 2)`. zle uses AVX-512 by itself when the CPU has it; building with
`RUSTFLAGS="-C target-cpu=x86-64-v4"` also skips the CPU check. The one step where AVX-512
is not ahead: on data shaped like Solana accounts, `compress` alone compresses about 10%
slower with AVX-512, but its output decompresses 1.5× faster. On data whose zeros come in
fewer, longer runs, AVX-512 compresses faster too.

## Benchmark

256 KiB of synthetic data shaped like Solana accounts (`account_like`), on an Intel Xeon
Platinum 8481C, built for `x86-64-v4`. Output is the share of the input; speeds are GiB/s of
uncompressed data.

| call | output | compress, AVX-512 | compress, scalar | decompress |
|---|---|---|---|---|
| `compress` | 59.8% | 1.23 | 1.36 | 2.95 |
| `compress_with_min_run(.., 8)` | 67.8% | 4.42 | 3.65 | 24.1 |
| `compress_with_min_run(.., 16)` | 67.8% | 22.6 | 9.33 | 24.1 |
| `compress_aligned(.., 2)` | 68.7% | 21.9 | 13.9 | 24.7 |

The benchmarks are in `benches/`. Timings under the test runner's emulator mean nothing:
build for the target machine and run there.

```text
RUSTFLAGS="-C target-cpu=x86-64-v4" cargo bench --features internals \
    --bench min_run_sweep --no-run \
    --target x86_64-unknown-linux-gnu --target-dir target/v4
```

## Tests

```text
cargo test
```

`.cargo/config.toml` runs the tests inside [Intel SDE], emulating a Sapphire Rapids CPU, so
the AVX-512 code is tested on machines without it. SDE is expected at `.tools/`.

[Intel SDE]: https://www.intel.com/content/www/us/en/developer/articles/tool/software-development-emulator.html

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option, except for the parts derived from Richard Patel's
[zle](https://github.com/riptl/zle), Copyright (c) 2026 Richard Patel, which are under the
Apache License, Version 2.0 only: the vector helpers `or3` and `is_zero_256` in
`src/avx512/ops.rs`, and the masked copy and zero helpers in `src/avx512/decompress.rs`.

Unless you explicitly state otherwise, any contribution intentionally submitted for
inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed
as above, without any additional terms or conditions.
