# zle

Zero-length encoding: every run of at least 3 zero bytes (or of a longer minimum you choose)
is left out of the output and replaced by its length. The format is the one of <https://github.com/riptl/zle>.

The output is a sequence of blocks. A block is some literal bytes, copied as they are,
followed by a count of zeros left out:

```text
block  = header  [leb128(L - 15)]  L literal bytes  [leb128(Z - 15)]
header = min(L, 15) << 4 | min(Z, 15)

input  01 00 00 00 02   →   13 01   10 02
                            └ L=1 Z=3  └ L=1 Z=0
```

If the blocks would not be shorter than the input, the output is the whole input as one
literal block instead, so the output is never more than `src.len() + 8` bytes.

## Usage

```rust
let src = [1, 0, 0, 0, 2];
let mut packed = vec![0u8; zle::compress_bound(src.len())];
let written = zle::compress(&src, &mut packed);
assert_eq!(packed[..written], [0x13, 1, 0x10, 2]);

// The format does not store the original length: the caller passes it along.
let mut out = vec![0u8; src.len()];
assert_eq!(zle::decompress(&packed[..written], &mut out), Ok(src.len()));
assert_eq!(out, src);
```

`compress_with_min_run(src, dst, min_zero_run)` leaves out only runs of at least
`min_zero_run` zeros (3 or more); `compress` is `compress_with_min_run` with 3. A longer
minimum keeps short runs in the literals: the output is larger, but has fewer and longer
blocks, so it decompresses faster, and it compresses faster on data full of short runs.
`decompress` reads blocks written with any minimum.

```rust
let mut src = vec![1];
src.extend([0; 16]);
src.push(2);
let mut packed = vec![0u8; zle::compress_bound(src.len())];
let written = zle::compress_with_min_run(&src, &mut packed, 16);
assert_eq!(packed[..written], [0x1f, 1, 1, 0x10, 2]);
```

`compress_aligned(src, dst, min_zero_words)` leaves out only runs of at least
`min_zero_words` (2 or more) whole 8-byte zero words, aligned to the input's start; zeros
that do not fill a word stay in the literals. Finding a run is one compare per word, with
no search for its exact ends, for slightly larger output than `compress_with_min_run` at
`8 * min_zero_words`.

```rust
// | 1 1 1 1 1 0 0 0 | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 | 0 2 2 2 2 2 2 2 |
let src = [&[1; 5][..], &[0; 20], &[2; 7]].concat();
let mut packed = vec![0u8; zle::compress_bound(src.len())];

// The two whole zero words are left out; the 4 zeros around them stay in the literals.
let written = zle::compress_aligned(&src, &mut packed, 2);
assert_eq!(
    packed[..written],
    [0x8f, 1, 1, 1, 1, 1, 0, 0, 0, 0x01, 0x80, 0, 2, 2, 2, 2, 2, 2, 2]
);

let mut out = vec![0u8; src.len()];
assert_eq!(zle::decompress(&packed[..written], &mut out), Ok(src.len()));
assert_eq!(out, src);
```

`decompress` treats its input as untrusted: a truncated block, a length that overflows or an
output longer than `dst` returns a `DecompressError`, never a panic.

## Implementations

| | x86_64 with AVX-512F, AVX-512BW and BMI2 | other CPUs |
|---|---|---|
| `compress`, `compress_with_min_run`, `compress_aligned` | AVX-512, 64 bytes at a time | scalar, 8 bytes at a time |
| `decompress` | scalar header parsing, AVX-512 masked copies up to 64 bytes | scalar |

Both compressors look for zero runs the same way. A run of at least `2W - 1` zeros always
covers a whole `W`-byte lane aligned to a multiple of `W`, so they look only for zero lanes,
as wide as the minimum allows, and extend each one to the run's real ends:

| minimum run | lane | scalar, per 8 bytes | AVX-512, per 64 bytes |
|---|---|---|---|
| 3 to 6 | 2 bytes | 4 lanes | `vpcmpeqw` |
| 7 to 14 | 4 bytes | 2 lanes | `vpcmpeqd` |
| 15 and more | 8 bytes | 1 compare | `vpcmpeqq` |

A zero byte stops the scan only when it completes a zero lane, so a longer minimum stops on
fewer of them: at a minimum of 16, the scan is one compare per 8 bytes.

Both compressors write the same bytes. The choice is made at compile time when it can be:

- built with the features enabled (`-C target-cpu=x86-64-v4`, for example): AVX-512 is
  called directly;
- built for x86_64 without them: the CPU is checked at run time;
- built for another architecture: scalar.

## Tests

```text
cargo test
```

`.cargo/config.toml` runs test binaries inside [Intel SDE] emulating a Sapphire Rapids CPU,
so the AVX-512 code is tested on CPUs without it. SDE is expected at `.tools/` (a symlink,
ignored by git).

[Intel SDE]: https://www.intel.com/content/www/us/en/developer/articles/tool/software-development-emulator.html

## Benchmarks

`benches/zle.rs` measures throughput, in uncompressed bytes per second, on six 256 KiB
inputs that alternate literals and zero runs with lengths drawn uniformly from:

| input | literal length | zero-run length |
|---|---|---|
| `tiny_blocks` | 1–8 | 3–8 |
| `short_blocks` | 8–48 | 3–48 |
| `long_blocks` | 512–4096 | 512–4096 |
| `incompressible` | the whole input | none |
| `all_zeros` | none | the whole input |
| `account_like` | 64–1024, with 1–6 zeros every 1–24 bytes | 16–512 |

The literals hold no zero byte, except in `account_like`, which looks like Solana account
data: small integers and padding put short zero runs everywhere. Every input is measured at
the default minimum run (groups `compress` and `decompress`) and at 16
(`compress_min_run_16` and `decompress_min_run_16`).

### Results

These were measured before the lane scan above, on the first five inputs at the default
minimum: the compress columns are out of date until measured again on that machine.

`benches/scalar_vs_avx512.rs` runs both implementations on the same inputs. On an Intel Xeon
Platinum 8481C (Sapphire Rapids, 2.7 GHz base), built for `x86-64-v4`; criterion's point
estimate over 100 samples, time per 256 KiB input in parentheses.

**Scalar**

| input | compress | decompress |
|---|---|---|
| `tiny_blocks` | 543 MiB/s (460 µs) | 636 MiB/s (393 µs) |
| `short_blocks` | 1.79 GiB/s (137 µs) | 3.52 GiB/s (69.4 µs) |
| `long_blocks` | 10.7 GiB/s (22.8 µs) | 34.9 GiB/s (7.00 µs) |
| `incompressible` | 8.52 GiB/s (28.7 µs) | 33.9 GiB/s (7.21 µs) |
| `all_zeros` | 12.1 GiB/s (20.3 µs) | 44.8 GiB/s (5.45 µs) |

**AVX-512**

| input | compress | decompress |
|---|---|---|
| `tiny_blocks` | 560 MiB/s (447 µs) | 1.73 GiB/s (141 µs) |
| `short_blocks` | 2.56 GiB/s (95.3 µs) | 4.41 GiB/s (55.4 µs) |
| `long_blocks` | 30.5 GiB/s (8.00 µs) | 34.4 GiB/s (7.10 µs) |
| `incompressible` | 20.9 GiB/s (11.7 µs) | 37.4 GiB/s (6.54 µs) |
| `all_zeros` | 69.4 GiB/s (3.52 µs) | 44.8 GiB/s (5.46 µs) |

- **Compress:** scanning 64 bytes at a time pays off when blocks are long, 2.5× to 5.8×
  faster than scalar. On `tiny_blocks` it gains nothing: about 26,000 blocks of 10 bytes,
  so the time goes to writing blocks, not scanning.
- **Decompress:** headers are still parsed one at a time, but literals and zero runs of up
  to 64 bytes are copied with one masked load and store instead of `memcpy`/`memset` calls.
  That takes `tiny_blocks` from about 15 ns to 5.4 ns per block, 2.8× faster. Long blocks
  already ran at `memcpy`/`memset` speed, and still do.

### Choosing a minimum

| you want | use |
|---|---|
| the smallest output | `compress` (runs of 3 or more) |
| speed, for a slightly larger output | `compress_with_min_run(src, dst, 16)` |
| speed, on a CPU without AVX-512 | `compress_aligned(src, dst, 2)` |
| Solana account data, highest speed | `compress_aligned(src, dst, 2)` with AVX-512 |

- **Minimums of 15 and more compress fastest:** 22.6 GiB/s with AVX-512, against 1.2 at 3
  and 4.4 at 7 to 14.
- **Fewer blocks decompress faster:** 24 GiB/s at 16, against 3 at 3.
- **`compress_aligned` decompresses fastest.** It compresses fastest without AVX-512, and
  within 3% with it; its output is a little larger.

### Solana account data

For the highest speed, use `compress_aligned(src, dst, 2)` on a CPU with AVX-512, ideally
built with `RUSTFLAGS="-C target-cpu=x86-64-v4"`. On `account_like` (synthetic data shaped
like accounts); output is the share of the input, speeds are GiB/s with AVX-512 unless
marked scalar:

| rule | output | compress | compress, scalar | decompress |
|---|---|---|---|---|
| `compress_aligned` 2 | 68.7% | 21.9 | 13.9 | 24.7 |
| `compress_with_min_run` 16 | 67.8% | 22.6 | 9.33 | 24.1 |
| `compress` (3) | 59.8% | 1.23 | 1.36 | 2.95 |

<details>
<summary>Every minimum, on two inputs</summary>

From `benches/min_run_sweep.rs`, on the Xeon above, built for `x86-64-v4`. `min_run_N` is
`compress_with_min_run` at N, `aligned_N` is `compress_aligned` at N/8 words. `mixed_runs`
has zero runs of every length from 1 to 64.

**`account_like`**

| rule | output | blocks | compress scalar (GiB/s) | compress AVX-512 (GiB/s) | decompress scalar (GiB/s) | decompress AVX-512 (GiB/s) |
|---|---|---|---|---|---|---|
| `min_run_3` | 59.8% | 7423 | 1.36 | 1.23 | 2.00 | 2.95 |
| `min_run_4` | 60.9% | 5614 | 1.30 | 1.27 | 2.88 | 4.00 |
| `min_run_6` | 65.1% | 2073 | 1.45 | 1.55 | 10.9 | 11.1 |
| `min_run_8` | 67.8% | 318 | 3.65 | 4.42 | 24.0 | 24.1 |
| `min_run_16` | 67.8% | 318 | 9.33 | 22.6 | 23.9 | 24.1 |
| `aligned_16` | 68.7% | 317 | 13.9 | 21.9 | 24.5 | 24.7 |
| `aligned_24` | 68.7% | 309 | 14.1 | 22.3 | 24.6 | 24.8 |
| `aligned_32` | 68.8% | 301 | 14.3 | 21.4 | 24.8 | 24.9 |

**`mixed_runs`**

| rule | output | blocks | compress scalar (GiB/s) | compress AVX-512 (GiB/s) | decompress scalar (GiB/s) | decompress AVX-512 (GiB/s) |
|---|---|---|---|---|---|---|
| `min_run_3` | 38.7% | 5194 | 1.62 | 1.94 | 3.05 | 3.54 |
| `min_run_4` | 38.7% | 5108 | 1.62 | 1.97 | 3.16 | 3.67 |
| `min_run_6` | 38.9% | 4922 | 1.62 | 2.04 | 3.37 | 3.91 |
| `min_run_8` | 39.2% | 4785 | 1.61 | 2.21 | 3.53 | 4.08 |
| `min_run_16` | 41.8% | 4124 | 2.07 | 2.56 | 4.68 | 5.51 |
| `aligned_16` | 53.8% | 3817 | 2.45 | 2.58 | 4.61 | 6.33 |
| `aligned_24` | 57.3% | 3120 | 2.63 | 2.67 | 5.99 | 7.59 |
| `aligned_32` | 62.8% | 2445 | 2.79 | 2.67 | 8.53 | 9.84 |

</details>

### Running

SDE timings mean nothing, so run natively, bypassing the runner in `.cargo/config.toml`:

```text
CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER=env cargo bench --bench zle
```

To build for an AVX-512 machine and run there, pass `--target` explicitly, so that
`RUSTFLAGS` does not apply to build scripts that run on the build machine:

```text
RUSTFLAGS="-C target-cpu=x86-64-v4" cargo bench --bench zle --no-run \
    --target x86_64-unknown-linux-gnu --target-dir target/v4
```

Copy the executable cargo prints and run it with `--bench` (without it, criterion runs each
benchmark once as a test).

The scalar-versus-AVX-512 comparison needs the `internals` feature, which exposes both
implementations, and an AVX-512 CPU to run on:

```text
RUSTFLAGS="-C target-cpu=x86-64-v4" cargo bench --features internals \
    --bench scalar_vs_avx512 --no-run \
    --target x86_64-unknown-linux-gnu --target-dir target/v4
```

`benches/account_like.rs` compares zle, at the default minimum run and at 16
(`zle_min_run_16`), and `compress_aligned` at 2 words (`zle_aligned_16`), with a plain copy,
on the `account_like` input alone. It prints how many bytes each one writes, then measures
two groups, `account_like_compress` and `account_like_decompress`. Build it the way
excelsior's server is built, and run it on an AVX-512 machine:

```text
RUSTFLAGS="-C target-cpu=x86-64-v4" cargo bench --bench account_like --no-run \
    --target x86_64-unknown-linux-gnu --target-dir target/v4
```

The sweep above needs the `internals` feature and an AVX-512 CPU too:

```text
RUSTFLAGS="-C target-cpu=x86-64-v4" cargo bench --features internals \
    --bench min_run_sweep --no-run \
    --target x86_64-unknown-linux-gnu --target-dir target/v4
```

## Credits

The ZLE format and the design of the AVX-512 compressor come from Richard Patel's
[zle](https://github.com/riptl/zle).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for
inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed
as above, without any additional terms or conditions.
