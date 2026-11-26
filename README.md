# zle

Zero-length encoding: every run of at least 3 zero bytes is left out of the output and
replaced by its length. The format is the one of <https://github.com/riptl/zle>.

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

`decompress` treats its input as untrusted: a truncated block, a length that overflows or an
output longer than `dst` returns a `DecompressError`, never a panic.

## Implementations

| | x86_64 with AVX-512F, AVX-512BW and BMI2 | other CPUs |
|---|---|---|
| `compress` | AVX-512, 64 bytes at a time | scalar, 8 bytes at a time |
| `decompress` | scalar header parsing, AVX-512 masked copies up to 64 bytes | scalar |

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

`benches/zle.rs` measures throughput, in uncompressed bytes per second, on five 256 KiB
inputs that alternate non-zero literals and zero runs with lengths drawn uniformly from:

| input | literal length | zero-run length |
|---|---|---|
| `tiny_blocks` | 1–8 | 3–8 |
| `short_blocks` | 8–48 | 3–48 |
| `long_blocks` | 512–4096 | 512–4096 |
| `incompressible` | the whole input | none |
| `all_zeros` | none | the whole input |

### Results

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

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for
inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed
as above, without any additional terms or conditions.
