# Streaming STFT (pfstream) — Acceptance Report

Real-time spectrum analysis as a continuous, bounded-memory stream built on
top of PFFFT. All paths use the repository's own CMake/CTest entry points.

## What was added

- `include/pffft/pfstream.h` — public C API (configurable block/hop,
  decoupled push/pull).
- `src/pfstream.c` — streaming STFT implementation.
- `tests/test_pfstream.c` — acceptance test, registered as CTest
  `test_pfstream`.
- `CMakeLists.txt` / `tests/CMakeLists.txt` — new `PFSTREAM` library and
  test (float builds only, mirroring `PFFASTCONV`).
- `README.md` — usage and design notes.

## Configuration and API

- `pffft_stream_new(block_len, hop)`: block length `N` must be a valid
  PFFFT real size; hop `H` satisfies `1 <= H <= N`, so overlap `N-H` is
  freely configurable (no overlap `H=N` up to max overlap `H=1`).
- `pffft_stream_push()` accepts chunks of any size; `pffft_stream_pull()`
  returns the oldest ordered spectrum (`N/2` interleaved complex bins).
- Back-pressure: when a spectrum is waiting, `push()` returns
  `PFSTREAM_OUTPUT_FULL` and consumes zero samples. The caller `pull()`s
  and retries the identical samples, so no sample is ever dropped and
  block boundaries never break.

## Bound 1 — streaming equals offline, with bounded error

For frame `f` both pipelines evaluate the identical mathematical object.
The offline reference windows the signal segment `x[fH .. fH+N)` with the
same periodic Hann window and calls the same `pffft_transform_ordered`;
the streamer stores exactly those samples in its ring, applies the same
window while assembling the frame, and calls the same transform. The ring
advances by `H` after every frame, which is precisely the offline
segmentation.

Hence every streamed coefficient equals the offline coefficient up to the
floating-point rounding of the shared FFT kernel. A standard forward-error
bound for an `N`-point FFT in unit-roundoff `eps` is

```
|S_stream[k] - S_offline[k]|  <=  eps * c * N * max_n |w[n] x[n]|
```

with a small kernel-dependent constant `c`. For the test signal
(`N=64`, `eps ~= 6e-8`, windowed magnitude <= 1.2) the conservative
bound is about `4.6e-6`; the measured maximum absolute error is
`1.91e-6`, inside the bound. The test additionally checks every frame
over the whole 1000-sample signal, and compares one-sample, fixed-size
(13) and one-shot pushes: all produce byte-identical frame sequences,
proving the result is independent of how the stream is chunked.

## Bound 2 — memory is at most four blocks and never grows

The stream owns exactly four `N`-float buffers:

1. input ring (`ring`),
2. windowed frame that is also the in-place FFT input/output (`frame`),
3. pending ordered spectrum (`spec`),
4. FFT scratch (`work`).

The Hann window is evaluated during frame assembly, so it needs no fifth
buffer. `pffft_stream_buffer_bytes()` returns
`4 * block_len * sizeof(float)`. The ring holds at most `N-1` live samples
and the pending slot at most one spectrum, both independent of the total
number of samples processed. The test processes a marker stretch, then
20x more data, and asserts the footprint is unchanged and ring occupancy
stays below one block. Memory is therefore `O(N)`, not `O(stream length)`.

## Reproduce

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
(cd build && ctest -R test_pfstream --output-on-failure)
./build/tests/test_pfstream
```

Also verified: `PFFFT_USE_SIMD=OFF` (scalar) and
`PFFFT_USE_TYPE_FLOAT=OFF` (pfstream correctly excluded); full suite
`ctest` = 16/16 passed; AddressSanitizer + LeakSanitizer clean.
