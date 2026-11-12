/*
  Copyright (c) 2026

  Acceptance tests for the bounded-memory streaming STFT (pfstream).

  Proves:
   1. each streamed frame equals the offline whole-signal STFT frame
      (bounded, measured error vs an analytic bound);
   2. block boundaries never break: one-sample, fixed-size and huge
      pushes produce byte-identical frame sequences;
   3. back-pressure never drops samples (OUTPUT_FULL -> retry);
   4. data memory stays at four blocks regardless of stream length.
*/

#include "pffft/pfstream.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define BLOCK_LEN 64
#define HOP_50    32
#define HOP_25    16
#define STREAM_LEN 1000

static int g_failures = 0;

#define CHECK(cond, msg) do {                                       \
    if (!(cond)) {                                                  \
      printf("  FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__);    \
      ++g_failures;                                                 \
    } else {                                                        \
      printf("  ok:   %s\n", (msg));                                \
    }                                                               \
  } while (0)

static void hann_window(float *w, int N)
{
  int i;
  for (i = 0; i < N; ++i)
    w[i] = 0.5f * (1.0f - (float)cos(2.0 * M_PI * (double)i / (double)N));
}

/* tiny synthetic material: two sinusoids plus an amplitude step */
static void make_signal(float *x, int n)
{
  int i;
  for (i = 0; i < n; ++i) {
    float t = (float)i;
    float env = i < n / 2 ? 0.7f : 1.0f;
    x[i] = env * (0.8f * sinf(2.0f * (float)M_PI * 3.0f * t / (float)BLOCK_LEN)
                 + 0.3f * sinf(2.0f * (float)M_PI * 7.0f * t / (float)BLOCK_LEN));
  }
}

/* offline reference: windowed, ordered forward FFT of one frame */
static void offline_frame(const float *signal, const float *window, int N,
                          float *out)
{
  PFFFT_Setup *setup = pffft_new_setup(N, PFFFT_REAL);
  float *in = (float *)pffft_aligned_malloc((size_t)N * sizeof(float));
  float *work = (float *)pffft_aligned_malloc((size_t)N * sizeof(float));
  int i;

  for (i = 0; i < N; ++i)
    in[i] = signal[i] * window[i];
  pffft_transform_ordered(setup, in, out, work, PFFFT_FORWARD);

  pffft_aligned_free(work);
  pffft_aligned_free(in);
  pffft_destroy_setup(setup);
}

/* run the whole signal through a stream with arbitrary push granularity */
static int run_stream(const float *signal, int len, int N, int hop,
                      int push_size, int max_frames, float *frames_out,
                      long long *indices_out, int *backpressure_events)
{
  PFFFT_Stream *st = pffft_stream_new(N, hop);
  int pos = 0, frames = 0, bp = 0;
  int range_errors = 0, pull_errors = 0, index_gaps = 0, bound_errors = 0;
  float *out = (float *)pffft_aligned_malloc((size_t)N * sizeof(float));

  while (pos < len) {
    pffft_stream_status_t s;
    int take = push_size;
    int consumed;
    if (take > len - pos)
      take = len - pos;

    consumed = pffft_stream_push(st, signal + pos, take, &s);
    if (consumed < 0 || consumed > take)
      ++range_errors;
    pos += consumed;

    if (s == PFSTREAM_OUTPUT_FULL) {
      long long idx;
      ++bp;
      if (pffft_stream_pull(st, out, &idx) != PFSTREAM_OK)
        ++pull_errors;
      if (indices_out && idx != frames)
        ++index_gaps;
      if (frames >= max_frames)
        ++bound_errors;
      if (frames_out)
        memcpy(frames_out + (long long)frames * N, out,
               (size_t)N * sizeof(float));
      if (indices_out)
        indices_out[frames] = idx;
      ++frames;
      /* pos already advanced by 'consumed' (0 here): retry on next loop */
      continue;
    }

    while (pffft_stream_pending(st)) {
      long long idx;
      if (pffft_stream_pull(st, out, &idx) != PFSTREAM_OK)
        ++pull_errors;
      if (indices_out && idx != frames)
        ++index_gaps;
      if (frames >= max_frames)
        ++bound_errors;
      if (frames_out)
        memcpy(frames_out + (long long)frames * N, out,
               (size_t)N * sizeof(float));
      if (indices_out)
        indices_out[frames] = idx;
      ++frames;
    }
  }

  CHECK(range_errors == 0, "every push reports a valid consumed count");
  CHECK(pull_errors == 0, "all pulls succeed, no sample lost at boundaries");
  CHECK(index_gaps == 0, "frame indices are contiguous");
  CHECK(bound_errors == 0, "all frames fit the caller buffer");

  if (backpressure_events)
    *backpressure_events = bp;
  pffft_aligned_free(out);
  pffft_stream_destroy(st);
  return frames;
}

static void test_equivalence_and_error_bound(const float *signal)
{
  const int N = BLOCK_LEN, hop = HOP_50;
  const int expect_frames = (STREAM_LEN - N) / hop + 1;
  const int max_frames = expect_frames;
  float *window = (float *)malloc((size_t)N * sizeof(float));
  float *streamed = (float *)malloc((size_t)max_frames * N * sizeof(float));
  float *ref = (float *)pffft_aligned_malloc((size_t)N * sizeof(float));
  float max_abs = 0.0f, sum_abs = 0.0f, sum_ref = 0.0f;
  int frames, f;

  printf("[1] streaming vs offline whole-signal STFT\n");
  hann_window(window, N);
  frames = run_stream(signal, STREAM_LEN, N, hop, 7, max_frames,
                      streamed, NULL, NULL);
  CHECK(frames == expect_frames, "frame count matches the offline segmentation");

  for (f = 0; f < frames; ++f) {
    int start = f * hop, k;
    offline_frame(signal + start, window, N, ref);
    for (k = 0; k < N; ++k) {
      float got = streamed[(long long)f * N + k];
      float d = fabsf(got - ref[k]);
      if (d > max_abs)
        max_abs = d;
      sum_abs += d;
      sum_ref += fabsf(ref[k]);
    }
  }

  /* Same window and the same FFT kernel on both paths, so results are
     bit-identical in exact arithmetic. The conservative analytic bound
     for the floating-point transform is eps*N*max|x|; the energy-scaled
     mean error (robust to near-zero bins) is reported alongside. */
  printf("    max abs error = %.3g, mean energy-scaled error = %.3g, "
         "analytic bound eps*N*max|x| ~= %.3g\n",
         (double)max_abs, (double)(sum_abs / (sum_ref + 1e-30f)),
         6e-8 * (double)N * 1.2);
  CHECK(max_abs <= 6e-8 * (double)N * 1.2, "error within analytic bound");

  pffft_aligned_free(ref);
  free(streamed);
  free(window);
}

static void test_boundary_continuity(const float *signal)
{
  const int N = BLOCK_LEN, hop = HOP_25;
  const int expect_frames = (STREAM_LEN - N) / hop + 1;
  const int max_frames = expect_frames;
  const long long bytes = (long long)max_frames * N * sizeof(float);
  float *one = (float *)malloc((size_t)bytes);
  float *chunked = (float *)malloc((size_t)bytes);
  float *huge = (float *)malloc((size_t)bytes);
  long long *idx_one = (long long *)malloc((size_t)max_frames * sizeof(long long));
  long long *idx_huge = (long long *)malloc((size_t)max_frames * sizeof(long long));
  int f1, f2, f3, bp = 0;

  printf("[2] boundaries never break: push granularity independence\n");
  f1 = run_stream(signal, STREAM_LEN, N, hop, 1, max_frames, one, idx_one, NULL);
  f2 = run_stream(signal, STREAM_LEN, N, hop, 13, max_frames, chunked, NULL, NULL);
  f3 = run_stream(signal, STREAM_LEN, N, hop, STREAM_LEN, max_frames,
                  huge, idx_huge, &bp);

  CHECK(f1 == f2 && f2 == f3 && f3 == expect_frames,
        "all push sizes yield the same frame count");
  CHECK(memcmp(one, chunked, (size_t)bytes) == 0,
        "1-sample pushes match 13-sample pushes");
  CHECK(memcmp(one, huge, (size_t)bytes) == 0,
        "1-sample pushes match one-shot push");
  CHECK(memcmp(idx_one, idx_huge, (size_t)f1 * sizeof(long long)) == 0,
        "frame indices identical across push sizes");
  CHECK(bp > 0, "one-shot push exercised back-pressure");

  free(idx_huge);
  free(idx_one);
  free(huge);
  free(chunked);
  free(one);
}

static void test_config_validation(void)
{
  PFFFT_Stream *st;
  printf("[3] configuration validation\n");
  CHECK(pffft_stream_new(31, 16) == NULL, "reject invalid block length");
  CHECK(pffft_stream_new(BLOCK_LEN, 0) == NULL, "reject zero hop");
  CHECK(pffft_stream_new(BLOCK_LEN, BLOCK_LEN + 1) == NULL,
        "reject hop larger than block");
  st = pffft_stream_new(BLOCK_LEN, BLOCK_LEN);
  CHECK(st != NULL,
        "accept zero overlap (hop == block)");
  pffft_stream_destroy(st);
  st = pffft_stream_new(BLOCK_LEN, 1);
  CHECK(st != NULL, "accept max overlap (hop 1)");
  pffft_stream_destroy(st);
}

static void test_memory_bounded(const float *signal)
{
  const int N = BLOCK_LEN, hop = HOP_50;
  const size_t four_blocks = (size_t)4 * N * sizeof(float);
  PFFFT_Stream *st = pffft_stream_new(N, hop);
  float *out = (float *)pffft_aligned_malloc((size_t)N * sizeof(float));
  size_t bytes_before, bytes_after;
  long long frames_at_marker, frames_later;
  int pos = 0, drain_errors = 0;

  printf("[4] memory bounded by four blocks, independent of stream length\n");
  CHECK(st != NULL, "stream created");
  CHECK(pffft_stream_buffer_bytes(st) == four_blocks,
        "internal data memory is exactly four blocks");
  CHECK(pffft_stream_buffered(st) == 0 && pffft_stream_pending(st) == 0,
        "fresh stream holds no samples");

  bytes_before = pffft_stream_buffer_bytes(st);

  /* drain a first long stretch, record the frontier */
  while (pos < 400) {
    pffft_stream_status_t s;
    int consumed = pffft_stream_push(st, signal + pos, 37, &s);
    pos += consumed;
    while (pffft_stream_pending(st))
      if (pffft_stream_pull(st, out, NULL) != PFSTREAM_OK)
        ++drain_errors;
  }
  frames_at_marker = pffft_stream_frames_done(st);
  CHECK(drain_errors == 0, "all marker frames drained");
  CHECK(pffft_stream_buffered(st) < N && pffft_stream_pending(st) == 0,
        "ring occupancy stays below one block after draining");

  /* process much more data, then compare the footprint again */
  while (pos < STREAM_LEN * 20) {
    pffft_stream_status_t s;
    int take = 123;
    if (pos + take > STREAM_LEN * 20)
      take = STREAM_LEN * 20 - pos;
    pos += pffft_stream_push(st, signal + (pos % (STREAM_LEN - take)),
                             take, &s);
    while (pffft_stream_pending(st))
      if (pffft_stream_pull(st, out, NULL) != PFSTREAM_OK)
        ++drain_errors;
  }
  frames_later = pffft_stream_frames_done(st);
  bytes_after = pffft_stream_buffer_bytes(st);

  CHECK(drain_errors == 0, "all tail frames drained");
  CHECK(frames_later > frames_at_marker * 10,
        "stream length grew by an order of magnitude");
  CHECK(bytes_after == bytes_before && bytes_after == four_blocks,
        "footprint unchanged after 20x more data");
  CHECK(pffft_stream_buffered(st) < N, "no unbounded sample accumulation");

  pffft_aligned_free(out);
  pffft_stream_destroy(st);
}

int main(void)
{
  float *signal = (float *)malloc((size_t)STREAM_LEN * sizeof(float));

  make_signal(signal, STREAM_LEN);
  printf("pfstream acceptance: N=%d, signal=%d samples, simd=%d (%s)\n\n",
         BLOCK_LEN, STREAM_LEN, pffft_simd_size(), pffft_simd_arch());

  test_equivalence_and_error_bound(signal);
  test_boundary_continuity(signal);
  test_config_validation();
  test_memory_bounded(signal);

  free(signal);
  printf("\n%s: %d failure(s)\n",
         g_failures ? "FAILED" : "ALL TESTS PASSED", g_failures);
  return g_failures ? 1 : 0;
}
