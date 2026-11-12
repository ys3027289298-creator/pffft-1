/*
  Copyright (c) 2026
*/

#include "pfstream.h"

#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* number of block_len-float buffers owned by a stream */
#define PFSTREAM_NB_BLOCKS 4

struct PFFFT_Stream {
  PFFFT_Setup *setup;
  float *ring;   /* block 1: pending input samples (linear, fill entries) */
  float *frame;  /* block 2: assembled windowed frame, FFT in/out in place */
  float *spec;   /* block 3: ordered spectrum waiting to be pulled */
  float *work;   /* block 4: FFT scratch */
  int block_len;
  int hop;
  int fill;            /* valid samples in ring, 0 .. block_len-1 */
  int out_pending;     /* nonzero while spec holds an unpulled spectrum */
  long long frames_done;
  long long spec_index;
};

PFFFT_Stream *pffft_stream_new(int block_len, int hop)
{
  PFFFT_Stream *st;
  size_t block_bytes;

  if (block_len < pffft_min_fft_size(PFFFT_REAL) ||
      !pffft_is_valid_size(block_len, PFFFT_REAL) ||
      hop < 1 || hop > block_len) {
    return NULL;
  }

  st = (PFFFT_Stream *)malloc(sizeof(PFFFT_Stream));
  if (!st)
    return NULL;

  block_bytes = (size_t)block_len * sizeof(float);
  st->ring = (float *)pffft_aligned_malloc(block_bytes);
  st->frame = (float *)pffft_aligned_malloc(block_bytes);
  st->spec = (float *)pffft_aligned_malloc(block_bytes);
  st->work = (float *)pffft_aligned_malloc(block_bytes);
  if (!st->ring || !st->frame || !st->spec || !st->work) {
    pffft_aligned_free(st->ring);
    pffft_aligned_free(st->frame);
    pffft_aligned_free(st->spec);
    pffft_aligned_free(st->work);
    free(st);
    return NULL;
  }

  st->setup = pffft_new_setup(block_len, PFFFT_REAL);
  if (!st->setup) {
    pffft_aligned_free(st->ring);
    pffft_aligned_free(st->frame);
    pffft_aligned_free(st->spec);
    pffft_aligned_free(st->work);
    free(st);
    return NULL;
  }

  st->block_len = block_len;
  st->hop = hop;
  st->fill = 0;
  st->out_pending = 0;
  st->frames_done = 0;
  st->spec_index = 0;
  return st;
}

void pffft_stream_destroy(PFFFT_Stream *st)
{
  if (!st)
    return;
  pffft_destroy_setup(st->setup);
  pffft_aligned_free(st->ring);
  pffft_aligned_free(st->frame);
  pffft_aligned_free(st->spec);
  pffft_aligned_free(st->work);
  free(st);
}

/* assemble one windowed frame out of the full ring and transform it */
static void stream_compute_frame(PFFFT_Stream *st)
{
  const int N = st->block_len;
  const float two_pi_over_N = (float)(2.0 * M_PI / (double)N);
  int i;

  /* periodic Hann window, fused with the copy so no extra block is needed */
  for (i = 0; i < N; ++i) {
    float w = 0.5f * (1.0f - cosf(two_pi_over_N * (float)i));
    st->frame[i] = st->ring[i] * w;
  }

  /* input == output alias is supported by pffft_transform_ordered */
  pffft_transform_ordered(st->setup, st->frame, st->frame, st->work,
                          PFFFT_FORWARD);
  memcpy(st->spec, st->frame, (size_t)N * sizeof(float));
  st->spec_index = st->frames_done;
  st->frames_done += 1;
  st->out_pending = 1;

  /* slide the ring: drop the consumed hop, keep the overlap */
  memmove(st->ring, st->ring + st->hop,
          (size_t)(st->fill - st->hop) * sizeof(float));
  st->fill -= st->hop;
}

int pffft_stream_push(PFFFT_Stream *st, const float *samples, int count,
                      pffft_stream_status_t *status)
{
  int consumed = 0;
  pffft_stream_status_t last = PFSTREAM_NEED_MORE;

  if (!st || !samples || count < 0) {
    if (status)
      *status = PFSTREAM_NEED_MORE;
    return 0;
  }

  while (consumed < count) {
    int space, take;

    if (st->out_pending) {
      last = PFSTREAM_OUTPUT_FULL;
      break;
    }

    space = st->block_len - st->fill;
    take = count - consumed;
    if (take > space)
      take = space;

    memcpy(st->ring + st->fill, samples + consumed,
           (size_t)take * sizeof(float));
    st->fill += take;
    consumed += take;

    if (st->fill == st->block_len) {
      stream_compute_frame(st);
      last = PFSTREAM_OK;
    }
  }

  if (status)
    *status = last;
  return consumed;
}

pffft_stream_status_t pffft_stream_pull(PFFFT_Stream *st, float *out,
                                        long long *frame_index)
{
  if (!st || !out)
    return PFSTREAM_NO_OUTPUT;
  if (!st->out_pending)
    return PFSTREAM_NO_OUTPUT;

  memcpy(out, st->spec, (size_t)st->block_len * sizeof(float));
  st->out_pending = 0;
  if (frame_index)
    *frame_index = st->spec_index;
  return PFSTREAM_OK;
}

int pffft_stream_pending(const PFFFT_Stream *st)
{
  return st ? st->out_pending : 0;
}

int pffft_stream_buffered(const PFFFT_Stream *st)
{
  return st ? st->fill : 0;
}

long long pffft_stream_frames_done(const PFFFT_Stream *st)
{
  return st ? st->frames_done : 0;
}

int pffft_stream_block_len(const PFFFT_Stream *st)
{
  return st ? st->block_len : 0;
}

int pffft_stream_hop(const PFFFT_Stream *st)
{
  return st ? st->hop : 0;
}

size_t pffft_stream_buffer_bytes(const PFFFT_Stream *st)
{
  return st ? (size_t)PFSTREAM_NB_BLOCKS * st->block_len * sizeof(float) : 0;
}

void pffft_stream_reset(PFFFT_Stream *st)
{
  if (!st)
    return;
  st->fill = 0;
  st->out_pending = 0;
  st->frames_done = 0;
  st->spec_index = 0;
}
