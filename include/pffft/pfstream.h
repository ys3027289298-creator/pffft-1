/*
  Copyright (c) 2026

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met: redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer;
  redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
*/

/*
   PFSTREAM : bounded-memory streaming STFT on top of PFFFT.

   A real-time spectrum analyzer turns block FFTs into a continuous
   stream by sliding a window over the input with a fixed hop.
   This module keeps exactly four N-float buffers internally, so the
   memory footprint is O(N) and does not grow with the stream length.

   The producer side (pffft_stream_push()) and the consumer side
   (pffft_stream_pull()) are decoupled: push() accepts arbitrary chunk
   lengths and never blocks on the consumer, pull() returns the oldest
   computed spectrum. Back-pressure is reported with return codes
   instead of dropped samples, hence block boundaries never break.

   Restrictions:
   - 1D real transforms only, 32-bit single precision.
   - block length N must be a valid PFFFT real-transform size
     (N = 2^a * 3^b * 5^c, a >= 5).
   - hop H must satisfy 1 <= H <= N; overlap = N - H.
*/

#ifndef PFSTREAM_H
#define PFSTREAM_H

#include <stddef.h>
#include "pffft.h"

#ifndef PFSTREAM_EXPORT
#  ifdef PFSTREAM_STATIC_DEFINE
#    define PFSTREAM_EXPORT
#  elif defined(_WIN32) || defined(__CYGWIN__)
#    ifdef PFSTREAM_EXPORTS
#      define PFSTREAM_EXPORT __declspec(dllexport)
#    else
#      define PFSTREAM_EXPORT __declspec(dllimport)
#    endif
#  elif defined(PFSTREAM_EXPORTS) && (defined(__GNUC__) || defined(__clang__))
#    define PFSTREAM_EXPORT __attribute__((visibility("default")))
#  else
#    define PFSTREAM_EXPORT
#  endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* status codes returned by push/pull */
typedef enum {
  PFSTREAM_OK = 0,          /* a spectrum was produced (push) or delivered (pull) */
  PFSTREAM_NEED_MORE = 1,   /* push: fewer than N samples buffered, no spectrum yet */
  PFSTREAM_OUTPUT_FULL = 2, /* push: one spectrum is waiting, pull() first */
  PFSTREAM_NO_OUTPUT = 3    /* pull: no spectrum available yet */
} pffft_stream_status_t;

typedef struct PFFFT_Stream PFFFT_Stream;

/*
   Create a streaming STFT with block length 'block_len' and hop size
   'hop' (overlap = block_len - hop). A periodic Hann window is
   applied before every forward transform.

   Returns NULL on invalid parameters or allocation failure.
   The stream owns exactly four block_len-float buffers; its size is
   independent of the total stream length.
*/
PFSTREAM_EXPORT PFFFT_Stream *pffft_stream_new(int block_len, int hop);

/*
   Destroy the stream and release all internal buffers.
*/
PFSTREAM_EXPORT void pffft_stream_destroy(PFFFT_Stream *st);

/*
   Feed 'count' input samples. Samples are consumed one at a time:
   whenever the internal ring buffer holds block_len samples, a
   windowed FFT is computed and hop samples are removed from the ring.

   Returns the number of samples actually consumed. *status (when
   non-NULL) receives PFSTREAM_OK if at least one spectrum was
   produced, PFSTREAM_NEED_MORE if the ring is not full yet, or
   PFSTREAM_OUTPUT_FULL if a previously computed spectrum has not been
   pulled: in that case no samples are consumed and no data is lost.
*/
PFSTREAM_EXPORT int pffft_stream_push(PFFFT_Stream *st, const float *samples,
                                      int count, pffft_stream_status_t *status);

/*
   Fetch the oldest spectrum, ordered as interleaved complex numbers:
   N/2 complex bins = N floats, layout [re0, im0, re1, im1, ...].
   'out' must hold at least block_len floats and be 16-byte aligned
   (use pffft_aligned_malloc).

   Returns PFSTREAM_OK on success, PFSTREAM_NO_OUTPUT when no spectrum
   is waiting. *frame_index (when non-NULL) receives the index of the
   delivered frame, i.e. its start sample position divided by hop.
*/
PFSTREAM_EXPORT pffft_stream_status_t pffft_stream_pull(PFFFT_Stream *st,
                                                        float *out,
                                                        long long *frame_index);

/* number of spectra currently waiting to be pulled (0 or 1) */
PFSTREAM_EXPORT int pffft_stream_pending(const PFFFT_Stream *st);

/* number of samples currently held in the input ring (0 .. block_len-1) */
PFSTREAM_EXPORT int pffft_stream_buffered(const PFFFT_Stream *st);

/* total number of spectra produced since creation or last reset */
PFSTREAM_EXPORT long long pffft_stream_frames_done(const PFFFT_Stream *st);

PFSTREAM_EXPORT int pffft_stream_block_len(const PFFFT_Stream *st);
PFSTREAM_EXPORT int pffft_stream_hop(const PFFFT_Stream *st);

/*
   Total size of the four internal block_len-float buffers, i.e. the
   stream's data memory; independent of the number of samples processed.
*/
PFSTREAM_EXPORT size_t pffft_stream_buffer_bytes(const PFFFT_Stream *st);

/* drop all buffered input and pending spectra, keep the configuration */
PFSTREAM_EXPORT void pffft_stream_reset(PFFFT_Stream *st);

#ifdef __cplusplus
}
#endif

#endif /* PFSTREAM_H */
