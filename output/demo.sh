#!/usr/bin/env bash
# Non-interactive, segmented demo for the bounded-memory streaming STFT.
# Each segment prints a banner title, then pauses so the title stays readable.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build"
OUT="$ROOT/output"

banner() {
  local title="$1"
  printf '\n\033[1;36m============================================================\n'
  printf '  %s\n' "$title"
  printf '============================================================\033[0m\n'
  sleep 2
}

banner "SECTION 1/5  CODE: bounded-memory streaming STFT (pfstream)"
sed -n '40,75p' "$ROOT/include/pffft/pfstream.h"
sleep 2

banner "SECTION 2/5  FOUR-BLOCK MEMORY BUDGET (no 5th buffer)"
grep -n "PFSTREAM_NB_BLOCKS\|float \*ring\|float \*frame\|float \*spec\|float \*work" "$ROOT/src/pfstream.c"
sleep 2

banner "SECTION 3/5  BUILD via repository entry point (cmake)"
cmake --build "$BUILD" --target test_pfstream --parallel
sleep 2

banner "SECTION 4/5  ACCEPTANCE TEST: equivalence, continuity, memory"
"$BUILD/tests/test_pfstream"
sleep 2

banner "SECTION 5/5  CTEST: full suite, no regressions"
( cd "$BUILD" && ctest -R "test_pfstream|test_pffft_float|test_fft_factors" --output-on-failure )
printf '\n\033[1;32mDemo finished. Artifacts in %s\033[0m\n' "$OUT"
sleep 2
