#!/bin/bash
# Build the imsrg++ executable in a podman image from a COPY of the sibling checkout, leaving
# ../../../imsrg untouched. Needs about 10 minutes and ~2 GB. Output: $WORK/imsrg/build/imsrg++.
# gcc 13 needs -include cstdint (missing include in Pwd.hh) and libIMSRG.so needs -fPIC, which the
# project only enables together with the Python bindings.
set -euo pipefail
SRC=${IMSRG_SRC:-$(cd "$(dirname "$0")/../../.." && pwd)/imsrg}   # checkout with src/
WORK=${WORK:-/tmp/nuham-imsrg}
mkdir -p "$WORK/imsrg/build" "$WORK/imsrg/data" "$WORK/imsrg/run/output"
rm -rf "$WORK/imsrg/src" && cp -r "$SRC/src" "$WORK/imsrg/src"
podman build -t imsrg-dev -f "$(dirname "$0")/Containerfile" "$(dirname "$0")"
podman run --rm -v "$WORK/imsrg:/work:Z" imsrg-dev bash -c \
  "cd build && cmake -DIMSRG_BUILD_PYTHON_BINDINGS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
   -DCMAKE_CXX_FLAGS='-include cstdint' ../src > cmake.log 2>&1 && make -j\$(nproc) imsrg++ > make.log 2>&1"
ls -la "$WORK/imsrg/build/imsrg++"
