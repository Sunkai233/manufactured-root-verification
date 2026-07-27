#!/usr/bin/env bash
export PATH=/usr/local/cuda/bin:$PATH
cd ~/exp1_mrv || exit 1
nvcc -O3 -arch=sm_103 src/solver_timed.cu -o src/solver_timed 2>/tmp/nv.err || { echo BUILD_FAIL; tail -5 /tmp/nv.err; exit 1; }
echo 'mode,N,ns,launch,tk_ms,tk_iqr,te_ms,te_iqr,Mroot/s,det,relerr_p50,relerr_p99'
for m in G3 G2 G4 G5; do src/solver_timed data/B_N16384.bin $m 8; done
