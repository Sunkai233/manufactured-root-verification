#!/usr/bin/env bash
# 等精度对照在饱和规模(REP=256 -> N≈4.19M)重跑, 纠 issue5(原 N=16384 启动延迟受限致 G2=G3 假象)。
export PATH=/usr/local/cuda/bin:$PATH
cd ~/exp1_mrv || exit 1
nvcc -O3 -arch=sm_103 src/solver_timed.cu -o src/solver_timed 2>/dev/null || { echo BUILD_FAIL; exit 1; }
echo 'mode,N,ns,launch,tk_ms,tk_iqr,te_ms,te_iqr,Mroot/s,det,relerr_p50,relerr_p99'
echo "--- N=16384 (非饱和, 对照) ---"
for m in G3 G2; do src/solver_timed data/B_N16384.bin $m 8; done
echo "--- REP=256 N≈4.19M (饱和) ---"
for m in G3 G2; do REP=256 src/solver_timed data/B_N16384.bin $m 8; done
