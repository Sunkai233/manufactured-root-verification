#!/usr/bin/env bash
# 落盘头条数字的原始输出(issue9): 饱和 eqacc + 修正能耗 → logs/ 与 results/。
export PATH=/usr/local/cuda/bin:$PATH
cd ~/exp1_mrv || exit 1
mkdir -p logs
nvcc -O3 -arch=sm_103 src/solver_timed.cu -o src/solver_timed 2>/dev/null
{ echo 'mode,N,ns,launch,tk_ms,tk_iqr,te_ms,te_iqr,Mroot_s,det,relerr_p50,relerr_p99'
  echo '# N=16384 非饱和'
  for m in G3 G2; do src/solver_timed data/B_N16384.bin $m 8; done
  echo '# REP=256 N=4.19M 饱和'
  for m in G3 G2 G4 G5; do REP=256 src/solver_timed data/B_N16384.bin $m 8; done
} | tee logs/eqacc_saturated.csv
cp logs/eqacc_saturated.csv results/throughput_v2.csv
echo "=== 能耗 ===" | tee logs/energy_v2.log
tr -d '\r' < v2_cpp/energy_v2.sh | bash 2>&1 | tee -a logs/energy_v2.log
echo "DONE_PERSIST"
