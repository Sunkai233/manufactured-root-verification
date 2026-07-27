#!/usr/bin/env bash
# v4 全链重跑: κ 无关初值偏移重生成 → export → 重dump xhat → canonical → build_canonical。
export PATH=/usr/local/cuda/bin:$PATH
cd ~/exp1_mrv || exit 1
echo "=== 1. 重生成 B (同 seed 20260726, κ无关偏移 δ∈[1e-3,1e-1]) ==="
./venv/bin/python src/gen_instances.py --comp B --N 16384 --seed 20260726 --out data/B_N16384.parquet
echo "=== 1b. reference_solver R0 (算 x_mach_star + 梯度基准, 写回 parquet) ==="
./venv/bin/python src/reference_solver.py data/B_N16384.parquet --procs 128
echo "=== 2. export bin ==="
./venv/bin/python src/export_bin.py data/B_N16384.parquet data/B_N16384.bin
echo "=== 3. 重 dump xhat G3/G4/G5 ==="
nvcc -O3 -arch=sm_103 src/solver_timed.cu -o src/solver_timed 2>/dev/null
for m in G3 G4 G5; do src/solver_timed data/B_N16384.bin $m 8 results/xhat_$m.bin >/dev/null 2>&1; done
echo "=== 4. canonical + build_canonical ==="
cd v2_cpp
g++ -O3 -march=native canonical.cpp -o canonical -Iinclude && ./canonical ../data/B_N16384.bin alpha_data.csv
../venv/bin/python build_canonical.py 2>/dev/null
echo "DONE_RERUN_V4"
