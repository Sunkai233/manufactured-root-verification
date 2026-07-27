#!/usr/bin/env bash
# 地板数据集(floorscaled δ=M·κu, M=100): 每实例起于地板上方 100 倍 → 真迭代~3步到地板。用于 fig2/C2/M1-M9。
export PATH=/usr/local/cuda/bin:$PATH
cd ~/exp1_mrv || exit 1
echo "=== gen floorscaled M=100 (同 seed) ==="
./venv/bin/python src/gen_instances.py --comp B --N 16384 --seed 20260726 --offset_mode floorscaled --M 100 --out data/B_floor.parquet
echo "=== reference_solver ==="
./venv/bin/python src/reference_solver.py data/B_floor.parquet --procs 128 >/dev/null 2>&1 && echo ref_done
echo "=== export bin ==="
./venv/bin/python src/export_bin.py data/B_floor.parquet data/B_floor.bin
echo "=== xhat (floor_) ==="
nvcc -O3 -arch=sm_103 src/solver_timed.cu -o src/solver_timed 2>/dev/null
for m in G3 G4 G5; do src/solver_timed data/B_floor.bin $m 8 results/xhat_floor_$m.bin >/dev/null 2>&1; done
echo "=== canonical + build (floor) ==="
cd v2_cpp
g++ -O3 -march=native canonical.cpp -o canonical -Iinclude 2>/dev/null
./canonical ../data/B_floor.bin alpha_data_floor.csv
../venv/bin/python build_canonical.py ../data/B_floor.parquet floor_ $PWD/instances_floor.csv $PWD/alpha_data_floor.csv 2>/dev/null
echo "=== M1-M9 在 floor 数据(全收敛→精度有意义) ==="
nvcc -O3 -arch=sm_103 solver_matrix.cu -o solver_matrix -I../src 2>/dev/null
./solver_matrix ../data/B_floor.bin 8 256 | tee matrix_floor.csv
echo DONE_FLOOR
