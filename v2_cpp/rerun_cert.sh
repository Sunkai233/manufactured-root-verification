#!/usr/bin/env bash
# kindep 认证数据集(B_cert, 用于 fig4/5/C3/认证价值) + 修好 fp32 的 M1-M9 重跑。
export PATH=/usr/local/cuda/bin:$PATH
cd ~/exp1_mrv || exit 1
echo "=== gen kindep (B_cert, 同 seed) ==="
./venv/bin/python src/gen_instances.py --comp B --N 16384 --seed 20260726 --offset_mode kindep --out data/B_cert.parquet
./venv/bin/python src/reference_solver.py data/B_cert.parquet --procs 128 >/dev/null 2>&1 && echo ref_done
./venv/bin/python src/export_bin.py data/B_cert.parquet data/B_cert.bin
nvcc -O3 -arch=sm_103 src/solver_timed.cu -o src/solver_timed 2>/dev/null
for m in G3 G4 G5; do src/solver_timed data/B_cert.bin $m 8 results/xhat_cert_$m.bin >/dev/null 2>&1; done
cd v2_cpp
g++ -O3 -march=native canonical.cpp -o canonical -Iinclude 2>/dev/null
./canonical ../data/B_cert.bin alpha_data_cert.csv
../venv/bin/python build_canonical.py ../data/B_cert.parquet cert_ $PWD/instances_cert.csv $PWD/alpha_data_cert.csv 2>/dev/null
echo "=== M1-M9 重跑(fp32 快速内建, 与 G5 一致)在 floor 数据 ==="
nvcc -O3 -arch=sm_103 solver_matrix.cu -o solver_matrix -I../src 2>/dev/null
./solver_matrix ../data/B_floor.bin 8 256 | tee matrix_floor.csv
echo DONE_CERT_MATRIX
