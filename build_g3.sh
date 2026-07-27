set -e
cd ~/exp1_mrv
NVCC=/usr/local/cuda/bin/nvcc
echo "=== 支持的 arch ==="; $NVCC --list-gpu-arch 2>/dev/null | tail -6
echo "=== 导出二进制 ==="
venv/bin/python src/export_bin.py data/B_N512.parquet data/B_N512.bin
echo "=== 编译 fp64 / fp32 (sm_103) ==="
$NVCC -O3 -arch=sm_103 src/solver.cu -o src/solver_fp64
$NVCC -O3 -arch=sm_103 -DUSE_FP32 src/solver.cu -o src/solver_fp32
echo "=== 运行 G3 ==="
src/solver_fp64 data/B_N512.bin 8
src/solver_fp32 data/B_N512.bin 8
