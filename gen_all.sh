set -e
cd ~/exp1_mrv
PY=venv/bin/python
for comp in A B; do
  echo "=== 生成 $comp N=16384 ==="
  $PY src/gen_instances.py --comp $comp --N 16384 --seed 20260726 --out data/${comp}_N16384.parquet
  echo "=== R0 参考 $comp ==="
  $PY src/reference_solver.py data/${comp}_N16384.parquet --procs 128
  echo "=== 导出二进制 $comp ==="
  $PY src/export_bin.py data/${comp}_N16384.parquet data/${comp}_N16384.bin
done
echo "=== done ==="; ls -la data/
