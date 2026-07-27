#!/usr/bin/env bash
cd ~/exp1_mrv/v2_cpp || exit 1
echo "=== gcc ==="; g++ --version | head -1
echo "=== build (O3 march=native openmp) ==="
g++ -O3 -march=native -fopenmp -Iinclude cpu_baseline.cpp -o cpu_baseline 2>build.err && echo BUILD_OK || { echo BUILD_FAIL; cat build.err; exit 1; }
echo "=== 向量化报告(是否 loop vectorized) ==="
g++ -O3 -march=native -fopenmp -Iinclude -fopt-info-vec cpu_baseline.cpp -o /dev/null 2>&1 | grep -iE 'vectorized' | head -8
echo "=== run on B_N16384 ==="
./cpu_baseline ../data/B_N16384.bin 8
echo "=== x86 反汇编: solve_fixed 热点(找 vfmadd*pd/ymm/zmm=向量FMA, 内层无 jXX) ==="
objdump -d -M intel --no-show-raw-insn cpu_baseline > asm_cpu.txt
grep -cE 'vfmadd[0-9]+pd|vfmadd[0-9]+sd' asm_cpu.txt | sed 's/^/  向量+标量FMA条数: /'
echo "  打包pd(ymm/zmm 向量FMA):"; grep -oE 'vfmadd[0-9]+pd[^,]*(ymm|zmm)' asm_cpu.txt | sort | uniq -c | head
echo "  标量sd(xmm):"; grep -cE 'vfmadd[0-9]+sd' asm_cpu.txt | sed 's/^/    /'
