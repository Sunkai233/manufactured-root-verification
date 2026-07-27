#!/usr/bin/env bash
# 步骤A: 重编 solver_timed 并 dump SASS, 列出核函数(去混淆)。
export PATH=/usr/local/cuda/bin:$PATH
cd ~/exp1_mrv || exit 1
mkdir -p asm
echo "=== nvcc ==="; nvcc --version | tail -1
echo "=== recompile (O3 sm_103) ==="
if nvcc -O3 -arch=sm_103 src/solver_timed.cu -o src/solver_timed 2>asm/nvcc.err; then
  echo COMPILE_OK
else
  echo COMPILE_FAIL; tail -8 asm/nvcc.err
fi
echo "=== dump SASS ==="
cuobjdump --dump-sass src/solver_timed > asm/solver_timed.sass 2>asm/sass.err
echo "SASS lines: $(wc -l < asm/solver_timed.sass)"
echo "=== kernels (mangled -> demangled) ==="
grep 'Function :' asm/solver_timed.sass | sed 's/.*Function : //' | while read m; do
  echo "$m  ::  $(echo "$m" | c++filt)"
done
