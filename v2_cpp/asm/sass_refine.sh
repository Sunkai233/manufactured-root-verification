#!/usr/bin/env bash
# 汇编级证据(精炼版, 按方法论调研): 每核可复现 grep 计数 + 抽 cubin 出 CFG。
# 判据: 数据相关分支(谓词溯源到残差)=业务发散; 固定步核应为0, 早退核应为1。
export PATH=/usr/local/cuda/bin:$PATH
cd ~/exp1_mrv || exit 1
mkdir -p asm/cfg
BIN=src/solver_timed
echo "=== toolchain ==="; nvcc --version | grep release; echo "dot: $(which dot 2>/dev/null || echo MISSING)"

for tag in k_g3 k_g2 k_g4 k_g5; do
  MG=$(cuobjdump -sass $BIN | grep "Function : .*${tag}\b" | head -1 | sed 's/.*Function : //')
  [ -z "$MG" ] && { echo "$tag: 未找到"; continue; }
  cuobjdump -sass --function "$MG" $BIN > asm/${tag}.sass 2>/dev/null
  fma=$(grep -cE 'DFMA|FFMA' asm/${tag}.sass)
  bss=$(grep -cE 'BSSY|BSYNC' asm/${tag}.sass)
  pbr=$(grep -cE '@!?P[0-9]+ +BRA' asm/${tag}.sass)
  abr=$(grep -cE '\bBRA\b' asm/${tag}.sass)
  cmp=$(grep -cE 'ISETP|FSETP|DSETP' asm/${tag}.sass)
  mufu=$(grep -cE '\bMUFU\b' asm/${tag}.sass)
  printf "%-8s FMA=%-5s BSSY/BSYNC=%-4s predBRA=%-4s BRA=%-4s SETP=%-4s MUFU=%-3s\n" \
         "$tag" "$fma" "$bss" "$pbr" "$abr" "$cmp" "$mufu"
done

echo "=== k_g2 相对 k_g3 的 SASS 助记符差量(应=早退分支) ==="
grep -oE '/\*[0-9a-f]+\*/ +(@!?P[0-9]+ +)?[A-Z][A-Z0-9_.]*' asm/k_g3.sass | grep -oE '[A-Z][A-Z0-9_.]*$' | sort | uniq -c > /tmp/g3op.txt
grep -oE '/\*[0-9a-f]+\*/ +(@!?P[0-9]+ +)?[A-Z][A-Z0-9_.]*' asm/k_g2.sass | grep -oE '[A-Z][A-Z0-9_.]*$' | sort | uniq -c > /tmp/g2op.txt
join -a1 -a2 -e0 -o '0,1.1,2.1' -1 2 -2 2 <(awk '{print $2,$1}' /tmp/g3op.txt|sort) <(awk '{print $2,$1}' /tmp/g2op.txt|sort) 2>/dev/null \
  | awk '{d=$3-$2; if(d!=0) printf "  %+d  %s  (g3=%d g2=%d)\n",d,$1,$2,$3}'

echo "=== 抽 cubin + nvdisasm CFG(.dot) ==="
cd asm/cfg
cuobjdump -xelf all ../../$BIN >/dev/null 2>&1
CUBIN=$(ls *.cubin 2>/dev/null | head -1)
echo "cubin: ${CUBIN:-MISSING}"
if [ -n "$CUBIN" ]; then
  for tag in k_g3 k_g2; do
    MG=$(nvdisasm -c "$CUBIN" 2>/dev/null | grep -m1 "${tag}\b" | tr -d ':' | awk '{print $NF}')
    nvdisasm -cfg -fun "$MG" "$CUBIN" > ${tag}.dot 2>/dev/null && echo "  ${tag}.dot: $(wc -l < ${tag}.dot) 行"
  done
  if which dot >/dev/null 2>&1; then
    for f in *.dot; do dot -Tpng "$f" -o "${f%.dot}.png" && echo "  rendered ${f%.dot}.png"; done
  else
    echo "  (dot 缺失, 仅生成 .dot, 本地渲染)"
  fi
fi
ls -la
