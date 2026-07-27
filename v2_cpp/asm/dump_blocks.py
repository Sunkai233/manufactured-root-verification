#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""拆分各核 SASS 到独立文件; 对 k_g3 vs k_g2 做结构对照, 定位 BSSY 来源。"""
import re, os, subprocess
SASS = os.path.expanduser("~/exp1_mrv/asm/solver_timed.sass")
D = os.path.expanduser("~/exp1_mrv/asm")

def parse():
    blocks, cur = {}, None
    for ln in open(SASS):
        m = re.search(r"Function\s*:\s*(\S+)", ln)
        if m: cur = m.group(1); blocks[cur] = []; continue
        if cur is not None: blocks[cur].append(ln.rstrip("\n"))
    return blocks

SHORT = {"k_g3":"k_g3","k_g2":"k_g2","k_g4":"k_g4","k_g5":"k_g5","k_g1step":"k_g1step"}
blocks = parse()
name2short = {}
for mง in blocks:
    for s,k in SHORT.items():
        if k in mง: name2short[mง]=s

# 只取指令助记符序列(去地址/寄存器), 便于结构 diff
INSTR = re.compile(r"/\*[0-9a-fA-F]+\*/\s+(@!?P\w+\s+)?([A-Z][A-Z0-9_.]*)")
def opseq(blk):
    seq=[]
    for ln in blk:
        m=INSTR.search(ln)
        if m:
            pred = (m.group(1) or "").strip()
            seq.append((pred, m.group(2).split(".")[0]))
    return seq

for mง,s in name2short.items():
    open(os.path.join(D,s+".sass"),"w").write("\n".join(blocks[mง]))

g3=opseq(blocks[[k for k,v in name2short.items() if v=="k_g3"][0]])
g2=opseq(blocks[[k for k,v in name2short.items() if v=="k_g2"][0]])

# g2 相对 g3 的多出指令(用简单 LCS 差集近似: 统计每种(pred,op)计数差)
from collections import Counter
c3=Counter(g3); c2=Counter(g2)
print("=== k_g2 相对 k_g3 多出的指令(差量应等于早退分支相关) ===")
for k in sorted(set(c2)|set(c3)):
    d=c2[k]-c3[k]
    if d!=0: print("  %+d  %s %s"%(d, k[0] or "   ", k[1]))

# 定位 k_g3 里每个 BSSY 前 4 条指令, 判断它跟随的是超越函数还是循环控制
print("\n=== k_g3 中各 BSSY 前序上下文(判来源) ===")
blk3=blocks[[k for k,v in name2short.items() if v=="k_g3"][0]]
lines=[l for l in blk3 if INSTR.search(l)]
for i,l in enumerate(lines):
    m=INSTR.search(l)
    if m and m.group(2).split(".")[0]=="BSSY":
        ctx=[INSTR.search(x).group(2) for x in lines[max(0,i-4):i] if INSTR.search(x)]
        print("  BSSY <= [%s]"%(" ".join(ctx)))
print("\nBSSY 总数 k_g3 =", sum(1 for l in lines if (INSTR.search(l) and INSTR.search(l).group(2).split('.')[0]=='BSSY')))
