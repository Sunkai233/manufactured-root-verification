#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇编级执行效果监控: 解析 solver_timed 的 SASS, 逐核统计指令构成,
判定"认证固定步核(k_g3)结构无线程束发散"这一主张。
输出: 每核 总指令/DFMA(fp64 FMA)/FFMA(fp32 FMA)/MUFU(超越)/DSETP+ISETP(比较)/
      谓词分支(@P..BRA, 数据相关控制流)/BSSY+BSYNC(发散重收敛屏障)/BRA(全部分支)。
判据: 循环体内数据相关分支数 = 发散潜在点。k_g3 应为 0(仅统一回边), k_g2 应 >=1。
"""
import re, json, subprocess, sys, os

SASS = os.path.expanduser("~/exp1_mrv/asm/solver_timed.sass")
OUT  = os.path.expanduser("~/exp1_mrv/asm/sass_analysis.json")

def demangle(name):
    try:
        return subprocess.run(["c++filt"], input=name, capture_output=True, text=True).stdout.strip()
    except Exception:
        return name

# 关注的核: 短名 -> 匹配子串(mangled 里含 k_g3 等)
KERNELS = {"k_g3":"k_g3", "k_g2":"k_g2", "k_g4":"k_g4", "k_g5":"k_g5", "k_g1step":"k_g1step"}

def parse():
    with open(SASS) as f:
        lines = f.readlines()
    # 切分函数块
    blocks = {}
    cur = None
    for ln in lines:
        m = re.search(r"Function\s*:\s*(\S+)", ln)
        if m:
            cur = m.group(1); blocks[cur] = []
            continue
        if cur is not None:
            blocks[cur].append(ln)
    return blocks

# 单条 SASS 指令行: /*addr*/  [@!P0] OPCODE ... ;
INSTR = re.compile(r"/\*[0-9a-fA-F]+\*/\s+(@!?P\w+\s+)?([A-Z][A-Z0-9_.]*)")

def analyze_block(blk):
    stat = dict(total=0, dfma=0, ffma=0, dadd=0, dmul=0, fadd=0, fmul=0,
                mufu=0, dsetp=0, isetp=0, fsetp=0, pred_branch=0, bra=0,
                bssy=0, bsync=0, sel=0, pred_any=0)
    for ln in blk:
        m = INSTR.search(ln)
        if not m:
            continue
        pred, op = m.group(1), m.group(2)
        base = op.split(".")[0]
        stat["total"] += 1
        if pred:
            stat["pred_any"] += 1
        if base == "DFMA": stat["dfma"] += 1
        elif base == "FFMA": stat["ffma"] += 1
        elif base == "DADD": stat["dadd"] += 1
        elif base == "DMUL": stat["dmul"] += 1
        elif base == "FADD": stat["fadd"] += 1
        elif base == "FMUL": stat["fmul"] += 1
        elif base == "MUFU": stat["mufu"] += 1
        elif base == "DSETP": stat["dsetp"] += 1
        elif base == "ISETP": stat["isetp"] += 1
        elif base == "FSETP": stat["fsetp"] += 1
        elif base == "SEL": stat["sel"] += 1
        elif base == "BSSY": stat["bssy"] += 1
        elif base == "BSYNC": stat["bsync"] += 1
        elif base == "BRA":
            stat["bra"] += 1
            if pred:  # 谓词化 BRA = 数据/条件相关控制流(可致发散)
                stat["pred_branch"] += 1
    return stat

def main():
    blocks = parse()
    # 建立 mangled 名 -> 短名
    found = {}
    for mangled in blocks:
        dem = demangle(mangled)
        for short, key in KERNELS.items():
            if key in mangled or key in dem:
                found[short] = mangled
    result = {}
    order = ["k_g3","k_g2","k_g4","k_g5","k_g1step"]
    print("%-9s %6s %5s %5s %5s %5s %6s %8s %5s %5s  %s" %
          ("kernel","instr","DFMA","FFMA","MUFU","SETP","BSSY","predBRA","BRA","SEL","判定"))
    for short in order:
        if short not in found:
            print("%-9s  (未找到)" % short); continue
        st = analyze_block(blocks[found[short]])
        setp = st["dsetp"]+st["isetp"]+st["fsetp"]
        # 发散潜在点 = 谓词分支中扣掉循环回边(回边通常1条统一分支, 非谓词或统一)
        # 更严: 用 BSSY(发散重收敛屏障)数量作发散区块数的下界证据
        diverge = st["bssy"]
        verdict = "直线FMA链·无发散区" if diverge==0 else ("含%d处发散区(数据相关分支)"%diverge)
        print("%-9s %6d %5d %5d %5d %5d %6d %8d %5d %5d  %s" %
              (short, st["total"], st["dfma"], st["ffma"], st["mufu"], setp,
               st["bssy"], st["pred_branch"], st["bra"], st["sel"], verdict))
        result[short] = dict(mangled=found[short], **st, setp=setp, diverge_regions=diverge)
    with open(OUT,"w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("\n-> 写入", OUT)
    # 核心对照结论
    if "k_g3" in result and "k_g2" in result:
        g3, g2 = result["k_g3"], result["k_g2"]
        print("\n[核心对照] 认证固定步 k_g3: 发散区 %d, 谓词分支 %d, DFMA %d 条/核" %
              (g3["diverge_regions"], g3["pred_branch"], g3["dfma"]))
        print("           带早退 k_g2: 发散区 %d, 谓词分支 %d, DFMA %d 条/核" %
              (g2["diverge_regions"], g2["pred_branch"], g2["dfma"]))

if __name__ == "__main__":
    main()
