# 制造根验证基准 · Manufactured-Root Verification Benchmark

> 论文《气动翼型非线性叶素方程簇的高阶迭代快速求解及应用》(投稿系统仿真学报)**补充实验一**的完整代码、数据处理与结果。
> A manufactured-solution verification benchmark for a differentiable, high-order iterative solver of scalar transcendental equations (fp64 / double-single df32 / fp32 on NVIDIA GPU).

把 PDE 的"制造解(manufactured solution)"方法搬到标量超越方程上:构造已知精确根 x\* 与局部导数 D 的残差 `f(x)=g(x)+s₁x+s₀`,从而在**不扰动高阶结构**的前提下独立扫描条件数 κ,对高阶迭代求解器做逐条判据核验。

基础族 `g(x)=x·e^{ax}+b·ln(1+cx)+d·sin(ωx)`(指数=Lambert 型代数×超越耦合;对数=可控奇异;正弦=拉高高阶导数)。

---

## 方法四要素

1. **制造根**:给定 x\*、D,由 `f(x*)=0` 定 s₀、`f'(x*)=D` 定 s₁;k≥2 阶 `f^{(k)}=g^{(k)}`,故 κ 可独立扫描。
2. **高阶迭代**:Newton(2 阶)/ Halley(3 阶)/ Householder-3(4 阶),直接有理式步(f 在分子,收敛后免除零)。
3. **Smale α 认证**:`α=β·γ`,阈值 `(13−3√17)/4 ≈ 0.157671`;认证点 ⇒ approximate zero ⇒ 固定步数 `k_max=⌈log₂(2+log₂(β/ε))⌉` 无分支迭代。
4. **混合精度**:fp64(u=2⁻⁵³)/ df32 双单精度(double-single,u≈2⁻⁴⁶)/ fp32(u=2⁻²⁴);可达相对误差下限 ≈ κ·u。

---

## 关键结果(全部机器实测,纯 C++/CUDA 手写)

- **收敛阶**(__float128 双偏移单步):Newton **2.000** / Halley **2.992** / HH-3 **3.885** = 形式阶。
- **可达精度下限**(floorscaled 初值,无构造偏置):三精度点云各贴 `κ·u` 线、按精度分层;C2(误差≤4κu)**95.57%**;e_repr 认证子集 G3 **1.11e-13** / G4 5.3e-12 / G5 1.4e-6。见 `v2_cpp/figs_v2/fig2_floor.*`。
- **★基本极限(独立小结论)**:κ > 1/√u ≈ 1.3e8(fp64)时,根的条件地板 κu 超过牛顿收敛盆地 ~1/κ,**任何有界初值都无法通过迭代到达地板**。可迭代展示地板的 κ 上界直接由精度 u 决定。
- **认证价值 = 步数分布**(非收敛率):认证点 Newton 中位 **2** 步(点质量,自适应停机 100% 满足 k_max 上界),未认证长尾;κ 无关初值下认证收敛 / 未认证不收敛的干净分离。
- **可微性三路互证**(纯 C++,无框架):前向对偶数 AD ↔ 隐函数定理闭式 **p50 2.0e-15 且随 κ·u**、零符号翻转;中心差分穿求解器在 κ<1e4 到差分地板。
- **阶次×精度全因子 M1–M9**:吞吐 fp64 526/460/397、df32 424/406/374、fp32 42862/35511/28960 M根/s(阶次↑→吞吐↓;fp32 碾压;fp32 用快速内建 __expf 与 G5 一致)。精度按 u 分层;p99 由 8 步预算未收敛支配(非精度)。见 `v2_cpp/figs_v2/fig9_matrix.*`。
- **硬件(峰值微基准)**:两张 Blackwell 卡 fp64 均被重度削减——B300 `R=fp32/fp64=71.4`(fp64 993 GFLOP/s)、RTX5090 `R=40.6`(1578);df32 纯 FMA 反超原生 fp64(3.8× / 2.0×)。df32 在求解器里更慢是**超越函数指令暴涨**(SASS:df32 核 1267 FFMA vs fp64 104 DFMA),非 fp64 快慢。
- **能耗**(修正协议,扣空载,饱和吞吐;照 `logs/energy_v2.log`):G3 fp64 0.2592 / G4 df32 0.6678 / G5 fp32 0.0200 J/百万根(P_idle 134.0W)。等精度对照(照 `logs/eqacc_saturated.csv`)G3 393.4 / G2 395.5 M根/s、精度同(p50 1.96e-11,p99 2.72e-3 为 8 步预算截断非精度损失)。
- **CPU 对照**:同算法 GPU 满卡 vs CPU 单核 ~109× / vs 128 线程 ~1.9×(纯 C++,非解释器口径)。

---

## ★方法学诚实性:五轮独立复核逐个纠错

本基准经过多轮从零独立复核(逐文件 diff + CSV 重算 + 独立交叉验证),每轮都定位并修复真问题——这是本工作最有分量的部分:

1. **CPU 基线口径**:旧 3917× 量的是 Python 解释器开销;纯 C++ 同算法真值 ~109×(单核)。
2. **认证 β=0 灾难性相消**:double 下 `β=|f(x0)/f'(x0)|` 在近根处相消恰好舍入成 0 → 假认证极病态实例(2937 个);修法=认证升 long double(`mrv.hpp` 补 long double 重载否则 `derivs<long double>` 静默退化)。
3. **基准构造偏置**:旧初值 `off∝1/γ` 使高 κ 实例起跑即在地板处、根本不迭代,撑高 C2、造 fig2 假下降支;修法=κ 无偏初值(floorscaled / kindep 双数据集)。
4. **停机容差**:固定 ε=1e-11 对 u·κ≥ε 的半数实例落 fp64 地板下停不下(假违反 k_max);修法=每实例自适应 ε=max(1e-11, C·u·κ) → 步数许可 100% 无条件成立。
5. **FMA 收缩伪影**:多根扫描 38.6% 是 `-march=native` 的 FMA 收缩产物;long double 网格 + 关 FMA 收缩后真值 ≈ **77%**(C4 括号构造的多根压力约 2 倍)。
6. **warp-waste / 汇编级**:旧"认证消发散 0.000"是硬编码字面量;SASS 反汇编证认证只消 1 条早退分支,超越函数内在发散消不掉;固定步靠**构造**归零。

权威结论表见 `v2_cpp/criteria_v2.md`;逐轮修正全记录见 `v2_cpp/纠正与重建计划.md`。仓库根 `results/criteria.txt` 与 `REPORT_*.md` 为早期版本、已打"已被取代"横幅仅存档。

---

## 目录结构

```
src/            制造根生成/参考解/CPU-GPU 求解器/分析(gen_instances/reference_solver/
                export_bin/solver*.cu/df32.cuh/analyze/order_mp/finalize …)
v2_cpp/         纠正后重建(权威):
  include/      mrv.hpp(模板核) dual.hpp(前向AD) alpha.hpp(α认证, 模板化)
  *.cpp/*.cu    canonical/build_canonical(唯一真源) cpu_baseline order_mp diff3
                stepdist multiroot check_alpha peak_flops solver_matrix(M1-M9) …
  figs_v2/      九张出版级图(各 PDF+PNG)
  instances*.csv / matrix_floor.csv   唯一真源派生数据
  criteria_v2.md            权威判据表
  纠正与重建计划.md          调研结论 + 五轮修正全记录
results/ logs/  判据/吞吐/能耗原始输出
data/           仅 manifest(原始 parquet/bin 大文件未入库, 可 seed 重生成)
```

---

## 复现

```bash
# 1. 生成制造根实例(mpmath 50 位; --offset_mode floorscaled 用于地板/C2, kindep 用于认证)
python src/gen_instances.py --comp B --N 16384 --seed 20260726 --offset_mode floorscaled --M 100 --out data/B_floor.parquet
python src/reference_solver.py data/B_floor.parquet --procs 128     # 高精度参考根 x_mach_star
python src/export_bin.py data/B_floor.parquet data/B_floor.bin
# 2. CUDA 求解 + 计时(sm_103 示例; 换 -arch 对应你的卡)
nvcc -O3 -arch=sm_103 src/solver_timed.cu -o solver_timed && ./solver_timed data/B_floor.bin G3 8 xhat.bin
# 3. 唯一真源 + 图(build_canonical 同时写 kappa.bin,multiroot 依赖它、缺则硬报错不静默回退)
g++ -O3 -march=native v2_cpp/canonical.cpp -o canonical -Iv2_cpp/include
python v2_cpp/build_canonical.py <parquet> <xhat前缀> <输出csv> <alpha_data>   # 见 rerun_cert.sh/rerun_floor.sh
python v2_cpp/figures.py
# 多根扫描(必须先有 kappa.bin): g++ -O3 -march=native -ffp-contract=off v2_cpp/multiroot.cpp -o multiroot -Iv2_cpp/include
```

> 两个数据集:`instances_floor.csv`(floorscaled,fig2/C2/M1-M9)与 `instances_cert.csv`(kindep,fig4 认证价值);fig5 warp 用 floorscaled。∝1/γ 旧构造已从 gen 移除,legacy 数据不再产生。

**硬件**:8× NVIDIA B300 SXM6(sm_103,CUDA 13.2)+ 对照卡 RTX 5090(sm_120)。CPU 基线 g++ -O3 -march=native + OpenMP。

## 许可

Apache-2.0(见 LICENSE)。若本代码对你的研究有帮助,欢迎引用上述论文。
