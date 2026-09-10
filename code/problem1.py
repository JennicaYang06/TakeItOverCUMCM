# -*- coding: utf-8 -*-
"""
2026 高教社杯 CUMCM  C 题  问题 1
================================
微网当天的「计划购电策略」——单日、信息完全已知的经济调度 / 储能套利问题。

一句话：已知一整天每 10 分钟的电价、小区负载、光伏发电（预测且视为准确），
在 0:00 一次性决定当天每格买多少电、储能怎么充放，使全天购电费最小，
且任何时刻供电不低于负载，储能 0:00 与 24:00 电量相同。

这是一个标准线性规划（LP），下面用 cvxpy 建模求解。

--------------------------------------------------------------------------
决策变量（t = 0 .. 143，每格 10 分钟）
    g[t]   计划购电量 (kWh)            >= 0
    c[t]   储能充电量 (kWh, 母线侧)     >= 0
    d[t]   储能放电量 (kWh, 母线侧)     >= 0
    soc[t] 储能储电量 (kWh)            t = 0 .. 144

目标
    min  Σ_t price[t] * g[t]                     （全天购电费，元）

约束
    (1) 电力平衡：  g[t] + PV[t] + d[t] - c[t] >= Load[t]
    (2) 储能递推：  soc[t+1] = soc[t] + ETA_C*c[t] - d[t]/ETA_D
    (3) 电量上下限：1200 <= soc[t] <= 10800
    (4) 功率上限：  0 <= c[t], d[t] <= 5000 kW * (1/6) h = 833.333 kWh
    (5) 首尾相同：  soc[0] = soc[144] = 6000
--------------------------------------------------------------------------
假设（务必写进论文「模型假设」一节）
  A1 充放电总效率 90% 在两个方向上平均分配：ETA_C = ETA_D = sqrt(0.9) ≈ 0.9487。
     （可改 SPLIT_MODE 做灵敏度分析，比如把损耗全放在充电侧。）
  A2 功率上限约束在「母线侧」的充/放电量 c[t], d[t] 上。
  A3 光伏、负载在每格内按恒功率处理，电量 = 功率 * (1/6) 小时。
  A4 多余光伏可自由弃光（平衡约束取 ">=" 而非 "="），弃光不计成本。
  A5 问题 1 不设「计划购电额度」上限（该限制留到问题 2 才需要）。

运行
  python problem1.py
输出（写到项目根目录 results/ 下）
  result1.xlsx          —— 在官方模板 附件5/result1.xlsx 上原地填数
  problem1_summary.txt  —— 论文表 1、表 2 的数值
  problem1_plot.png     —— 可选，若装了 matplotlib
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import cvxpy as cp

# ============================ 路径 ============================
BASE = Path(__file__).resolve().parents[1]
ATT1 = BASE / "题目和附件" / "C题" / "附件" / "附件1.xlsx"
TEMPLATE = BASE / "题目和附件" / "C题" / "附件" / "附件5" / "result1.xlsx"
OUTDIR = BASE / "results"
OUTDIR.mkdir(exist_ok=True)
OUT_XLSX = OUTDIR / "result1.xlsx"
OUT_TXT = OUTDIR / "problem1_summary.txt"
OUT_PNG = OUTDIR / "problem1_plot.png"

# ===================== 公共参数（附录 1）=====================
DT = 1.0 / 6.0                    # 每格时长（小时）
SOC_MIN, SOC_MAX = 1200.0, 10800.0
CAP_MAX = 12000.0                 # 最大容量（此处 SOC_MAX < CAP_MAX，实际以 10800 为准）
SOC0 = 6000.0                     # 2025-1-1 0:00 初始电量
P_MAX = 5000.0                    # 最大充放电功率 kW
E_STEP_MAX = P_MAX * DT           # 每格最大充/放电量 kWh ≈ 833.333
RT_EFF = 0.90                     # 充放电总效率（往返）

SPLIT_MODE = "sqrt"              # "sqrt": 两侧各 sqrt(0.9)；"charge": 损耗全在充电侧
if SPLIT_MODE == "sqrt":
    ETA_C = np.sqrt(RT_EFF)
    ETA_D = np.sqrt(RT_EFF)
elif SPLIT_MODE == "charge":
    ETA_C = RT_EFF
    ETA_D = 1.0
else:
    raise ValueError(SPLIT_MODE)

EPS_REG = 1e-6                    # 极小正则项，仅用于打破「同时充放电」数值并列

# ======================= 读取附件 1 =======================
df = pd.read_excel(ATT1, engine="openpyxl")
# 列顺序：时间 | 电价(元/kWh) | 小区负载(kW) | 光伏发电预测功率(kW)
price = df.iloc[:, 1].to_numpy(dtype=float)
load_kw = df.iloc[:, 2].to_numpy(dtype=float)
pv_kw = df.iloc[:, 3].to_numpy(dtype=float)
T = len(price)
assert T == 144, f"附件1 应为 144 行数据，实际读到 {T} 行"

load_e = load_kw * DT            # 每格负载电量 kWh
pv_e = pv_kw * DT               # 每格光伏电量 kWh

# ======================= 建立 LP 模型 =======================
g = cp.Variable(T, nonneg=True, name="plan_buy")
c = cp.Variable(T, nonneg=True, name="charge")
d = cp.Variable(T, nonneg=True, name="discharge")
soc = cp.Variable(T + 1, name="soc")

cons = [
    soc[0] == SOC0,
    soc[T] == SOC0,                                   # (5) 首尾相同
    soc >= SOC_MIN,
    soc <= SOC_MAX,                                   # (3) 电量上下限
    soc[1:] == soc[:-1] + ETA_C * c - d / ETA_D,      # (2) 储能递推
    c <= E_STEP_MAX,
    d <= E_STEP_MAX,                                   # (4) 功率上限
    g + pv_e + d - c >= load_e,                        # (1) 电力平衡
]

cost = price @ g                                       # 全天购电费（元）
prob = cp.Problem(cp.Minimize(cost + EPS_REG * cp.sum(c + d)), cons)

solver = next((s for s in ["HIGHS", "CLARABEL", "ECOS", "SCS"]
               if s in cp.installed_solvers()), None)
prob.solve(solver=getattr(cp, solver)) if solver else prob.solve()
print(f"求解状态: {prob.status} | 求解器: {solver or 'default'}")
if prob.status not in ("optimal", "optimal_inaccurate"):
    sys.exit("模型未求得最优解，请检查数据与约束。")

# ======================= 取解 + 校验 =======================
gv = np.clip(np.asarray(g.value).ravel(), 0.0, None)
cv = np.clip(np.asarray(c.value).ravel(), 0.0, None)
dv = np.clip(np.asarray(d.value).ravel(), 0.0, None)
socv = np.asarray(soc.value).ravel()

supply = gv + pv_e + dv - cv
viol = float(np.min(supply - load_e))
both = int(np.sum((cv > 1e-6) & (dv > 1e-6)))
print(f"最小(供电-负载) = {viol:.6e}  (>=0 视为满足)")
print(f"soc 范围 = [{socv.min():.3f}, {socv.max():.3f}]  应 ⊆ [1200, 10800]")
print(f"同时充放电的时段数 = {both}  (期望 0)")
print(f"0:00 储电量 = {socv[0]:.3f} | 24:00 储电量 = {socv[-1]:.3f}")

daily_buy = float(gv.sum())
daily_cost = float(price @ gv)
print(f"\n全天购电量 = {daily_buy:.3f} kWh")
print(f"全天购电费 = {daily_cost:.3f} 元")

# ======================= 论文表 1 / 表 2 =======================
# 表 1：指定单个 10 分钟时段的购电量。t = 起始分钟 / 10
TBL1 = [("10:00-10:10", 60), ("12:00-12:10", 72), ("14:00-14:10", 84),
        ("16:00-16:10", 96), ("18:00-18:10", 108), ("20:00-20:10", 120)]
# 表 2：6 个 4 小时时段的充/放电量合计
TBL2 = [("0:00-4:00", 0, 24), ("4:00-8:00", 24, 48), ("8:00-12:00", 48, 72),
        ("12:00-16:00", 72, 96), ("16:00-20:00", 96, 120), ("20:00-24:00", 120, 144)]

lines = []
lines.append("=" * 60)
lines.append("问题 1  结果汇总")
lines.append("=" * 60)
lines.append(f"求解器: {solver or 'default'} | 效率分配: {SPLIT_MODE} "
             f"(ETA_C={ETA_C:.4f}, ETA_D={ETA_D:.4f})")
lines.append("")
lines.append("表 1  微网在指定时间段的购电量 (kWh)")
lines.append("-" * 60)
for name, t in TBL1:
    lines.append(f"  {name:<14s} {gv[t]:12.4f}")
lines.append("-" * 60)
lines.append(f"  {'全天购电量 (kWh)':<14s} {daily_buy:12.4f}")
lines.append(f"  {'全天购电费 (元)':<14s} {daily_cost:12.4f}")
lines.append("")
lines.append("表 2  储能设备在指定时间段的充放电量 (kWh)")
lines.append("-" * 60)
lines.append(f"  {'时间段':<14s} {'充电量':>12s} {'放电量':>12s}")
for name, a, b in TBL2:
    lines.append(f"  {name:<14s} {cv[a:b].sum():12.4f} {dv[a:b].sum():12.4f}")
lines.append("-" * 60)
lines.append(f"  0:00 储电量 (kWh)  = {socv[0]:.4f}")
lines.append(f"  24:00 储电量 (kWh) = {socv[-1]:.4f}")
lines.append("")
lines.append("校验：")
lines.append(f"  min(供电 - 负载) = {viol:.3e}  (>=0 合格)")
lines.append(f"  soc ∈ [{socv.min():.2f}, {socv.max():.2f}]")
lines.append(f"  同时充放电时段数 = {both}")
report = "\n".join(lines)
print("\n" + report)
OUT_TXT.write_text(report, encoding="utf-8")
print(f"\n已写出: {OUT_TXT}")

# ======================= 写 result1.xlsx（原地填模板）=======================
# 注意：官方模板「计划购电量」工作表 A 列的时间标签从 "0:10-0:20" 起、
#      末行为 "0:00+1-0:10+1"，疑似整体错位一格。这里按「行序 = 第 t 个 10 分钟
#      时段」填 B 列（第 2 行 = 0:00-0:10，...，第 145 行 = 23:50-24:00），
#      不改动模板 A 列。若评阅要求按标签对齐，另行整体平移即可。
try:
    from openpyxl import load_workbook
    wb = load_workbook(TEMPLATE)
    ws1 = wb["计划购电量"]
    for t in range(T):
        ws1.cell(row=2 + t, column=2, value=round(float(gv[t]), 6))
    ws2 = wb["充放电量"]
    for i, (_, a, b) in enumerate(TBL2):
        ws2.cell(row=2 + i, column=2, value=round(float(cv[a:b].sum()), 6))
        ws2.cell(row=2 + i, column=3, value=round(float(dv[a:b].sum()), 6))
    ws2.cell(row=2, column=5, value=round(float(socv[0]), 6))    # E2 = 0:00 储电量
    ws2.cell(row=3, column=5, value=round(float(socv[-1]), 6))   # E3 = 24:00 储电量
    wb.save(OUT_XLSX)
    print(f"已写出: {OUT_XLSX}")
except Exception as e:  # noqa: BLE001
    print(f"[警告] 写 xlsx 失败：{e}")

# ======================= 可选作图 =======================
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:  # noqa: BLE001
        pass

    h = np.arange(T) * DT
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    ax[0].plot(h, price, color="tab:red")
    ax[0].set_ylabel("电价 (元/kWh)")
    ax[0].set_title("问题1：电价 / 功率平衡 / 储能电量")
    ax[1].plot(h, load_kw, label="负载", color="k")
    ax[1].plot(h, pv_kw, label="光伏", color="tab:orange")
    ax[1].plot(h, gv / DT, label="购电功率", color="tab:blue")
    ax[1].plot(h, (dv - cv) / DT, label="储能净放电功率", color="tab:green")
    ax[1].legend(ncol=4, fontsize=8)
    ax[1].set_ylabel("功率 (kW)")
    ax[2].plot(np.arange(T + 1) * DT, socv, color="tab:purple")
    ax[2].axhline(SOC_MIN, ls="--", c="gray")
    ax[2].axhline(SOC_MAX, ls="--", c="gray")
    ax[2].set_ylabel("储电量 (kWh)")
    ax[2].set_xlabel("时刻 (h)")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"已写出: {OUT_PNG}")
except Exception as e:  # noqa: BLE001
    print(f"[提示] 未作图（{e}）")
