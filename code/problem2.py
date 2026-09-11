# -*- coding: utf-8 -*-
"""
2026 高教社杯 CUMCM  C 题  问题 2
================================
电价逐日相同（附件1 曲线），小区负载与光伏为全年逐日实际数据（附件2）。
在每天 0:00 制定当天「计划购电策略」；若某时段供电仍低于负载，则以 5×电价「紧急购电」。
计划区间：2025-02-01 ~ 2025-12-31，共 334 天。

建模思路（见指导讨论）
----------------------------------------------------------------------
· 与问题 1 同一套 LP，逐日独立求解 334 次（rolling / look-ahead economic dispatch）。
· 储能日间衔接方式：**日周期** —— 每天 s(0)=s(24)=6000 kWh。
  依据：电价曲线逐日相同 ⇒ 最优策略具有以日为周期的稳态特性；且与「每天 0:00
  制定当天计划」的表述一致。跨日套利在同一条日电价曲线下几乎无收益。
· 紧急购电 = 平衡约束里的高价追索变量 u_t（文献中的 unserved-energy penalty）。

两个版本（本脚本都会跑）
----------------------------------------------------------------------
2A  基础版：计划购电量 g_t 无上限。完全信息 ⇒ 紧急购电量恒为 0。
    → 写入官方模板 result2.xlsx（提交用主结果）。
2B  拓展版：假设外网正常供电容量有限，g_t ≤ Ḡ·Δt，Ḡ = ρ·(全年最大负载)。
    对 ρ ∈ {0.80,0.85,0.90,0.95,1.00} 扫一遍 ⇒ 灵敏度分析；
    以 ρ = RHO_PRIMARY_2B 生成 result2_capacity_limited.xlsx 备用。

决策变量（第 d 天，t = 0..143）
    g_t 计划购电量, u_t 紧急购电量, c_t/d_t 充/放电量(母线侧), s_t 储能电量(t=0..144)
目标（逐日）
    min  Σ π_t·g_t + Σ 5π_t·u_t
约束
    g_t + u_t + PV_t + d_t - c_t ≥ L_t          电力平衡
    s_{t+1} = s_t + ηc·c_t - d_t/ηd              储能递推 (ηc=ηd=√0.9)
    1200 ≤ s_t ≤ 10800                           电量上下限
    0 ≤ c_t,d_t ≤ 5000·(1/6) = 833.33            功率上限
    0 ≤ g_t ≤ Ḡ·(1/6)                            计划购电上限（2A: Ḡ=∞）
    u_t ≥ 0                                       紧急购电（无上限，兜底）
    s_0 = s_144 = 6000                            日周期

运行:  python problem2.py
输出(写到项目根 results/):
    result2.xlsx                       —— 2A，在官方模板上原地填数（提交用）
    result2_capacity_limited.xlsx      —— 2B (ρ=RHO_PRIMARY_2B)
    problem2_summary.txt               —— 表3 四个指定日期的表1/表2 + 全年汇总
    problem2_sensitivity_capacity.txt  —— 2B 对 Ḡ 的灵敏度表
    problem2_plot_*.png                —— 四个指定日期曲线（装了 matplotlib 才有）
"""

from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import cvxpy as cp

# ============================ 路径 ============================
BASE = Path(__file__).resolve().parents[1]
ATT1 = BASE / "题目和附件" / "C题" / "附件" / "附件1.xlsx"
ATT2 = BASE / "题目和附件" / "C题" / "附件" / "附件2.xlsx"
TEMPLATE2 = BASE / "题目和附件" / "C题" / "附件" / "附件5" / "result2.xlsx"
OUTDIR = BASE / "results"
OUTDIR.mkdir(exist_ok=True)

# ===================== 公共参数（附录 1）=====================
DT = 1.0 / 6.0
SOC_MIN, SOC_MAX = 1200.0, 10800.0
SOC0 = 6000.0
P_MAX = 5000.0
E_STEP_MAX = P_MAX * DT                 # ≈ 833.333 kWh / 10min
RT_EFF = 0.90
ETA_C = np.sqrt(RT_EFF)                 # 效率两侧平均分配；改这里做灵敏度
ETA_D = np.sqrt(RT_EFF)
EMERGENCY_MULT = 5.0                    # 紧急购电价 = 5× 交易时刻电价
EPS_REG = 1e-6

# ===================== 计划区间 =====================
DAY0 = datetime(2025, 1, 1)             # 附件2 第 1 天
PLAN_START = datetime(2025, 2, 1)
PLAN_END = datetime(2025, 12, 31)
IDX_START = (PLAN_START - DAY0).days    # 31
IDX_END = (PLAN_END - DAY0).days        # 364
ND = IDX_END - IDX_START + 1           # 334
DATES = [PLAN_START + timedelta(days=k) for k in range(ND)]

# 表3 指定日期
TBL3_DATES = [datetime(2025, 3, 20), datetime(2025, 6, 21),
              datetime(2025, 9, 23), datetime(2025, 12, 21)]

# ===================== 2B 灵敏度设置 =====================
# Ḡ = ρ · 全年最大负载。ρ 高时 12000kWh/5000kW 储能足以完全削峰 ⇒ 紧急购电为 0；
# 实测阈值在 ρ≈0.6~0.7：ρ≥0.70 紧急购电=0；ρ=0.60 仅 8 时段/2 天；ρ≤0.50 迅速上升。
RHO_LIST = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
# result2_capacity_limited.xlsx 用哪个 ρ：取 0.50 —— 外网容量降到峰值一半，
# 紧急购电占总费用约 4%（明显但不极端），作为「容量受限」的代表情形。
RHO_PRIMARY_2B = 0.50
# 论文里额外展示这几个 ρ 下表3四日期的紧急购电
RHO_SHOW = [0.30, 0.40, 0.50]

# 表1 指定单个 10 分钟时段；t = 起始分钟/10
TBL1_SLOTS = [("10:00-10:10", 60), ("12:00-12:10", 72), ("14:00-14:10", 84),
              ("16:00-16:10", 96), ("18:00-18:10", 108), ("20:00-20:10", 120)]
# 表2 六个 4 小时块
BLOCKS = [("0:00-4:00", 0, 24), ("4:00-8:00", 24, 48), ("8:00-12:00", 48, 72),
          ("12:00-16:00", 72, 96), ("16:00-20:00", 96, 120), ("20:00-24:00", 120, 144)]

# ============================ 读数据 ============================
price = pd.read_excel(ATT1, engine="openpyxl").iloc[:, 1].to_numpy(dtype=float)
assert price.shape == (144,), price.shape

load_all = pd.read_excel(ATT2, sheet_name="小区负载", engine="openpyxl").iloc[:, 1:145].to_numpy(dtype=float)
pv_all = pd.read_excel(ATT2, sheet_name="光伏发电实际功率", engine="openpyxl").iloc[:, 1:145].to_numpy(dtype=float)
assert load_all.shape == (365, 144) and pv_all.shape == (365, 144), (load_all.shape, pv_all.shape)

# 截取计划区间，并把 kW 换成每格 kWh
load_kw = load_all[IDX_START:IDX_END + 1]          # (334,144) kW
pv_kw = pv_all[IDX_START:IDX_END + 1]
load_e = load_kw * DT                              # (334,144) kWh
pv_e = pv_kw * DT
LOAD_PEAK = float(load_kw.max())                   # 全年最大负载(kW)，用于 Ḡ

# ============================ 建 LP（含参数，只建一次）============================
_SOLVERS = [s for s in ["HIGHS", "CLARABEL", "ECOS", "SCS"] if s in cp.installed_solvers()]
SOLVER = _SOLVERS[0] if _SOLVERS else None


def make_model():
    g = cp.Variable(144, nonneg=True)
    u = cp.Variable(144, nonneg=True)
    c = cp.Variable(144, nonneg=True)
    d = cp.Variable(144, nonneg=True)
    s = cp.Variable(145)
    Le = cp.Parameter(144, nonneg=True)
    PVe = cp.Parameter(144, nonneg=True)
    Gstep = cp.Parameter(nonneg=True)                 # 计划购电每格上限 (kWh)
    cons = [
        s[0] == SOC0, s[144] == SOC0,
        s >= SOC_MIN, s <= SOC_MAX,
        s[1:] == s[:-1] + ETA_C * c - d / ETA_D,
        c <= E_STEP_MAX, d <= E_STEP_MAX,
        g <= Gstep,
        g + u + PVe + d - c >= Le,
    ]
    obj = cp.Minimize(price @ g + EMERGENCY_MULT * (price @ u) + EPS_REG * cp.sum(c + d))
    return dict(prob=cp.Problem(obj, cons), g=g, u=u, c=c, d=d, s=s,
               Le=Le, PVe=PVe, Gstep=Gstep)


def solve_period(gbar_kw):
    """gbar_kw: 计划购电功率上限(kW)，None 表示 2A(无上限)。返回逐日结果数组。"""
    M = make_model()
    gstep = 1e9 if gbar_kw is None else gbar_kw * DT
    G = np.zeros((ND, 144)); U = np.zeros((ND, 144))
    C = np.zeros((ND, 144)); D = np.zeros((ND, 144))
    S = np.zeros((ND, 145))
    bad = []
    for i in range(ND):
        M["Le"].value = load_e[i]
        M["PVe"].value = pv_e[i]
        M["Gstep"].value = gstep
        M["prob"].solve(solver=getattr(cp, SOLVER) if SOLVER else None, warm_start=True)
        if M["prob"].status not in ("optimal", "optimal_inaccurate"):
            bad.append((i, M["prob"].status))
            continue
        G[i] = np.clip(M["g"].value, 0, None)
        U[i] = np.clip(M["u"].value, 0, None)
        C[i] = np.clip(M["c"].value, 0, None)
        D[i] = np.clip(M["d"].value, 0, None)
        S[i] = M["s"].value
    if bad:
        print(f"[警告] {len(bad)} 天未求得最优解: {bad[:5]} ...")
    plan_cost = float((G * price).sum())
    emer_cost = float(EMERGENCY_MULT * (U * price).sum())
    return dict(G=G, U=U, C=C, D=D, S=S,
               plan_cost=plan_cost, emer_cost=emer_cost, total_cost=plan_cost + emer_cost)


# ============================ 跑 2A ============================
print(f"求解器: {SOLVER or 'default'} | 计划天数: {ND} | 全年最大负载: {LOAD_PEAK:.1f} kW")
print("求解 2A（无计划购电上限）...")
r2a = solve_period(None)
n_emer_int = int((r2a["U"] > 1e-3).sum())
print(f"2A: 全年计划购电费 {r2a['plan_cost']:.2f} 元 | 紧急购电费 {r2a['emer_cost']:.2f} 元 "
      f"| 紧急购电时段数 {n_emer_int}")


def check_result(res):
    """物理合理性校验：同一格同时充放电 / SOC 越界 / 供电缺口。"""
    G, U, C, D, S = res["G"], res["U"], res["C"], res["D"], res["S"]
    both = int(((C > 1e-3) & (D > 1e-3)).sum())
    soc_lo = float(S.min()); soc_hi = float(S.max())
    supply = G + U + pv_e + D - C
    gap = float((load_e - supply).max())
    cyc = float((C.sum(axis=1) * ETA_C).mean())        # 日均充入电量（近似日均循环量）
    return dict(both=both, soc_lo=soc_lo, soc_hi=soc_hi, gap=gap, cyc=cyc)


chk = check_result(r2a)
print(f"2A 校验: 同格充放电={chk['both']} | SOC∈[{chk['soc_lo']:.1f},{chk['soc_hi']:.1f}] "
      f"| max(负载-供电)={chk['gap']:.2e} | 日均充电量≈{chk['cyc']:.0f} kWh")

# ============================ 跑 2B 灵敏度 ============================
print("求解 2B（外网容量受限）灵敏度 ...")
sens_rows = []
results_2b = {}
for rho in RHO_LIST:
    gbar = rho * LOAD_PEAK
    r = solve_period(gbar)
    results_2b[rho] = r
    ei = int((r["U"] > 1e-3).sum())
    ed = int((r["U"].sum(axis=1) > 1e-3).sum())
    sens_rows.append((rho, gbar, r["plan_cost"], r["emer_cost"], r["total_cost"], ei, ed))
    print(f"  ρ={rho:.2f}  Ḡ={gbar:7.1f}kW  计划费={r['plan_cost']:.1f}  "
          f"紧急费={r['emer_cost']:.1f}  总={r['total_cost']:.1f}  紧急时段={ei} 紧急天数={ed}")

r2b_primary = results_2b[RHO_PRIMARY_2B]
print(f"2B primary: ρ={RHO_PRIMARY_2B:.2f}  Ḡ={RHO_PRIMARY_2B*LOAD_PEAK:.1f} kW")

# ============================ 工具函数 ============================

def daily_tables(res, di):
    """给定 solve_period 结果与日索引，返回论文表1/表2所需数字。"""
    G, U, C, D, S = res["G"][di], res["U"][di], res["C"][di], res["D"][di], res["S"][di]
    t1 = [(name, float(G[t])) for name, t in TBL1_SLOTS]
    day_buy = float(G.sum())
    day_cost = float((G * price).sum() + EMERGENCY_MULT * (U * price).sum())
    t2 = [(name, float(C[a:b].sum()), float(D[a:b].sum())) for name, a, b in BLOCKS]
    emer = [(t, float(U[t])) for t in range(144) if U[t] > 1e-3]
    return t1, day_buy, day_cost, t2, S[0], S[144], emer


def slot_label(t):
    m0 = t * 10; m1 = (t + 1) * 10
    return f"{m0 // 60}:{m0 % 60:02d}-{m1 // 60}:{m1 % 60:02d}"


def fmt_date(d):
    return f"{d.year}/{d.month}/{d.day}"          # 2025/3/1 风格，跨平台


# ============================ 写 problem2_summary.txt ============================
lines = ["=" * 70, "问题 2  结果汇总", "=" * 70,
         f"求解器 {SOLVER or 'default'} | 效率 ηc=ηd={ETA_C:.4f} | 计划天数 {ND}",
         f"全年最大负载 {LOAD_PEAK:.1f} kW", ""]

lines += ["【2A 基础版：计划购电无上限】",
          f"  全年计划购电费 = {r2a['plan_cost']:.2f} 元",
          f"  全年紧急购电费 = {r2a['emer_cost']:.2f} 元   (完全信息 ⇒ 应为 0)",
          f"  全年总费用     = {r2a['total_cost']:.2f} 元",
          f"  出现紧急购电的时段数 = {n_emer_int}",
          f"  校验: 同格同时充放电时段数 = {chk['both']}  (期望 0)",
          f"  校验: SOC 全程 ∈ [{chk['soc_lo']:.1f}, {chk['soc_hi']:.1f}]  (界 [1200,10800])",
          f"  校验: max(负载 - 供电) = {chk['gap']:.2e} kWh  (≤0 合格)",
          f"  储能日均充入电量 ≈ {chk['cyc']:.0f} kWh (≈ {chk['cyc']/12000:.2f} 倍额定容量/天)", ""]

for D_ in TBL3_DATES:
    di = (D_ - PLAN_START).days
    t1, day_buy, day_cost, t2, s0, s24, emer = daily_tables(r2a, di)
    lines.append("-" * 70)
    lines.append(f"表1/表2  {D_:%Y-%m-%d}   (2A)")
    lines.append("  表1 指定时段购电量(kWh):")
    for name, v in t1:
        lines.append(f"    {name:<14s}{v:12.4f}")
    lines.append(f"    {'全天购电量':<14s}{day_buy:12.4f}")
    lines.append(f"    {'全天购电费':<14s}{day_cost:12.4f}")
    lines.append("  表2 指定时段充/放电量(kWh):")
    for name, cc, dd in t2:
        lines.append(f"    {name:<14s} 充 {cc:11.4f}   放 {dd:11.4f}")
    lines.append(f"    0:00 储电量 = {s0:.2f}    24:00 储电量 = {s24:.2f}")
    lines.append(f"  表3 紧急购电: {'无' if not emer else ''}")
    for t, v in emer:
        lines.append(f"    {slot_label(t):<16s}{v:12.4f}")
lines.append("")

lines += ["【2B 拓展版：外网正常供电容量 Ḡ = ρ·全年最大负载】",
          f"{'ρ':>6s}{'Ḡ(kW)':>12s}{'计划购电费':>16s}{'紧急购电费':>16s}"
          f"{'总费用':>16s}{'紧急时段':>10s}{'紧急天数':>10s}"]
for rho, gbar, pc, ec, tc, ei, ed in sens_rows:
    lines.append(f"{rho:6.2f}{gbar:12.1f}{pc:16.1f}{ec:16.1f}{tc:16.1f}{ei:10d}{ed:10d}")
lines.append("")

for rho in RHO_SHOW:
    res = results_2b[rho]
    lines.append(f"—— ρ={rho:.2f}（Ḡ={rho*LOAD_PEAK:.1f}kW）表3四日期的紧急购电 ——")
    for D_ in TBL3_DATES:
        di = (D_ - PLAN_START).days
        _, _, _, _, _, _, emer = daily_tables(res, di)
        tot = sum(v for _, v in emer)
        lines.append(f"  {D_:%Y-%m-%d}: {'无' if not emer else f'合计 {tot:.2f} kWh，{len(emer)} 个时段'}")
        for t, v in emer:
            lines.append(f"    {slot_label(t):<16s}{v:12.4f}")

(OUTDIR / "problem2_summary.txt").write_text("\n".join(lines), encoding="utf-8")
(OUTDIR / "problem2_sensitivity_capacity.txt").write_text(
    "\n".join(["问题2 - 2B 外网容量灵敏度  (Ḡ = ρ·全年最大负载 = ρ·%.1f kW)" % LOAD_PEAK,
               f"{'ρ':>6s}{'Ḡ(kW)':>12s}{'计划购电费(元)':>18s}{'紧急购电费(元)':>18s}"
               f"{'总费用(元)':>16s}{'紧急时段数':>12s}{'紧急天数':>10s}"]
              + [f"{rho:6.2f}{gbar:12.1f}{pc:18.1f}{ec:18.1f}{tc:16.1f}{ei:12d}{ed:10d}"
                 for rho, gbar, pc, ec, tc, ei, ed in sens_rows]),
    encoding="utf-8")
print(f"已写出: {OUTDIR/'problem2_summary.txt'}")
print(f"已写出: {OUTDIR/'problem2_sensitivity_capacity.txt'}")


# ============================ 写 result2 系列 xlsx ============================

def write_result2(res, out_path):
    from openpyxl import load_workbook
    wb = load_workbook(TEMPLATE2)
    for ws in wb.worksheets:                       # 先解除所有合并单元格，便于写值
        for rng in list(ws.merged_cells.ranges):
            ws.unmerge_cells(str(rng))

    # --- sheet 计划购电量: 每天一行, B..EO = 144 格, EP=全天购电量, EQ=全天购电费 ---
    ws1 = wb["计划购电量"]
    for i, D_ in enumerate(DATES):
        r = 2 + i
        ws1.cell(r, 1, fmt_date(D_))
        gi = res["G"][i]
        for t in range(144):
            ws1.cell(r, 2 + t, round(float(gi[t]), 6))
        ws1.cell(r, 146, round(float(gi.sum()), 6))
        ws1.cell(r, 147, round(float((gi * price).sum()
                                     + EMERGENCY_MULT * (res["U"][i] * price).sum()), 6))

    # --- sheet 充放电量: 每天 6 行(4h 块) ---
    ws2 = wb["充放电量"]
    for i, D_ in enumerate(DATES):
        base = 2 + i * 6
        Ci, Di, Si = res["C"][i], res["D"][i], res["S"][i]
        for b, (name, a, z) in enumerate(BLOCKS):
            r = base + b
            ws2.cell(r, 1, fmt_date(D_) if b == 0 else None)
            ws2.cell(r, 2, name)
            ws2.cell(r, 3, round(float(Ci[a:z].sum()), 6))
            ws2.cell(r, 4, round(float(Di[a:z].sum()), 6))
        ws2.cell(base, 5, "0:00");  ws2.cell(base, 6, round(float(Si[0]), 6))
        ws2.cell(base + 1, 5, "24:00"); ws2.cell(base + 1, 6, round(float(Si[144]), 6))

    # --- sheet 紧急购电量 ---
    ws3 = wb["紧急购电量"]
    for row in ws3.iter_rows(min_row=2):           # 清空旧占位内容
        for cell in row:
            cell.value = None
    r = 2
    for i, D_ in enumerate(DATES):
        emer = [(t, float(res["U"][i][t])) for t in range(144) if res["U"][i][t] > 1e-3]
        if not emer:
            ws3.cell(r, 1, fmt_date(D_)); ws3.cell(r, 2, "无"); ws3.cell(r, 3, 0)
            r += 1
        else:
            for k, (t, v) in enumerate(emer):
                ws3.cell(r, 1, fmt_date(D_) if k == 0 else None)
                ws3.cell(r, 2, slot_label(t))
                ws3.cell(r, 3, round(v, 6))
                r += 1
    wb.save(out_path)
    print(f"已写出: {out_path}")


write_result2(r2a, OUTDIR / "result2.xlsx")
if r2b_primary is not None:
    write_result2(r2b_primary, OUTDIR / "result2_capacity_limited.xlsx")


# ============================ 可选作图（表3 四日期）============================
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass
    hh = np.arange(144) * DT
    for D_ in TBL3_DATES:
        di = (D_ - PLAN_START).days
        G, U, C, D, S = (r2a[k][di] for k in ("G", "U", "C", "D", "S"))
        fig, ax = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
        ax[0].plot(hh, load_kw[di], "k", label="负载")
        ax[0].plot(hh, pv_kw[di], color="tab:orange", label="光伏")
        ax[0].plot(hh, G / DT, color="tab:blue", label="计划购电功率")
        ax[0].plot(hh, (D - C) / DT, color="tab:green", label="储能净放电功率")
        if U.max() > 1e-3:
            ax[0].plot(hh, U / DT, color="tab:red", label="紧急购电功率")
        ax[0].legend(ncol=3, fontsize=8); ax[0].set_ylabel("功率 (kW)")
        ax[0].set_title(f"问题2 (2A)  {D_:%Y-%m-%d}")
        ax[1].plot(np.arange(145) * DT, S, color="tab:purple")
        ax[1].axhline(SOC_MIN, ls="--", c="gray"); ax[1].axhline(SOC_MAX, ls="--", c="gray")
        ax[1].set_ylabel("储电量 (kWh)"); ax[1].set_xlabel("时刻 (h)")
        fig.tight_layout()
        fig.savefig(OUTDIR / f"problem2_plot_{D_:%Y%m%d}.png", dpi=150)
        plt.close(fig)
    print("已写出: results/problem2_plot_*.png")
except Exception as e:  # noqa: BLE001
    print(f"[提示] 未作图（{e}）")
