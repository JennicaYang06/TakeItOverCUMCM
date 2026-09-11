"""
问题2：负荷/光伏随日变化 + 预测 + 每日 0:00 计划购电（全年滚动仿真）

思路（在问题1单日模型基础上）：
  1. 电价复用附件1（"每天电价相同"），每天固定不变。
  2. 每天0:00并不知道当天真实负荷/光伏，只能用"过去数据"预测出当天的负荷、光伏曲线，
     代入问题1的单日 MILP 求得当天的计划购电 g、充放电 c/d（储能按计划执行，不随实际负荷/光伏调整）。
  3. 白天过去后，用附件2的真实负荷/光伏回代：若 g+光伏(实际)+d-c < 负荷(实际)，
     缺口按当时电价的5倍紧急购电；多余电量视为弃用（无收益）。
  4. 储能电量在全年内连续滚动（第二天初始电量 = 前一天计划末电量），
     不再要求problem1那样"0:00=24:00"的单日闭环。
  5. 从 2025-1-1（SOC0=6000kWh）开始逐日仿真，用1月做预测模型的历史预热，
     只导出 2025-2-1~12-31 (334天) 到 result2.xlsx。
  6. 安全边际（newsvendor）：逐日独立的MILP只对"点预测"取等号满足，没有理由为吸收预测误差
     多买电，导致储能天天被放空、几乎每天都有紧急购电。多买1单位电正常价p，白买了（用不完，
     无残值）就浪费p；少买1单位、缺口要按5倍价紧急买单，比提前买多花 5p-p=4p。按经典报童模型，
     最优服务水位（即目标净负荷分位数）= 4p/(4p+p) = 0.8，与p本身无关。故把送入LP的负荷目标
     由"点预测"上移到"(负荷-光伏)预测误差的滚动80分位数"，逐时刻单独估计（PV在夜间误差趋近0，
     白天误差更大，因此不能用统一边际）。
"""
from pathlib import Path
import sys
import time
import datetime as dt
import numpy as np
import pandas as pd
import cvxpy as cp

np.set_printoptions(suppress=True)

# ===================== 公共参数（附录 1，与问题1一致）=====================
DT = 1.0 / 6.0                    # 每格时长（小时）
SOC_MIN, SOC_MAX = 1200.0, 10800.0
SOC0_INIT = 6000.0                 # 2025-1-1 0:00 初始电量
P_MAX = 5000.0                     # 最大充放电功率 kW
E_STEP_MAX = P_MAX * DT             # 每格最大充/放电量 kWh
RT_EFF = 0.90                       # 充放电总效率（往返）
ETA_C = np.sqrt(RT_EFF)
ETA_D = np.sqrt(RT_EFF)
T = 144
EMERGENCY_MULT = 5.0                # 紧急购电价格倍数

SOLVER = "HIGHS"                    # 本机没装 GLPK_MI，用开源 HIGHS（MILP 求解快且稳定）

# ===================== 预测模型超参数 =====================
LOAD_ALPHA = 0.25   # 负荷 Holt-Winters 水平项平滑系数
LOAD_GAMMA = 0.25   # 负荷 Holt-Winters 周季节项平滑系数（周期=7天）
PV_WINDOW = 30      # 光伏"晴空包络"滚动窗口（天）
PV_QUANTILE = 0.90  # 包络分位数
PV_BETA = 0.35      # 光伏晴空指数指数平滑系数

USE_SAFETY_MARGIN = True     # 是否在点预测基础上叠加报童安全边际
# 报童临界分位数：欠量单位成本 Cu=(EMERGENCY_MULT-1)*p（紧急价比计划价多付的部分），
# 超量单位成本 Co=p（多买用不完，无残值）。q* = Cu/(Cu+Co)，与价格 p 本身无关。
SAFETY_QUANTILE = (EMERGENCY_MULT - 1) / EMERGENCY_MULT   # = 0.8
SAFETY_WINDOW = 60           # 净负荷预测误差滚动分位数的历史窗口（天）

EXPORT_START = "2025-02-01"
EXPORT_END = "2025-12-31"
REP_DATES = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]  # 论文表3指定日期


# ======================= 数据路径（兼容 code/附件 本地拷贝 或 仓库根目录原始附件）=======================
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def _find_dir(candidates):
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("找不到数据目录，尝试过：\n" + "\n".join(str(c) for c in candidates))


DATA_DIR = _find_dir([
    HERE / "附件",
    ROOT / "题目和附件" / "C题" / "附件",
])
TEMPLATE_DIR = _find_dir([
    DATA_DIR / "附件5",
    ROOT / "题目和附件" / "C题" / "附件" / "附件5",
])
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

print(f"[数据目录] {DATA_DIR}")
print(f"[模板目录] {TEMPLATE_DIR}")


# ========== 首尾相接平均函数（与问题1一致：把整点瞬时功率折算成"区间平均功率"）==========
def adjacent_average(arr):
    arr = np.asarray(arr, dtype=float)
    result = np.zeros(len(arr))
    result[0] = (arr[-1] + arr[0]) / 2
    result[1:] = (arr[:-1] + arr[1:]) / 2
    return result


def time_range_str(i):
    start_min, end_min = i * 10, (i + 1) * 10
    if end_min >= 1440:
        end_h, end_m = 24, 0
    else:
        end_h, end_m = end_min // 60, end_min % 60
    start_h, start_m = start_min // 60, start_min % 60
    return f"{start_h:02d}:{start_m:02d}-{end_h:02d}:{end_m:02d}"


# ======================= 读取附件1：电价（每天固定）+ 预测初值种子曲线 =======================
a1 = pd.read_excel(DATA_DIR / "附件1.xlsx", header=None).values
price_raw = a1[1:, 1].astype(float)
load_kw_seed = a1[1:, 2].astype(float)   # 用作预测模型第0天的先验种子（无历史数据时的初始猜测）
pv_kw_seed = a1[1:, 3].astype(float)
assert len(price_raw) == T, f"附件1 应为 144 行，实际 {len(price_raw)}"
price_avg = adjacent_average(price_raw)   # 全年每天复用同一条电价曲线

# ======================= 读取附件2：全年实际负荷 / 光伏 =======================
xls2 = DATA_DIR / "附件2.xlsx"
df_load = pd.read_excel(xls2, sheet_name="小区负载", header=0)
df_pv = pd.read_excel(xls2, sheet_name="光伏发电实际功率", header=0)

dates = pd.to_datetime(df_load.iloc[:, 0]).dt.normalize().to_numpy()
load_kw_all = df_load.iloc[:, 1:1 + T].to_numpy(dtype=float)
pv_kw_all = df_pv.iloc[:, 1:1 + T].to_numpy(dtype=float)
n_days = len(dates)
assert load_kw_all.shape == (n_days, T) and pv_kw_all.shape == (n_days, T)
dates_str = np.array([pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates])

if len(sys.argv) > 1:
    n_days = min(n_days, int(sys.argv[1]))
    print(f"[调试模式] 仅仿真前 {n_days} 天")

export_start_idx = int(np.where(dates_str == EXPORT_START)[0][0])
export_end_idx = int(np.where(dates_str == EXPORT_END)[0][0]) if EXPORT_END in dates_str else n_days - 1
export_end_idx = min(export_end_idx, n_days - 1)
do_export = export_start_idx < n_days
if not do_export:
    print(f"[调试模式] n_days={n_days} 未覆盖导出起点，跳过 result2.xlsx 导出")
print(f"[导出区间] {EXPORT_START} (day_idx={export_start_idx}) ~ {EXPORT_END} (day_idx={export_end_idx})")


# ======================= 预测模型 =======================
class LoadForecaster:
    """按144个时刻分别做加性 Holt-Winters（周期=7天，无趋势项）滚动预测。"""

    def __init__(self, seed_curve, alpha=LOAD_ALPHA, gamma=LOAD_GAMMA):
        self.alpha = alpha
        self.gamma = gamma
        self.level = seed_curve.copy()
        self.seasonal = np.zeros((len(seed_curve), 7))

    def predict(self, day_idx):
        dow = day_idx % 7
        return np.clip(self.level + self.seasonal[:, dow], 0, None)

    def update(self, day_idx, actual):
        dow = day_idx % 7
        pred = self.level + self.seasonal[:, dow]
        err = actual - pred
        self.level = self.level + self.alpha * err
        self.seasonal[:, dow] = self.seasonal[:, dow] + self.gamma * err


class PVForecaster:
    """晴空包络（滚动分位数，反映季节性最大出力）× 晴空指数（指数平滑，反映近期天气持续性）。"""

    def __init__(self, seed_curve, window=PV_WINDOW, quantile=PV_QUANTILE, beta=PV_BETA):
        self.window = window
        self.quantile = quantile
        self.beta = beta
        self.seed_curve = seed_curve
        self.buffer = []
        self.k_ewma = np.ones(len(seed_curve))

    def _envelope(self):
        if not self.buffer:
            return np.maximum(self.seed_curve, 1e-6)
        arr = np.array(self.buffer[-self.window:])
        return np.maximum(np.quantile(arr, self.quantile, axis=0), 1e-6)

    def predict(self, day_idx):
        return np.clip(self.k_ewma * self._envelope(), 0, None)

    def update(self, day_idx, actual):
        env = self._envelope()
        k_today = np.clip(np.where(env > 1e-6, actual / env, 0.0), 0, 1.5)
        self.k_ewma = self.beta * k_today + (1 - self.beta) * self.k_ewma
        self.buffer.append(actual)


class NetErrorTracker:
    """按144个时刻分别维护"净负荷(=负荷-光伏)预测残差"的滚动历史，供报童安全边际取分位数。"""

    def __init__(self, T, window=SAFETY_WINDOW, quantile=SAFETY_QUANTILE):
        self.window = window
        self.quantile = quantile
        self.buffer = []
        self.T = T

    def get_margin(self, day_idx):
        if not self.buffer:
            return np.zeros(self.T)
        arr = np.array(self.buffer[-self.window:])
        return np.quantile(arr, self.quantile, axis=0)

    def update(self, forecast_net_e, actual_net_e):
        self.buffer.append(actual_net_e - forecast_net_e)


# ======================= 每日 MILP（复用问题1结构，soc0/负荷/光伏做成 Parameter 以复用编译结果）=======================
g = cp.Variable(T, nonneg=True, name="plan_buy")
c = cp.Variable(T, nonneg=True, name="charge")
d = cp.Variable(T, nonneg=True, name="discharge")
z = cp.Variable(T, boolean=True, name="charge_discharge_flag")
soc = cp.Variable(T + 1, name="soc")

soc0_p = cp.Parameter(name="soc0")
load_e_p = cp.Parameter(T, name="load_e")
pv_e_p = cp.Parameter(T, name="pv_e")

cons = [
    soc[0] == soc0_p,
    soc >= SOC_MIN,
    soc <= SOC_MAX,
    soc[1:] == soc[:-1] + ETA_C * c - d / ETA_D,
    c <= E_STEP_MAX,
    d <= E_STEP_MAX,
    g + pv_e_p + d - c >= load_e_p,
    c <= E_STEP_MAX * z,
    d <= E_STEP_MAX * (1 - z),
]
prob = cp.Problem(cp.Minimize(price_avg @ g), cons)


def solve_day(soc0_val, load_e_val, pv_e_val):
    soc0_p.value = float(soc0_val)
    load_e_p.value = load_e_val
    pv_e_p.value = pv_e_val
    prob.solve(solver=SOLVER)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"day solve failed: status={prob.status}")
    gv = np.clip(np.asarray(g.value).ravel(), 0.0, None)
    cv = np.clip(np.asarray(c.value).ravel(), 0.0, None)
    dv = np.clip(np.asarray(d.value).ravel(), 0.0, None)
    socv = np.asarray(soc.value).ravel()
    return gv, cv, dv, socv


# ======================= 全年滚动仿真 =======================
load_forecaster = LoadForecaster(load_kw_seed)
pv_forecaster = PVForecaster(pv_kw_seed)
net_error_tracker = NetErrorTracker(T)

soc_prev_end = SOC0_INIT
results = []
t_start = time.time()

for day_idx in range(n_days):
    f_load_kw = load_forecaster.predict(day_idx)
    f_pv_kw = pv_forecaster.predict(day_idx)
    f_load_e = adjacent_average(f_load_kw) * DT
    f_pv_e = adjacent_average(f_pv_kw) * DT
    f_net_e = f_load_e - f_pv_e

    if USE_SAFETY_MARGIN:
        margin = net_error_tracker.get_margin(day_idx)
    else:
        margin = np.zeros(T)
    f_load_e_target = np.clip(f_load_e + margin, 0.0, None)

    gv, cv, dv, socv = solve_day(soc_prev_end, f_load_e_target, f_pv_e)

    actual_load_kw = load_kw_all[day_idx]
    actual_pv_kw = pv_kw_all[day_idx]
    actual_load_e = adjacent_average(actual_load_kw) * DT
    actual_pv_e = adjacent_average(actual_pv_kw) * DT
    actual_net_e = actual_load_e - actual_pv_e

    actual_supply = gv + actual_pv_e + dv - cv
    deficit_e = np.clip(actual_load_e - actual_supply, 0.0, None)
    emerg_cost_t = EMERGENCY_MULT * price_avg * deficit_e

    plan_cost = float(price_avg @ gv)
    emerg_cost = float(emerg_cost_t.sum())

    results.append(dict(
        day_idx=day_idx, date=dates_str[day_idx],
        gv=gv, cv=cv, dv=dv, socv=socv,
        f_load_kw=f_load_kw, f_pv_kw=f_pv_kw,
        actual_load_kw=actual_load_kw, actual_pv_kw=actual_pv_kw,
        deficit_e=deficit_e, plan_cost=plan_cost, emerg_cost=emerg_cost,
    ))

    load_forecaster.update(day_idx, actual_load_kw)
    pv_forecaster.update(day_idx, actual_pv_kw)
    net_error_tracker.update(f_net_e, actual_net_e)
    soc_prev_end = float(socv[-1])

    if day_idx % 30 == 0 or day_idx == n_days - 1:
        print(f"[{day_idx + 1}/{n_days}] {dates_str[day_idx]}  "
              f"计划购电费={plan_cost:8.2f}  紧急购电费={emerg_cost:8.2f}  soc_end={soc_prev_end:8.1f}")

print(f"\n仿真完成，用时 {time.time() - t_start:.1f}s")
print(f"安全边际: {'启用 (q=' + str(SAFETY_QUANTILE) + ')' if USE_SAFETY_MARGIN else '未启用（点预测基线）'}")

# ======================= 预测误差评估（仅统计导出区间 2-12月）=======================
exp_slice = results[export_start_idx:export_end_idx + 1]
f_load_mat = np.array([r["f_load_kw"] for r in exp_slice])
a_load_mat = np.array([r["actual_load_kw"] for r in exp_slice])
f_pv_mat = np.array([r["f_pv_kw"] for r in exp_slice])
a_pv_mat = np.array([r["actual_pv_kw"] for r in exp_slice])


def error_metrics(pred, actual, mape_thresh=None):
    err = pred - actual
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    if mape_thresh is not None:
        mask = actual > mape_thresh
    else:
        mask = np.ones_like(actual, dtype=bool)
    mape = float(np.mean(np.abs(err[mask] / actual[mask])) * 100) if mask.sum() > 0 else float("nan")
    return mae, rmse, mape


load_mae, load_rmse, load_mape = error_metrics(f_load_mat, a_load_mat)
pv_mae, pv_rmse, pv_mape = error_metrics(f_pv_mat, a_pv_mat, mape_thresh=50.0)  # PV 只在有明显出力时算 MAPE

total_plan_cost = sum(r["plan_cost"] for r in exp_slice)
total_emerg_cost = sum(r["emerg_cost"] for r in exp_slice)
emerg_days = sum(1 for r in exp_slice if r["emerg_cost"] > 1e-6)
emerg_blocks = sum(int(np.sum(r["deficit_e"] > 1e-6)) for r in exp_slice)

print("\n===== 预测误差（2-12月） =====")
print(f"负荷: MAE={load_mae:.2f}kW RMSE={load_rmse:.2f}kW MAPE={load_mape:.2f}%")
print(f"光伏: MAE={pv_mae:.2f}kW RMSE={pv_rmse:.2f}kW MAPE(出力>50kW时)={pv_mape:.2f}%")
print("\n===== 全年费用（2-12月，334天） =====")
print(f"计划购电费合计 = {total_plan_cost:.2f} 元")
print(f"紧急购电费合计 = {total_emerg_cost:.2f} 元")
print(f"总费用 = {total_plan_cost + total_emerg_cost:.2f} 元")
print(f"发生紧急购电的天数 = {emerg_days} / {len(exp_slice)}")
print(f"发生紧急购电的时段数 = {emerg_blocks}")


# ======================= 写入 result2.xlsx 模板 =======================
def merge_deficit_segments(deficit_e, eps=1e-6):
    segs = []
    t = 0
    n = len(deficit_e)
    while t < n:
        if deficit_e[t] > eps:
            start = t
            s = 0.0
            while t < n and deficit_e[t] > eps:
                s += deficit_e[t]
                t += 1
            segs.append((start, t, s))
        else:
            t += 1
    return segs


try:
    if not do_export:
        raise RuntimeError("调试模式下仿真天数未覆盖导出区间，跳过导出")
    from openpyxl import load_workbook

    template_file = TEMPLATE_DIR / "result2.xlsx"
    wb = load_workbook(template_file)

    # ---------- Sheet1: 计划购电量 ----------
    ws1 = wb["计划购电量"]
    for i in range(export_start_idx, export_end_idx + 1):
        r = results[i]
        row = 2 + (i - export_start_idx)
        gv = r["gv"]
        for col in range(2, 145):          # col2..144 -> gv[1..143]
            ws1.cell(row=row, column=col, value=round(float(gv[col - 1]), 6))
        # col145: 下一天第一个时段 gv[0]（若无下一天数据，退化为当天自身 gv[0]）
        next_gv0 = results[i + 1]["gv"][0] if i + 1 < n_days else gv[0]
        ws1.cell(row=row, column=145, value=round(float(next_gv0), 6))
        ws1.cell(row=row, column=146, value=round(float(gv.sum()), 6))
        ws1.cell(row=row, column=147, value=round(float(price_avg @ gv), 6))

    # ---------- Sheet2: 充放电量 ----------
    ws2 = wb["充放电量"]
    if ws2.max_row > 1:
        ws2.delete_rows(2, ws2.max_row - 1)
    blocks = [("0:00-4:00", 0, 24), ("4:00-8:00", 24, 48), ("8:00-12:00", 48, 72),
              ("12:00-16:00", 72, 96), ("16:00-20:00", 96, 120), ("20:00-24:00", 120, 144)]
    row = 2
    for i in range(export_start_idx, export_end_idx + 1):
        r = results[i]
        cv, dv, socv = r["cv"], r["dv"], r["socv"]
        date_obj = dt.datetime.strptime(r["date"], "%Y-%m-%d")
        for bi, (label, a, b) in enumerate(blocks):
            ws2.cell(row=row, column=1, value=date_obj if bi == 0 else None)
            ws2.cell(row=row, column=2, value=label)
            ws2.cell(row=row, column=3, value=round(float(cv[a:b].sum()), 6))
            ws2.cell(row=row, column=4, value=round(float(dv[a:b].sum()), 6))
            if bi == 0:
                ws2.cell(row=row, column=5, value=dt.time(0, 0))
                ws2.cell(row=row, column=6, value=round(float(socv[0]), 6))
            elif bi == 1:
                ws2.cell(row=row, column=5, value="24:00")
                ws2.cell(row=row, column=6, value=round(float(socv[-1]), 6))
            row += 1

    # ---------- Sheet3: 紧急购电量 ----------
    ws3 = wb["紧急购电量"]
    if ws3.max_row > 1:
        ws3.delete_rows(2, ws3.max_row - 1)
    row = 2
    for i in range(export_start_idx, export_end_idx + 1):
        r = results[i]
        segs = merge_deficit_segments(r["deficit_e"])
        if not segs:
            continue
        date_obj = dt.datetime.strptime(r["date"], "%Y-%m-%d")
        for si, (s, e, val) in enumerate(segs):
            ws3.cell(row=row, column=1, value=date_obj if si == 0 else None)
            ws3.cell(row=row, column=2, value=time_range_str(s)[:5] + "-" + time_range_str(e - 1)[6:])
            ws3.cell(row=row, column=3, value=round(float(val), 6))
            row += 1

    wb.save(RESULTS_DIR / "result2.xlsx")
    print(f"\n[模板填充] 已写入: {RESULTS_DIR / 'result2.xlsx'}")
except Exception as e:
    print(f"[提示] 未写入 result2.xlsx: {e}")
    if do_export:
        raise


# ======================= 论文素材：4个代表日的表1/表2/表3 + 汇总 =======================
TBL1 = [("10:00-10:10", 60), ("12:00-12:10", 72), ("14:00-14:10", 84),
        ("16:00-16:10", 96), ("18:00-18:10", 108), ("20:00-20:10", 120)]
TBL2 = [("0:00-4:00", 0, 24), ("4:00-8:00", 24, 48), ("8:00-12:00", 48, 72),
        ("12:00-16:00", 72, 96), ("16:00-20:00", 96, 120), ("20:00-24:00", 120, 144)]

lines = []
lines.append("=" * 70)
lines.append("问题2  结果汇总")
lines.append("=" * 70)
lines.append(f"求解器: {SOLVER}  (ETA_C={ETA_C:.4f}, ETA_D={ETA_D:.4f})")
lines.append(f"负荷预测: Holt-Winters(alpha={LOAD_ALPHA}, gamma={LOAD_GAMMA}, 周期=7)")
lines.append(f"光伏预测: 晴空包络(window={PV_WINDOW}, q={PV_QUANTILE}) x 晴空指数EWMA(beta={PV_BETA})")
lines.append("")
lines.append("预测误差（2025.2.1-12.31）")
lines.append("-" * 70)
lines.append(f"  负荷  MAE={load_mae:8.2f} kW   RMSE={load_rmse:8.2f} kW   MAPE={load_mape:6.2f}%")
lines.append(f"  光伏  MAE={pv_mae:8.2f} kW   RMSE={pv_rmse:8.2f} kW   MAPE(出力>50kW)={pv_mape:6.2f}%")
lines.append("")
lines.append("全年费用（2025.2.1-12.31，334天）")
lines.append("-" * 70)
lines.append(f"  计划购电费合计 = {total_plan_cost:12.2f} 元")
lines.append(f"  紧急购电费合计 = {total_emerg_cost:12.2f} 元")
lines.append(f"  总费用         = {total_plan_cost + total_emerg_cost:12.2f} 元")
lines.append(f"  发生紧急购电天数 = {emerg_days} / {len(exp_slice)}")
lines.append(f"  发生紧急购电时段数 = {emerg_blocks}")
lines.append("")

for rd in REP_DATES:
    idx = int(np.where(dates_str == rd)[0][0])
    if idx >= len(results):
        continue
    r = results[idx]
    gv, cv, dv, socv, deficit_e = r["gv"], r["cv"], r["dv"], r["socv"], r["deficit_e"]
    lines.append("=" * 70)
    lines.append(f"代表日 {rd}")
    lines.append("=" * 70)
    lines.append("表1  微网在指定时间段的购电量 (kWh)")
    for name, t in TBL1:
        lines.append(f"  {name:<14s} {gv[t]:12.4f}")
    lines.append(f"  {'全天购电量 (kWh)':<14s} {gv.sum():12.4f}")
    lines.append(f"  {'全天购电费 (元)':<14s} {float(price_avg @ gv):12.2f}")
    lines.append("")
    lines.append("表2  储能设备在指定时间段的充放电量 (kWh)")
    lines.append(f"  {'时间段':<14s} {'充电量':>12s} {'放电量':>12s}")
    for name, a, b in TBL2:
        lines.append(f"  {name:<14s} {cv[a:b].sum():12.4f} {dv[a:b].sum():12.4f}")
    lines.append(f"  0:00 储电量 (kWh)  = {socv[0]:.4f}")
    lines.append(f"  24:00 储电量 (kWh) = {socv[-1]:.4f}")
    lines.append("")
    lines.append("表3  紧急购电")
    segs = merge_deficit_segments(deficit_e)
    if not segs:
        lines.append("  (无紧急购电)")
    else:
        for s, e, val in segs:
            seg_label = time_range_str(s)[:5] + "-" + time_range_str(e - 1)[6:]
            lines.append(f"  {seg_label:<14s} {val:12.4f}")
    lines.append("")

report = "\n".join(lines)
print("\n" + report)

summary_path = RESULTS_DIR / "problem2_summary.txt"
summary_path.write_text(report, encoding="utf-8")
print(f"\n[汇总] 已写入: {summary_path}")

# ======================= 可选：4个代表日的曲线图 =======================
try:
    import matplotlib.pyplot as plt
    try:
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

    h = np.arange(T) * DT
    for rd in REP_DATES:
        idx = int(np.where(dates_str == rd)[0][0])
        if idx >= len(results):
            continue
        r = results[idx]
        gv, cv, dv, socv = r["gv"], r["cv"], r["dv"], r["socv"]
        fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        ax[0].plot(h, price_raw, color="tab:red")
        ax[0].set_ylabel("电价 (元/kWh)")
        ax[0].set_title(f"问题2：{rd} 电价 / 功率平衡 / 储能电量")
        ax[1].plot(h, r["actual_load_kw"], label="实际负荷", color="k")
        ax[1].plot(h, r["actual_pv_kw"], label="实际光伏", color="tab:orange")
        ax[1].plot(h, r["f_load_kw"], label="预测负荷", color="k", ls="--", alpha=0.6)
        ax[1].plot(h, r["f_pv_kw"], label="预测光伏", color="tab:orange", ls="--", alpha=0.6)
        ax[1].plot(h, gv / DT, label="计划购电功率", color="tab:blue")
        ax[1].plot(h, (dv - cv) / DT, label="储能净放电功率", color="tab:green")
        ax[1].legend(ncol=3, fontsize=7)
        ax[1].set_ylabel("功率 (kW)")
        ax[2].plot(np.arange(T + 1) * DT, socv, color="tab:purple")
        ax[2].axhline(SOC_MIN, ls="--", c="gray")
        ax[2].axhline(SOC_MAX, ls="--", c="gray")
        ax[2].set_ylabel("储电量 (kWh)")
        ax[2].set_xlabel("时刻 (h)")
        fig.tight_layout()
        out_png = RESULTS_DIR / f"problem2_plot_{rd.replace('-', '')}.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"[作图] 已保存: {out_png}")
except Exception as e:
    print(f"[提示] 未作图（{e}）")

print("\n完成。")
