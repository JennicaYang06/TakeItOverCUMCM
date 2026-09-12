"""
问题2：负荷/光伏随日变化 + 预测 + 每日 0:00 计划购电（全年滚动仿真）
预测方法：Holt-Winters（负荷）+ 晴空包络×晴空指数EWMA（光伏）。

思路（在问题1单日模型基础上，公共部分见 day_ahead_common.py）：
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
     多买电，导致储能天天被放空、几乎每天都有紧急购电。按经典报童模型，最优服务水位
     = 4p/(4p+p) = 0.8，与p本身无关。故把送入LP的负荷目标由"点预测"上移到"(负荷-光伏)
     预测误差的滚动80分位数"，逐时刻单独估计。
"""
from pathlib import Path
import sys
import time
import numpy as np

import day_ahead_common as dac

np.set_printoptions(suppress=True)

# ===================== 预测模型超参数 =====================
LOAD_ALPHA = 0.25   # 负荷 Holt-Winters 水平项平滑系数
LOAD_GAMMA = 0.25   # 负荷 Holt-Winters 周季节项平滑系数（周期=7天）
PV_WINDOW = 30      # 光伏"晴空包络"滚动窗口（天）
PV_QUANTILE = 0.90  # 包络分位数
PV_BETA = 0.35      # 光伏晴空指数指数平滑系数

USE_SAFETY_MARGIN = True     # 是否在点预测基础上叠加报童安全边际

T, DT = dac.T, dac.DT

HERE = Path(__file__).resolve().parent
_root, DATA_DIR, TEMPLATE_DIR = dac.resolve_data_dirs(HERE)
RESULTS_DIR = _root / "results" / "problem2" / "holtwinters"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"[数据目录] {DATA_DIR}")
print(f"[模板目录] {TEMPLATE_DIR}")

pack = dac.load_price_and_actuals(DATA_DIR)
price_raw, price_avg = pack["price_raw"], pack["price_avg"]
load_kw_seed, pv_kw_seed = pack["load_kw_seed"], pack["pv_kw_seed"]
dates_str = pack["dates_str"]
load_kw_all, pv_kw_all = pack["load_kw_all"], pack["pv_kw_all"]
n_days = pack["n_days"]

if len(sys.argv) > 1:
    n_days = min(n_days, int(sys.argv[1]))
    print(f"[调试模式] 仅仿真前 {n_days} 天")

export_start_idx, export_end_idx, do_export = dac.compute_export_range(dates_str, n_days)
if not do_export:
    print(f"[调试模式] n_days={n_days} 未覆盖导出起点，跳过 result2.xlsx 导出")
print(f"[导出区间] {dac.EXPORT_START} (day_idx={export_start_idx}) ~ {dac.EXPORT_END} (day_idx={export_end_idx})")


# ======================= 预测模型（负荷预测用 day_ahead_common.LoadForecaster，光伏预测是本版特有）=======================
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


# ======================= 全年滚动仿真 =======================
milp = dac.DayAheadMILP(price_avg)
load_forecaster = dac.LoadForecaster(load_kw_seed, alpha=LOAD_ALPHA, gamma=LOAD_GAMMA)
pv_forecaster = PVForecaster(pv_kw_seed)
net_error_tracker = dac.NetErrorTracker(T)

soc_prev_end = dac.SOC0_INIT
results = []
t_start = time.time()

for day_idx in range(n_days):
    f_load_kw = load_forecaster.predict(day_idx)
    f_pv_kw = pv_forecaster.predict(day_idx)
    f_load_e = dac.adjacent_average(f_load_kw) * DT
    f_pv_e = dac.adjacent_average(f_pv_kw) * DT
    f_net_e = f_load_e - f_pv_e

    margin = net_error_tracker.get_margin() if USE_SAFETY_MARGIN else np.zeros(T)
    f_load_e_target = np.clip(f_load_e + margin, 0.0, None)

    gv, cv, dv, socv = milp.solve(soc_prev_end, f_load_e_target, f_pv_e)

    actual_load_kw = load_kw_all[day_idx]
    actual_pv_kw = pv_kw_all[day_idx]
    actual_load_e = dac.adjacent_average(actual_load_kw) * DT
    actual_pv_e = dac.adjacent_average(actual_pv_kw) * DT
    actual_net_e = actual_load_e - actual_pv_e

    deficit_e, plan_cost, emerg_cost = dac.settle_day(gv, cv, dv, actual_load_e, actual_pv_e, price_avg)

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
print(f"安全边际: {'启用 (q=' + str(dac.SAFETY_QUANTILE) + ')' if USE_SAFETY_MARGIN else '未启用（点预测基线）'}")

# ======================= 导出 result2.xlsx 模板 =======================
try:
    if not do_export:
        raise RuntimeError("调试模式下仿真天数未覆盖导出区间，跳过导出")
    template_file = TEMPLATE_DIR / "result2.xlsx"
    out_file = RESULTS_DIR / "result2.xlsx"
    dac.export_result2_template(template_file, out_file, results, export_start_idx, export_end_idx,
                                 price_avg, n_days)
    print(f"\n[模板填充] 已写入: {out_file}")
except Exception as e:
    print(f"[提示] 未写入 result2.xlsx: {e}")
    if do_export:
        raise

# ======================= 论文素材：预测误差 + 全年费用 + 4个代表日 =======================
header_lines = [
    f"求解器: {dac.SOLVER}  (ETA_C={dac.ETA_C:.4f}, ETA_D={dac.ETA_D:.4f})",
    f"负荷预测: Holt-Winters(alpha={LOAD_ALPHA}, gamma={LOAD_GAMMA}, 周期=7)",
    f"光伏预测: 晴空包络(window={PV_WINDOW}, q={PV_QUANTILE}) x 晴空指数EWMA(beta={PV_BETA})",
    f"安全边际: {'启用 q=' + str(dac.SAFETY_QUANTILE) if USE_SAFETY_MARGIN else '未启用'}",
]
report, metrics = dac.build_summary_report(header_lines, results, dates_str, export_start_idx, export_end_idx,
                                            price_avg)
print("\n" + report)

summary_path = RESULTS_DIR / "problem2_summary.txt"
summary_path.write_text(report, encoding="utf-8")
print(f"\n[汇总] 已写入: {summary_path}")

# ======================= 可选：4个代表日的曲线图 =======================
try:
    saved = dac.plot_representative_days(results, dates_str, price_raw, RESULTS_DIR, file_prefix="problem2_plot")
    for p in saved:
        print(f"[作图] 已保存: {p}")
except Exception as e:
    print(f"[提示] 未作图（{e}）")

print("\n完成。")
