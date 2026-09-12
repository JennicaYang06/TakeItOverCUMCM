# -*- coding: utf-8 -*-
"""
问题3(LightGBM负荷预测, --epochs 4)敏感性分析：报童分位数q、安全边际窗口W、
可用容量E_usable、往返效率η，各扫5个点（其余参数固定在题目/项目默认值）。

**不修改任何现有文件**：
- q / W 直接用 day_ahead_common.NetErrorTracker 构造函数本来就有的 window/quantile
  参数传入，不涉及任何改动。
- E_usable / η 没有现成的构造参数，靠运行时改写 day_ahead_common 模块级常量
  （SOC_MIN/SOC_MAX/SOC0_INIT/RT_EFF/ETA_C/ETA_D）实现：Python的名字查找是运行时的，
  DayAheadMILP/SegmentMILP 的 __init__ 在被调用的那一刻才去查这些全局变量的值，所以
  只要在 `dac.DayAheadMILP(...)` 之前把 `dac.SOC_MAX = 新值` 这样改了，新建的实例
  用的就是新值——不需要碰 day_ahead_common.py 源文件一个字。每次调用完都会用
  reset_battery_params() 还原，避免状态串到下一个实验。

负荷(LightGBM)、光伏(附件3插值)预测和这4个参数完全无关，只在最开始算一次、缓存起来
（build_cache），每个扫描点只重新解MILP+结算，不重新训练LightGBM——这是整个脚本能在
十几分钟内跑完 17 组全年仿真（4个参数×5点，基准点共用，17=1+4x4）的关键。

用法：
    python code/sensitivity_analysis.py                # 全年，4个参数各扫5点
    python code/sensitivity_analysis.py --debug-days 40 # 调试：先用少量天数验证脚本本身没问题
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import day_ahead_common as dac
from problem2_lightgbm import read_attachment2, add_features, add_lags, train_predict_one_day
from problem3_lightgbm import read_attachment3, EPOCH_CONFIGS

DT = dac.DT
T = dac.T
MIN_TRAIN_ROWS = 100

# ===================== 基准值（等于 day_ahead_common.py 当前的默认值） =====================
BASE_SOC_MIN = dac.SOC_MIN
BASE_SOC_MAX = dac.SOC_MAX
BASE_SOC0 = dac.SOC0_INIT
BASE_RT_EFF = dac.RT_EFF
BASE_P_MAX = dac.P_MAX
BASE_E_STEP_MAX = dac.E_STEP_MAX

BASE_Q = dac.SAFETY_QUANTILE     # 0.8
BASE_W = dac.SAFETY_WINDOW       # 60

# ===================== 5点扫描范围（每个都包含基准值，方便和已知结果对齐校验） =====================
SWEEP_Q = [0.6, 0.7, 0.8, 0.9, 0.95]
SWEEP_W = [15, 30, 60, 90, 120]
SWEEP_CAP_SCALE = [0.5, 0.75, 1.0, 1.5, 2.0]      # 可用容量按比例缩放（min/max/SOC0同比例）
SWEEP_ETA = [0.80, 0.85, 0.90, 0.95, 0.99]


def set_battery_params(capacity_scale=1.0, eta=None):
    """运行时改写 day_ahead_common 模块级常量。跑完必须调用 reset_battery_params() 还原。"""
    dac.SOC_MIN = BASE_SOC_MIN * capacity_scale
    dac.SOC_MAX = BASE_SOC_MAX * capacity_scale
    dac.SOC0_INIT = BASE_SOC0 * capacity_scale
    if eta is not None:
        dac.RT_EFF = eta
        dac.ETA_C = float(np.sqrt(eta))
        dac.ETA_D = float(np.sqrt(eta))


def reset_battery_params():
    dac.SOC_MIN = BASE_SOC_MIN
    dac.SOC_MAX = BASE_SOC_MAX
    dac.SOC0_INIT = BASE_SOC0
    dac.RT_EFF = BASE_RT_EFF
    dac.ETA_C = float(np.sqrt(BASE_RT_EFF))
    dac.ETA_D = float(np.sqrt(BASE_RT_EFF))
    dac.P_MAX = BASE_P_MAX
    dac.E_STEP_MAX = BASE_E_STEP_MAX


# ======================= 预计算：负荷/光伏预测与4个待扫参数无关，只算一次 =======================
def build_cache(dates_str, all_dates, load_kw_seed, load_kw_all, pv_kw_all, data,
                 pv_forecast_lookup, EPOCHS, n_days):
    cache = []
    n_fallback = 0
    for day_idx in range(n_days):
        day = all_dates[day_idx]
        date_str = dates_str[day_idx]

        f_load_kw = train_predict_one_day(data, day, "load", MIN_TRAIN_ROWS)
        if f_load_kw is None:
            f_load_kw = (load_kw_all[day_idx - 1] if day_idx > 0 else load_kw_seed).copy()
            n_fallback += 1
        f_load_e_full = dac.adjacent_average(f_load_kw) * DT

        anchor0 = pv_kw_all[day_idx - 1, -1] if day_idx > 0 else 0.0
        pv0_hourly = pv_forecast_lookup[date_str]["0:00"]
        pv0_kw = dac.interp_hourly_to_10min(anchor0, pv0_hourly, T)
        f_pv_e_full = dac.adjacent_average_with_prev(pv0_kw, anchor0) * DT

        actual_load_kw = load_kw_all[day_idx]
        actual_pv_kw = pv_kw_all[day_idx]
        actual_load_e = dac.adjacent_average(actual_load_kw) * DT
        actual_pv_e = dac.adjacent_average(actual_pv_kw) * DT

        seg_pv_e = {}
        for issue, start in EPOCHS:
            seg_len = T - start
            n_hours = seg_len // 6
            anchor = pv_kw_all[day_idx, start - 1]
            hourly = pv_forecast_lookup[date_str][issue][:n_hours]
            pv_seg_kw = dac.interp_hourly_to_10min(anchor, hourly, seg_len)
            seg_pv_e[start] = dac.adjacent_average_with_prev(pv_seg_kw, anchor) * DT

        cache.append(dict(
            f_load_e_full=f_load_e_full, f_pv_e_full=f_pv_e_full,
            actual_load_e=actual_load_e, actual_pv_e=actual_pv_e,
            seg_pv_e=seg_pv_e,
        ))
        if day_idx % 60 == 0 or day_idx == n_days - 1:
            print(f"  [缓存预测 {day_idx + 1}/{n_days}]")
    return cache, n_fallback


# ======================= 单个参数组合：只重新解MILP+结算，复用缓存的预测 =======================
def run_config(q, W, capacity_scale, eta, cache, price_avg, EPOCHS, LOCK_LEN, n_days,
               export_start_idx, export_end_idx):
    set_battery_params(capacity_scale=capacity_scale, eta=eta)
    try:
        day_milp = dac.DayAheadMILP(price_avg)
        seg_milps = {start: dac.SegmentMILP(T - start) for _, start in EPOCHS}
        net_error_tracker = dac.NetErrorTracker(T, window=W, quantile=q)
        seg_error_trackers = {start: dac.NetErrorTracker(LOCK_LEN, window=W, quantile=q) for _, start in EPOCHS}

        first_boundary = EPOCHS[0][1] if EPOCHS else T
        soc_prev_end = dac.SOC0_INIT
        day_results = []

        for day_idx in range(n_days):
            c = cache[day_idx]
            f_load_e_full, f_pv_e_full = c["f_load_e_full"], c["f_pv_e_full"]

            margin_full = net_error_tracker.get_margin()
            f_load_e_target_full = np.clip(f_load_e_full + margin_full, 0.0, None)
            g_plan, c_plan, d_plan, soc_plan = day_milp.solve(soc_prev_end, f_load_e_target_full, f_pv_e_full)

            g_final, c_final, d_final = g_plan.copy(), c_plan.copy(), d_plan.copy()
            soc_pieces = [soc_plan[0:first_boundary + 1]]
            current_soc = float(soc_plan[first_boundary])

            for issue, start in EPOCHS:
                pv_seg_e = c["seg_pv_e"][start]
                margin_lock = seg_error_trackers[start].get_margin()
                load_seg_target = f_load_e_full[start:T].copy()
                load_seg_target[:LOCK_LEN] = np.clip(load_seg_target[:LOCK_LEN] + margin_lock, 0.0, None)

                g_adj, c_seg, d_seg, soc_seg, _ = seg_milps[start].solve(
                    current_soc, load_seg_target, pv_seg_e, g_plan[start:T], price_avg[start:T])

                g_final[start:start + LOCK_LEN] = g_adj[:LOCK_LEN]
                c_final[start:start + LOCK_LEN] = c_seg[:LOCK_LEN]
                d_final[start:start + LOCK_LEN] = d_seg[:LOCK_LEN]
                soc_pieces.append(soc_seg[1:LOCK_LEN + 1])
                current_soc = float(soc_seg[LOCK_LEN])

            soc_final = np.concatenate(soc_pieces)
            up = np.clip(g_final - g_plan, 0.0, None)
            down = np.clip(g_plan - g_final, 0.0, None)
            adjustment_cost = float(np.sum(price_avg * (1.5 * up - 0.5 * down)))

            actual_load_e, actual_pv_e = c["actual_load_e"], c["actual_pv_e"]
            actual_net_e = actual_load_e - actual_pv_e
            deficit_e, _, emerg_cost = dac.settle_day(g_final, c_final, d_final, actual_load_e, actual_pv_e, price_avg)
            plan_cost = float(price_avg @ g_plan)

            day_results.append(dict(plan_cost=plan_cost, adjustment_cost=adjustment_cost,
                                     emerg_cost=emerg_cost, deficit_e=deficit_e))

            f_net_e_full = f_load_e_full - f_pv_e_full
            net_error_tracker.update(f_net_e_full, actual_net_e)
            for issue, start in EPOCHS:
                a, b = start, start + LOCK_LEN
                f_net_e_seg = f_load_e_full[a:b] - c["seg_pv_e"][start][:LOCK_LEN]
                seg_error_trackers[start].update(f_net_e_seg, actual_net_e[a:b])
            soc_prev_end = float(soc_final[-1])
    finally:
        reset_battery_params()

    exp = day_results[export_start_idx:export_end_idx + 1]
    total_plan = sum(r["plan_cost"] for r in exp)
    total_adj = sum(r["adjustment_cost"] for r in exp)
    total_emerg = sum(r["emerg_cost"] for r in exp)
    emerg_days = sum(1 for r in exp if r["emerg_cost"] > 1e-6)
    emerg_blocks = sum(int(np.sum(r["deficit_e"] > 1e-6)) for r in exp)
    return dict(plan=total_plan, adjustment=total_adj, emergency=total_emerg,
                total=total_plan + total_adj + total_emerg,
                emerg_days=emerg_days, emerg_blocks=emerg_blocks, n_days=len(exp))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--debug-days", type=int, default=None, help="调试：仅仿真前N天")
    ap.add_argument("--outdir", default=None, help="输出目录，默认 results/sensitivity/problem3_epochs4")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    root, data_dir, template_dir = dac.resolve_data_dirs(here)
    outdir = Path(args.outdir) if args.outdir else root / "results" / "sensitivity" / "problem3_epochs4"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[数据目录] {data_dir}")
    print(f"[输出目录] {outdir}")

    pack = dac.load_price_and_actuals(data_dir)
    price_avg = pack["price_avg"]
    load_kw_seed = pack["load_kw_seed"]
    dates_str = pack["dates_str"]
    load_kw_all, pv_kw_all = pack["load_kw_all"], pack["pv_kw_all"]
    n_days_total = pack["n_days"]
    all_dates = pd.to_datetime(dates_str)

    print("准备读取附件2构造负荷LightGBM特征表 ...")
    data = read_attachment2(data_dir / "附件2.xlsx")
    data = add_features(data)
    data = add_lags(data, "load")

    print("准备读取附件3光伏预报 ...")
    pv_forecast_lookup = read_attachment3(data_dir / "附件3.xlsx")

    cfg = EPOCH_CONFIGS["4"]
    EPOCHS, LOCK_LEN = cfg["epochs"], cfg["lock_len"]

    n_days = min(n_days_total, args.debug_days) if args.debug_days else n_days_total
    if args.debug_days:
        print(f"[调试模式] 仅仿真前 {n_days} 天")
    export_start_idx, export_end_idx, do_export = dac.compute_export_range(dates_str, n_days)
    print(f"[导出/统计区间] day_idx={export_start_idx}~{export_end_idx}")

    print("\n预计算负荷/光伏预测（与4个敏感性参数无关，全程只算这一次）...")
    t0 = time.time()
    cache, n_fallback = build_cache(dates_str, all_dates, load_kw_seed, load_kw_all, pv_kw_all,
                                     data, pv_forecast_lookup, EPOCHS, n_days)
    print(f"预计算完成，用时 {time.time() - t0:.1f}s，冷启动回退 {n_fallback} 天")

    def run(q, W, cap, eta):
        return run_config(q, W, cap, eta, cache, price_avg, EPOCHS, LOCK_LEN, n_days,
                           export_start_idx, export_end_idx)

    print("\n运行基准点 (q=%.2f, W=%d, capacity_scale=1.0, eta=%.2f) 作为对照 ..." % (BASE_Q, BASE_W, BASE_RT_EFF))
    t0 = time.time()
    baseline = run(BASE_Q, BASE_W, 1.0, BASE_RT_EFF)
    print(f"基准点用时 {time.time() - t0:.1f}s -> 总费用 = {baseline['total']:.2f} 元")
    if not args.debug_days:
        known = 13798212.44
        diff = abs(baseline["total"] - known) / known * 100
        print(f"[核对] 与此前 problem3_lightgbm.py --epochs 4 全年已知结果 {known:.2f} 元相差 {diff:.4f}%"
              + ("（一致，缓存复现逻辑正确）" if diff < 0.5 else "  ！！差异较大，请检查脚本逻辑！！"))

    sweeps = {
        "q": (SWEEP_Q, lambda v: run(v, BASE_W, 1.0, BASE_RT_EFF), BASE_Q),
        "W": (SWEEP_W, lambda v: run(BASE_Q, v, 1.0, BASE_RT_EFF), BASE_W),
        "capacity": (SWEEP_CAP_SCALE, lambda v: run(BASE_Q, BASE_W, v, BASE_RT_EFF), 1.0),
        "eta": (SWEEP_ETA, lambda v: run(BASE_Q, BASE_W, 1.0, v), BASE_RT_EFF),
    }

    all_results = {}
    for name, (values, runner, base_val) in sweeps.items():
        print(f"\n===== 扫描参数: {name}  (基准值={base_val}) =====")
        rows = []
        for v in values:
            if abs(v - base_val) < 1e-9:
                r = baseline
                print(f"  {name}={v}  (=基准点，复用已算结果)  总费用={r['total']:.2f}")
            else:
                t0 = time.time()
                r = runner(v)
                print(f"  {name}={v}  用时{time.time() - t0:5.1f}s  总费用={r['total']:.2f}  "
                      f"(计划={r['plan']:.2f} 调整={r['adjustment']:.2f} 紧急={r['emergency']:.2f})")
            rows.append((v, r))
        all_results[name] = rows

    # ---------- 导出 CSV + 汇总文本 ----------
    summary_lines = []
    summary_lines.append("=" * 78)
    summary_lines.append("问题3(LightGBM, --epochs 4) 敏感性分析汇总")
    summary_lines.append("=" * 78)
    summary_lines.append(f"基准点: q={BASE_Q}, W={BASE_W}天, 可用容量缩放=1.0(={BASE_SOC_MAX - BASE_SOC_MIN:.0f}kWh), "
                          f"η={BASE_RT_EFF}")
    summary_lines.append(f"基准点总费用 = {baseline['total']:.2f} 元"
                          + ("" if args.debug_days else f"（与已知全年结果 13798212.44 元相差 {diff:.4f}%）"))
    summary_lines.append(f"统计区间: day_idx {export_start_idx}~{export_end_idx}，共{baseline['n_days']}天")
    summary_lines.append("")

    param_label = {"q": "报童分位数q", "W": "安全边际窗口W(天)", "capacity": "可用容量缩放比例ρ", "eta": "往返效率η"}
    for name, rows in all_results.items():
        summary_lines.append("-" * 78)
        summary_lines.append(f"[{param_label[name]}]")
        summary_lines.append(f"{'参数值':>10s} {'计划购电费':>14s} {'调整相关费用':>14s} {'紧急购电费':>12s} "
                              f"{'总费用':>14s} {'紧急天数':>8s} {'紧急时段数':>10s}")
        for v, r in rows:
            summary_lines.append(f"{v:>10.4g} {r['plan']:>14.2f} {r['adjustment']:>14.2f} {r['emergency']:>12.2f} "
                                  f"{r['total']:>14.2f} {r['emerg_days']:>8d} {r['emerg_blocks']:>10d}")
        summary_lines.append("")

        csv_path = outdir / f"sensitivity_{name}.csv"
        df = pd.DataFrame([
            dict(param_value=v, plan_cost=r["plan"], adjustment_cost=r["adjustment"],
                 emergency_cost=r["emergency"], total_cost=r["total"],
                 emergency_days=r["emerg_days"], emergency_blocks=r["emerg_blocks"])
            for v, r in rows
        ])
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"[CSV已保存] {csv_path}")

    report = "\n".join(summary_lines)
    print("\n" + report)
    summary_path = outdir / "sensitivity_summary.txt"
    summary_path.write_text(report, encoding="utf-8")
    print(f"\n[汇总已保存] {summary_path}")

    # ---------- 画图 ----------
    try:
        import matplotlib.pyplot as plt
        try:
            plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
        except Exception:
            pass

        xlabel = {"q": "报童分位数 q", "W": "安全边际窗口 W (天)",
                  "capacity": "可用容量缩放比例 ρ", "eta": "往返效率 η"}
        for name, rows in all_results.items():
            xs = [v for v, _ in rows]
            plan = [r["plan"] for _, r in rows]
            adj = [r["adjustment"] for _, r in rows]
            emerg = [r["emergency"] for _, r in rows]
            total = [r["total"] for _, r in rows]

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(xs, total, marker="o", color="tab:red", label="总费用", linewidth=2)
            ax.plot(xs, plan, marker="s", color="tab:blue", label="计划购电费", alpha=0.7)
            ax.plot(xs, emerg, marker="^", color="tab:orange", label="紧急购电费", alpha=0.7)
            ax.plot(xs, adj, marker="v", color="tab:green", label="调整相关费用", alpha=0.7)
            ax.axvline(sweeps[name][2], ls="--", color="gray", alpha=0.6, label="基准值")
            ax.set_xlabel(xlabel[name])
            ax.set_ylabel("费用 (元)")
            ax.set_title(f"敏感性分析：{param_label[name]}")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            out_png = outdir / f"sensitivity_{name}.png"
            fig.savefig(out_png, dpi=150)
            plt.close(fig)
            print(f"[作图已保存] {out_png}")
    except Exception as e:
        print(f"[提示] 未作图（{e}）")

    print("\n完成。")


if __name__ == "__main__":
    main()
