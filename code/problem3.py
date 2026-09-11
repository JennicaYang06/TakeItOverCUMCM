"""
问题3：0:00制定计划购电，6:00/12:00/18:00根据新光伏预报调整购电（全年滚动仿真）。

与问题2的区别：
  1. 光伏预测不再自己建模——附件3直接给出每天0:00/6:00/12:00/18:00发布的"未来24小时整点"
     光伏预报，问题3只需要把这些整点预报插值到10分钟粒度即可。负荷没有对应的日内预报，
     仍用 day_ahead_common.LoadForecaster（Holt-Winters）在0:00做一次预测，全天不再修正。
  2. 0:00用全天(144格)的0点光伏预报解出"计划购电量" g_plan——这是问题2同款的日前MILP
     （见 day_ahead_common.DayAheadMILP），结果既是对外报告的"计划"，也是后续调整费用的基准。
  3. 6:00/12:00/18:00各自用新预报重新决策"剩余一整天"（不是只决策接下来6小时！），但只把
     "接下来6小时=36格"锁定为最终执行结果，其余格子只是这次求解里的预览、马上会被下一次
     调整覆盖。这么做是因为最初按"只优化接下来6小时"实现时，实测储能会被过度放空：
     调整量比计划少的部分能按0.5倍价格"抵扣"（见下），只要供需平衡不等式仍满足，模型就有
     动机不计代价地多放电、少买电去薅这个折扣，而单段优化看不到"后面几段还要不要用这些电"，
     于是每段都在放空电池，反而把后面时段的缺口越掏越大。把优化范围扩到"到24:00为止"以后，
     储能递推约束贯穿整个剩余时段，模型才会意识到"现在放太多、后面会不够"，不再无脑放电。
     负荷目标仍是0点那版预测（只在真正锁定的36格上叠加报童安全边际，预览部分不加，因为
     预览部分反正会被重新决策，不需要现在就买安全垫）。
     每段调整以"上一段实际执行完的储电量"为起点，最终执行值与0点计划的差额：
       调整量 > 计划量的部分，超出部分按1.5倍价格多付；
       调整量 < 计划量的部分，少买的部分按0.5倍价格计入违约金
       （即比照单纯不调整"多付计划价"能省下0.5倍价格——细节推导见 SegmentMILP 的类注释）。
  4. 一天结束后用附件2真实值回代最终执行的 g/c/d：供给仍不够的部分按5倍价紧急购电（与问题2相同）。
  5. 总费用 = 计划购电费(price*g_plan) + 调整相关费用(各段SegmentMILP目标值之和) + 紧急购电费。
  6. 全年从2025-1-1（SOC0=6000）跑起，1月做负荷预测模型的历史预热，只导出2025-2-1~12-31。
  7. 提供 --no-adjustment 开关：只执行0点计划、不做任何日内调整，用于和"有调整"对比，
     回答题目"是否需要引入其他时刻的预报制定调整购电策略"。
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import day_ahead_common as dac

DT = dac.DT
T = dac.T

LOCK_LEN = 36                     # 每次调整只锁定接下来6小时=36格为最终结果
EPOCHS = [("6:00", 36), ("12:00", 72), ("18:00", 108)]   # (预报发布时刻, 该时刻对应的块起点)


# ======================= 读取附件3：光伏预报（每天0/6/12/18点各一条，未来24小时整点） =======================
def read_attachment3(path: Path):
    raw = pd.read_excel(path, header=0)
    date_col = raw.columns[0]
    raw[date_col] = raw[date_col].ffill()
    date_str = pd.to_datetime(raw[date_col]).dt.strftime("%Y-%m-%d")
    hourly_cols = list(raw.columns[2:26])
    assert len(hourly_cols) == 24, f"附件3 应有24个整点预报列，实际 {len(hourly_cols)}"

    lookup = {}
    for d, idx in pd.Series(date_str).groupby(date_str).groups.items():
        rows = raw.loc[idx, hourly_cols].to_numpy(dtype=float)
        assert rows.shape[0] == 4, f"{d} 应有4条预报(0/6/12/18点)，实际 {rows.shape[0]}"
        lookup[d] = {"0:00": rows[0], "6:00": rows[1], "12:00": rows[2], "18:00": rows[3]}
    return lookup


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default=dac.EXPORT_START)
    ap.add_argument("--end", default=dac.EXPORT_END)
    ap.add_argument("--debug-days", type=int, default=None, help="调试：仅仿真前N天")
    ap.add_argument("--no-adjustment", action="store_true",
                     help="关闭6/12/18点的日内调整，只执行0点计划（用于对比是否需要调整）")
    ap.add_argument("--no-safety-margin", action="store_true", help="关闭报童安全边际")
    ap.add_argument("--outdir", default=None, help="输出目录，默认 results/problem3")
    args = ap.parse_args()

    use_adjustment = not args.no_adjustment
    use_safety_margin = not args.no_safety_margin

    here = Path(__file__).resolve().parent
    root, data_dir, template_dir = dac.resolve_data_dirs(here)
    tag = "with_adjustment" if use_adjustment else "no_adjustment"
    outdir = Path(args.outdir) if args.outdir else root / "results" / "problem3" / tag
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[数据目录] {data_dir}")
    print(f"[模板目录] {template_dir}")
    print(f"[输出目录] {outdir}")
    print(f"[模式] 日内调整: {'开启' if use_adjustment else '关闭'}  安全边际: {'开启' if use_safety_margin else '关闭'}")

    pack = dac.load_price_and_actuals(data_dir)
    price_raw, price_avg = pack["price_raw"], pack["price_avg"]
    load_kw_seed = pack["load_kw_seed"]
    dates_str = pack["dates_str"]
    load_kw_all, pv_kw_all = pack["load_kw_all"], pack["pv_kw_all"]
    n_days_total = pack["n_days"]

    print("准备读取附件3光伏预报 ...")
    pv_forecast_lookup = read_attachment3(data_dir / "附件3.xlsx")
    print(f"附件3读取完成: {len(pv_forecast_lookup)} 天 x 4次预报/天")

    n_days = min(n_days_total, args.debug_days) if args.debug_days else n_days_total
    if args.debug_days:
        print(f"[调试模式] 仅仿真前 {n_days} 天")

    export_start_idx, export_end_idx, do_export = dac.compute_export_range(
        dates_str, n_days, export_start=args.start, export_end=args.end)
    if not do_export:
        print(f"[调试模式] n_days={n_days} 未覆盖导出起点，跳过导出")
    print(f"[导出区间] {args.start} (day_idx={export_start_idx}) ~ {args.end} (day_idx={export_end_idx})")

    day_milp = dac.DayAheadMILP(price_avg)
    # 每个调整时刻对应的"剩余时段"MILP：6点(剩108格)/12点(剩72格)/18点(剩36格)，尺寸不同各建一个。
    seg_milps = {start: dac.SegmentMILP(T - start) for _, start in EPOCHS}
    load_forecaster = dac.LoadForecaster(load_kw_seed)
    net_error_tracker = dac.NetErrorTracker(T)
    # 每个调整时刻各自的"锁定的那36格"净负荷预测误差滚动分位数，不能和0点(24小时视距)那条
    # 共用——视距越短预报通常越准，误差分布也越小，用同一条会把24小时视距算出的大边际
    # 错误地叠加到6小时视距的调整上（详见上面 problem3.py 顶部注释里的踩坑记录）。
    seg_error_trackers = {start: dac.NetErrorTracker(LOCK_LEN) for _, start in EPOCHS}

    soc_prev_end = dac.SOC0_INIT
    results = []
    t_start = time.time()

    for day_idx in range(n_days):
        date_str = dates_str[day_idx]

        # ---------- 0点：负荷预测 + 附件3的0点光伏预报 -> 全天计划 ----------
        f_load_kw = load_forecaster.predict(day_idx)
        f_load_e_full = dac.adjacent_average(f_load_kw) * DT

        anchor0 = pv_kw_all[day_idx - 1, -1] if day_idx > 0 else 0.0
        pv0_hourly = pv_forecast_lookup[date_str]["0:00"]
        pv0_kw = dac.interp_hourly_to_10min(anchor0, pv0_hourly, T)
        f_pv_e_full = dac.adjacent_average_with_prev(pv0_kw, anchor0) * DT

        margin_full = net_error_tracker.get_margin() if use_safety_margin else np.zeros(T)
        f_load_e_target_full = np.clip(f_load_e_full + margin_full, 0.0, None)

        g_plan, c_plan, d_plan, soc_plan = day_milp.solve(soc_prev_end, f_load_e_target_full, f_pv_e_full)

        # ---------- 6:00/12:00/18:00：每次都对"剩余一整天"重新优化，只锁定接下来36格 ----------
        g_final = g_plan.copy()
        c_final = c_plan.copy()
        d_final = d_plan.copy()
        soc_pieces = [soc_plan[0:37]]          # 0:00-6:00 直接沿用计划
        current_soc = float(soc_plan[36])
        seg_lock_pv_e = {}                     # start -> 锁定36格的光伏预测，留着结算后更新tracker

        for issue, start in EPOCHS:
            if not use_adjustment:
                soc_pieces.append(soc_plan[start + 1:start + LOCK_LEN + 1])
                current_soc = float(soc_plan[start + LOCK_LEN])
                continue

            seg_len = T - start
            n_hours = seg_len // 6
            anchor = pv_kw_all[day_idx, start - 1]
            hourly = pv_forecast_lookup[date_str][issue][:n_hours]
            pv_seg_kw = dac.interp_hourly_to_10min(anchor, hourly, seg_len)
            pv_seg_e = dac.adjacent_average_with_prev(pv_seg_kw, anchor) * DT

            margin_lock = seg_error_trackers[start].get_margin() if use_safety_margin else np.zeros(LOCK_LEN)
            load_seg_target = f_load_e_full[start:T].copy()
            load_seg_target[:LOCK_LEN] = np.clip(load_seg_target[:LOCK_LEN] + margin_lock, 0.0, None)

            g_adj, c_seg, d_seg, soc_seg, _ = seg_milps[start].solve(
                current_soc, load_seg_target, pv_seg_e, g_plan[start:T], price_avg[start:T])

            g_final[start:start + LOCK_LEN] = g_adj[:LOCK_LEN]
            c_final[start:start + LOCK_LEN] = c_seg[:LOCK_LEN]
            d_final[start:start + LOCK_LEN] = d_seg[:LOCK_LEN]
            soc_pieces.append(soc_seg[1:LOCK_LEN + 1])
            current_soc = float(soc_seg[LOCK_LEN])
            seg_lock_pv_e[start] = pv_seg_e[:LOCK_LEN]

        soc_final = np.concatenate(soc_pieces)
        up = np.clip(g_final - g_plan, 0.0, None)
        down = np.clip(g_plan - g_final, 0.0, None)
        adjustment_cost = float(np.sum(price_avg * (1.5 * up - 0.5 * down)))

        # ---------- 结算：真实负荷/光伏回代 ----------
        actual_load_kw = load_kw_all[day_idx]
        actual_pv_kw = pv_kw_all[day_idx]
        actual_load_e = dac.adjacent_average(actual_load_kw) * DT
        actual_pv_e = dac.adjacent_average(actual_pv_kw) * DT
        actual_net_e = actual_load_e - actual_pv_e

        deficit_e, _, emerg_cost = dac.settle_day(g_final, c_final, d_final, actual_load_e, actual_pv_e, price_avg)
        plan_cost = float(price_avg @ g_plan)
        total_cost = plan_cost + adjustment_cost + emerg_cost

        results.append(dict(
            day_idx=day_idx, date=date_str,
            g_plan=g_plan, g_final=g_final, gv=g_final,       # gv 别名，兼容通用画图/统计函数
            cv=c_final, dv=d_final, socv=soc_final,
            f_load_kw=f_load_kw, f_pv_kw=pv0_kw,
            actual_load_kw=actual_load_kw, actual_pv_kw=actual_pv_kw,
            deficit_e=deficit_e, plan_cost=plan_cost, adjustment_cost=adjustment_cost,
            emerg_cost=emerg_cost, total_cost=total_cost,
        ))

        f_net_e_full = f_load_e_full - f_pv_e_full
        load_forecaster.update(day_idx, actual_load_kw)
        net_error_tracker.update(f_net_e_full, actual_net_e)
        if use_adjustment:
            for issue, start in EPOCHS:
                a, b = start, start + LOCK_LEN
                f_net_e_seg = f_load_e_full[a:b] - seg_lock_pv_e[start]
                seg_error_trackers[start].update(f_net_e_seg, actual_net_e[a:b])
        soc_prev_end = float(soc_final[-1])

        if day_idx % 30 == 0 or day_idx == n_days - 1:
            print(f"[{day_idx + 1}/{n_days}] {date_str}  计划={plan_cost:8.2f}  "
                  f"调整={adjustment_cost:8.2f}  紧急={emerg_cost:8.2f}  soc_end={soc_prev_end:8.1f}")

    print(f"\n仿真完成，用时 {time.time() - t_start:.1f}s")

    # ---------- 导出 result3.xlsx ----------
    try:
        if not do_export:
            raise RuntimeError("调试模式下仿真天数未覆盖导出区间，跳过导出")
        template_file = template_dir / "result3.xlsx"
        out_file = outdir / "result3.xlsx"
        dac.export_result3_template(template_file, out_file, results, export_start_idx, export_end_idx,
                                     price_avg, n_days)
        print(f"\n[模板填充] 已写入: {out_file}")
    except Exception as e:
        print(f"[提示] 未写入 result3.xlsx: {e}")
        if do_export:
            raise

    # ---------- 汇总报告 ----------
    exp_slice = results[export_start_idx:export_end_idx + 1]
    f_load_mat = np.array([r["f_load_kw"] for r in exp_slice])
    a_load_mat = np.array([r["actual_load_kw"] for r in exp_slice])
    f_pv_mat = np.array([r["f_pv_kw"] for r in exp_slice])
    a_pv_mat = np.array([r["actual_pv_kw"] for r in exp_slice])
    load_mae, load_rmse, load_mape = dac.error_metrics(f_load_mat, a_load_mat)
    pv_mae, pv_rmse, pv_mape = dac.error_metrics(f_pv_mat, a_pv_mat, mape_thresh=50.0)

    total_plan = sum(r["plan_cost"] for r in exp_slice)
    total_adj = sum(r["adjustment_cost"] for r in exp_slice)
    total_emerg = sum(r["emerg_cost"] for r in exp_slice)
    total_all = total_plan + total_adj + total_emerg
    emerg_days = sum(1 for r in exp_slice if r["emerg_cost"] > 1e-6)
    emerg_blocks = sum(int(np.sum(r["deficit_e"] > 1e-6)) for r in exp_slice)

    lines = []
    lines.append("=" * 70)
    lines.append("问题3  结果汇总")
    lines.append("=" * 70)
    lines.append(f"求解器: {dac.SOLVER}  (ETA_C={dac.ETA_C:.4f}, ETA_D={dac.ETA_D:.4f})")
    lines.append("光伏预测: 直接使用附件3官方预报(0/6/12/18点发布，整点值线性插值到10分钟)")
    lines.append("负荷预测: Holt-Winters(0点做一次，全天不修正，附件3不提供负荷预报)")
    lines.append(f"日内调整: {'开启（6/12/18点各调整接下来6小时）' if use_adjustment else '关闭（只执行0点计划）'}")
    lines.append(f"安全边际: {'启用 q=' + str(dac.SAFETY_QUANTILE) if use_safety_margin else '未启用'}")
    lines.append("")
    lines.append(f"预测误差（{args.start}~{args.end}，0点预报口径）")
    lines.append("-" * 70)
    lines.append(f"  负荷  MAE={load_mae:8.2f} kW   RMSE={load_rmse:8.2f} kW   MAPE={load_mape:6.2f}%")
    lines.append(f"  光伏  MAE={pv_mae:8.2f} kW   RMSE={pv_rmse:8.2f} kW   MAPE(出力>50kW)={pv_mape:6.2f}%")
    lines.append("")
    lines.append(f"全年费用（{args.start}~{args.end}，{len(exp_slice)}天）")
    lines.append("-" * 70)
    lines.append(f"  计划购电费合计   = {total_plan:12.2f} 元")
    lines.append(f"  调整相关费用合计 = {total_adj:12.2f} 元  （正=多付1.5倍溢价，负=少买省下0.5倍价）")
    lines.append(f"  紧急购电费合计   = {total_emerg:12.2f} 元")
    lines.append(f"  总费用           = {total_all:12.2f} 元")
    lines.append(f"  发生紧急购电天数 = {emerg_days} / {len(exp_slice)}")
    lines.append(f"  发生紧急购电时段数 = {emerg_blocks}")
    lines.append("")

    for rd in dac.REP_DATES:
        idx = int(np.where(dates_str == rd)[0][0])
        if idx >= len(results):
            continue
        r = results[idx]
        g_plan, g_final, cv, dv, socv, deficit_e = r["g_plan"], r["g_final"], r["cv"], r["dv"], r["socv"], r["deficit_e"]
        lines.append("=" * 70)
        lines.append(f"代表日 {rd}")
        lines.append("=" * 70)
        lines.append("表1a  计划购电量（0点制定，指定时间段，kWh）")
        for name, t in dac.TBL1:
            lines.append(f"  {name:<14s} {g_plan[t]:12.4f}")
        lines.append(f"  {'全天计划购电量':<14s} {g_plan.sum():12.4f}")
        lines.append(f"  {'全天计划购电费':<14s} {float(price_avg @ g_plan):12.2f}")
        lines.append("")
        lines.append("表1b  调整购电量（最终执行，指定时间段，kWh）")
        for name, t in dac.TBL1:
            lines.append(f"  {name:<14s} {g_final[t]:12.4f}")
        lines.append(f"  {'全天调整购电量':<14s} {g_final.sum():12.4f}")
        lines.append(f"  {'全天调整购电费':<14s} {float(price_avg @ g_final):12.2f}")
        lines.append(f"  {'当日调整相关费用':<14s} {r['adjustment_cost']:12.2f}")
        lines.append("")
        lines.append("表2  储能设备在指定时间段的充放电量 (kWh，最终执行)")
        lines.append(f"  {'时间段':<14s} {'充电量':>12s} {'放电量':>12s}")
        for name, a, b in dac.TBL2:
            lines.append(f"  {name:<14s} {cv[a:b].sum():12.4f} {dv[a:b].sum():12.4f}")
        lines.append(f"  0:00 储电量 (kWh)  = {socv[0]:.4f}")
        lines.append(f"  24:00 储电量 (kWh) = {socv[-1]:.4f}")
        lines.append("")
        lines.append("表3  紧急购电")
        segs = dac.merge_deficit_segments(deficit_e)
        if not segs:
            lines.append("  (无紧急购电)")
        else:
            for s, e, val in segs:
                seg_label = dac.time_range_str(s)[:5] + "-" + dac.time_range_str(e - 1)[6:]
                lines.append(f"  {seg_label:<14s} {val:12.4f}")
        lines.append("")

    report = "\n".join(lines)
    print("\n" + report)

    summary_path = outdir / "problem3_summary.txt"
    summary_path.write_text(report, encoding="utf-8")
    print(f"\n[汇总] 已写入: {summary_path}")

    # ---------- 代表日画图 ----------
    try:
        saved = dac.plot_representative_days(results, dates_str, price_raw, outdir,
                                              title_prefix="问题3(计划vs调整)", file_prefix="problem3_plot")
        for p in saved:
            print(f"[作图] 已保存: {p}")
    except Exception as e:
        print(f"[提示] 未作图（{e}）")

    print("\n完成。")


if __name__ == "__main__":
    main()
