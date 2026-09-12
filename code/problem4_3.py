# -*- coding: utf-8 -*-
"""
问题4（对应问题3部分）：波动电价下的"0点计划+日内调整"购电策略——LightGBM负荷预测，
光伏用附件3官方预报，日内调整逻辑与 problem3_lightgbm-0点6点12点18点.py 一致，
只是电价换成附件4的"每天一条、365天各不相同"波动电价。

和 problem4_2.py 同样的理由：day_ahead_common.py 里 export_result3_template /
build_summary_report 假设全年一条价格，不能直接拿来处理附件4，这里自己写导出/汇总逻辑
（复用 problem4_2.py 里已经写好的按天电价版购电量表填充函数，充放电/紧急购电两张表仍是
直接复用 day_ahead_common 的价格无关函数）。day_ahead_common.py 本身不做任何改动。

--epochs 4（默认）：6:00/12:00/18:00 各调整一次，每次对"剩余一整天"联合重优化、只锁定
接下来6小时——这是之前问题3做"是否需要引入其他时刻预报"对比时验证过、能避免储能被过度
放空的正确做法（只优化局部窗口会让模型为了拿0.5倍价格的"少买折扣"不计后果地放电，
把优化窗口扩到当天24:00为止、只锁定近期结果，才会考虑到后面几段还要不要用这些电）。
--epochs 2：只在12:00调整一次（对剩余12小时联合重优化，全部锁定），对应"少了6点、18点
两次预报"的场景，用于和--epochs 4对比"是否有必要引入更多次预报"。
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import day_ahead_common as dac
from problem2_lightgbm import add_features, add_lags, train_predict_one_day
from problem4_2 import read_attachment4_price, export_purchase_sheet_perday

DT = dac.DT
T = dac.T

EPOCH_CONFIGS = {
    "4": {"epochs": [("6:00", 36), ("12:00", 72), ("18:00", 108)], "lock_len": 36},
    "2": {"epochs": [("12:00", 72)], "lock_len": 72},
}


# ======================= 附件2 长表读取（这里只需要load列，但和problem2_lightgbm共用同一份读取函数，
# 顺带把pv也读进来不影响正确性，反正后面只用 add_lags(data,"load")） =======================
def read_attachment2(path: Path):
    xl = pd.ExcelFile(path)

    def long(sheet):
        df = xl.parse(sheet)
        df = df.rename(columns={df.columns[0]: "date"})
        df["date"] = pd.to_datetime(df["date"])
        return df.melt(id_vars="date", var_name="time", value_name="value")

    load = long("小区负载").rename(columns={"value": "load"})
    pv = long("光伏发电实际功率").rename(columns={"value": "pv"})
    data = load.merge(pv, on=["date", "time"], how="inner")
    data["t"] = data.groupby("date").cumcount()
    data = data.sort_values(["date", "t"]).reset_index(drop=True)
    return data


# ======================= 附件3：光伏预报（每天0/6/12/18点各一条，未来24小时整点） =======================
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


def export_result4_3_template(template_file, out_file, results, export_start_idx, export_end_idx, n_days):
    from openpyxl import load_workbook
    wb = load_workbook(template_file)
    export_purchase_sheet_perday(wb["计划购电量"], results, "g_plan", export_start_idx, export_end_idx, n_days)
    export_purchase_sheet_perday(wb["调整购电量"], results, "g_final", export_start_idx, export_end_idx, n_days)
    dac._fill_charge_sheet(wb["充放电量"], results, export_start_idx, export_end_idx)
    dac._fill_emergency_sheet(wb["紧急购电量"], results, export_start_idx, export_end_idx)
    wb.save(out_file)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default=dac.EXPORT_START)
    ap.add_argument("--end", default=dac.EXPORT_END)
    ap.add_argument("--debug-days", type=int, default=None, help="调试：仅仿真前N天")
    ap.add_argument("--epochs", choices=["2", "4"], default="4",
                     help="调整时刻个数：4=6/12/18点各调整一次；2=只在12点调整一次")
    ap.add_argument("--no-adjustment", action="store_true", help="关闭日内调整，只执行0点计划")
    ap.add_argument("--no-safety-margin", action="store_true", help="关闭报童安全边际")
    ap.add_argument("--min-train-rows", type=int, default=100)
    ap.add_argument("--outdir", default=None, help="输出目录，默认 results/problem4/problem3_volatile/<tag>")
    args = ap.parse_args()

    use_adjustment = not args.no_adjustment
    use_safety_margin = not args.no_safety_margin
    cfg = EPOCH_CONFIGS[args.epochs]
    EPOCHS, LOCK_LEN = cfg["epochs"], cfg["lock_len"]

    here = Path(__file__).resolve().parent
    root, data_dir, template_dir = dac.resolve_data_dirs(here)
    if use_adjustment:
        tag = f"{args.epochs}epoch_adjustment"
    else:
        tag = "no_adjustment"
    outdir = Path(args.outdir) if args.outdir else root / "results" / "problem4" / "problem3_volatile" / tag
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[数据目录] {data_dir}")
    print(f"[模板目录] {template_dir}")
    print(f"[输出目录] {outdir}")
    print(f"[模式] 调整时刻数: {args.epochs}  日内调整: {'开启' if use_adjustment else '关闭'}  "
          f"安全边际: {'开启' if use_safety_margin else '关闭'}")

    pack = dac.load_price_and_actuals(data_dir)
    load_kw_seed = pack["load_kw_seed"]
    dates_str = pack["dates_str"]
    load_kw_all, pv_kw_all = pack["load_kw_all"], pack["pv_kw_all"]
    n_days_total = pack["n_days"]
    all_dates = pd.to_datetime(dates_str)

    print("准备读取附件4波动电价 ...")
    price_dates_str, price_kw_all = read_attachment4_price(data_dir / "附件4.xlsx")
    assert np.array_equal(price_dates_str, dates_str), "附件4 与附件2 的日期顺序不一致"
    print(f"附件4读取完成: {len(price_dates_str)} 天")

    print("准备读取附件2构造负荷LightGBM特征表 ...")
    data = read_attachment2(data_dir / "附件2.xlsx")
    data = add_features(data)
    data = add_lags(data, "load")
    print(f"特征表构造完成: {len(data)} 行")

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

    seg_milps = {start: dac.SegmentMILP(T - start) for _, start in EPOCHS}
    net_error_tracker = dac.NetErrorTracker(T)
    seg_error_trackers = {start: dac.NetErrorTracker(LOCK_LEN) for _, start in EPOCHS}

    soc_prev_end = dac.SOC0_INIT
    results = []
    n_fallback = 0
    t_start = time.time()

    for day_idx in range(n_days):
        date_str = dates_str[day_idx]
        day = all_dates[day_idx]

        price_raw_day = price_kw_all[day_idx]
        price_avg_day = dac.adjacent_average(price_raw_day)
        day_milp = dac.DayAheadMILP(price_avg_day)   # 电价天天不同，只能每天重新构建

        # ---------- 0点：LightGBM负荷预测 + 附件3的0点光伏预报 -> 全天计划 ----------
        f_load_kw = train_predict_one_day(data, day, "load", args.min_train_rows)
        if f_load_kw is None:
            f_load_kw = (load_kw_all[day_idx - 1] if day_idx > 0 else load_kw_seed).copy()
            n_fallback += 1
        f_load_e_full = dac.adjacent_average(f_load_kw) * DT

        anchor0 = pv_kw_all[day_idx - 1, -1] if day_idx > 0 else 0.0
        pv0_hourly = pv_forecast_lookup[date_str]["0:00"]
        pv0_kw = dac.interp_hourly_to_10min(anchor0, pv0_hourly, T)
        f_pv_e_full = dac.adjacent_average_with_prev(pv0_kw, anchor0) * DT

        margin_full = net_error_tracker.get_margin() if use_safety_margin else np.zeros(T)
        f_load_e_target_full = np.clip(f_load_e_full + margin_full, 0.0, None)

        g_plan, c_plan, d_plan, soc_plan = day_milp.solve(soc_prev_end, f_load_e_target_full, f_pv_e_full)

        # ---------- 调整：对"剩余一整天"联合重优化，只锁定接下来LOCK_LEN格 ----------
        g_final = g_plan.copy()
        c_final = c_plan.copy()
        d_final = d_plan.copy()
        soc_pieces = [soc_plan[0:EPOCHS[0][1] + 1]]
        current_soc = float(soc_plan[EPOCHS[0][1]])
        seg_lock_pv_e = {}

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
                current_soc, load_seg_target, pv_seg_e, g_plan[start:T], price_avg_day[start:T])

            g_final[start:start + LOCK_LEN] = g_adj[:LOCK_LEN]
            c_final[start:start + LOCK_LEN] = c_seg[:LOCK_LEN]
            d_final[start:start + LOCK_LEN] = d_seg[:LOCK_LEN]
            soc_pieces.append(soc_seg[1:LOCK_LEN + 1])
            current_soc = float(soc_seg[LOCK_LEN])
            seg_lock_pv_e[start] = pv_seg_e[:LOCK_LEN]

        soc_final = np.concatenate(soc_pieces)
        up = np.clip(g_final - g_plan, 0.0, None)
        down = np.clip(g_plan - g_final, 0.0, None)
        adjustment_cost = float(np.sum(price_avg_day * (1.5 * up - 0.5 * down)))

        # ---------- 结算：真实负荷/光伏回代 ----------
        actual_load_kw = load_kw_all[day_idx]
        actual_pv_kw = pv_kw_all[day_idx]
        actual_load_e = dac.adjacent_average(actual_load_kw) * DT
        actual_pv_e = dac.adjacent_average(actual_pv_kw) * DT
        actual_net_e = actual_load_e - actual_pv_e

        deficit_e, _, emerg_cost = dac.settle_day(g_final, c_final, d_final, actual_load_e, actual_pv_e, price_avg_day)
        plan_cost = float(price_avg_day @ g_plan)
        total_cost = plan_cost + adjustment_cost + emerg_cost

        results.append(dict(
            day_idx=day_idx, date=date_str,
            g_plan=g_plan, g_final=g_final, gv=g_final,
            cv=c_final, dv=d_final, socv=soc_final,
            price_raw=price_raw_day, price_avg=price_avg_day,
            f_load_kw=f_load_kw, f_pv_kw=pv0_kw,
            actual_load_kw=actual_load_kw, actual_pv_kw=actual_pv_kw,
            deficit_e=deficit_e, plan_cost=plan_cost, adjustment_cost=adjustment_cost,
            emerg_cost=emerg_cost, total_cost=total_cost,
        ))

        f_net_e_full = f_load_e_full - f_pv_e_full
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
    print(f"负荷LightGBM冷启动回退天数: {n_fallback}")

    # ---------- 导出 result4-3.xlsx ----------
    try:
        if not do_export:
            raise RuntimeError("调试模式下仿真天数未覆盖导出区间，跳过导出")
        template_file = template_dir / "result4-3.xlsx"
        out_file = outdir / "result4-3.xlsx"
        export_result4_3_template(template_file, out_file, results, export_start_idx, export_end_idx, n_days)
        print(f"\n[模板填充] 已写入: {out_file}")
    except Exception as e:
        print(f"[提示] 未写入 result4-3.xlsx: {e}")
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
    lines.append("问题4(对应问题3，波动电价)  结果汇总")
    lines.append("=" * 70)
    lines.append(f"求解器: {dac.SOLVER}  (ETA_C={dac.ETA_C:.4f}, ETA_D={dac.ETA_D:.4f})")
    lines.append("电价: 附件4波动电价（每天一条144点曲线，逐日不同）")
    lines.append("光伏预测: 直接使用附件3官方预报，整点值线性插值到10分钟")
    lines.append("负荷预测: LightGBM 逐日滚动重训")
    if use_adjustment:
        issue_desc = "、".join(issue for issue, _ in EPOCHS)
        lines.append(f"日内调整: 开启（{issue_desc}联合剩余日重优化，只锁定接下来{LOCK_LEN * 10}分钟）")
    else:
        lines.append("日内调整: 关闭（只执行0点计划）")
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
        price_avg_day = r["price_avg"]
        lines.append("=" * 70)
        lines.append(f"代表日 {rd}")
        lines.append("=" * 70)
        lines.append("表1a  计划购电量（0点制定，指定时间段，kWh）")
        for name, t in dac.TBL1:
            lines.append(f"  {name:<14s} {g_plan[t]:12.4f}")
        lines.append(f"  {'全天计划购电量':<14s} {g_plan.sum():12.4f}")
        lines.append(f"  {'全天计划购电费':<14s} {float(price_avg_day @ g_plan):12.2f}")
        lines.append("")
        lines.append("表1b  调整购电量（最终执行，指定时间段，kWh）")
        for name, t in dac.TBL1:
            lines.append(f"  {name:<14s} {g_final[t]:12.4f}")
        lines.append(f"  {'全天调整购电量':<14s} {g_final.sum():12.4f}")
        lines.append(f"  {'全天调整购电费':<14s} {float(price_avg_day @ g_final):12.2f}")
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

    summary_path = outdir / "problem4_3_summary.txt"
    summary_path.write_text(report, encoding="utf-8")
    print(f"\n[汇总] 已写入: {summary_path}")

    # ---------- 代表日画图 ----------
    try:
        import matplotlib.pyplot as plt
        try:
            plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
        except Exception:
            pass

        h = np.arange(T) * DT
        for rd in dac.REP_DATES:
            idx = int(np.where(dates_str == rd)[0][0])
            if idx >= len(results):
                continue
            r = results[idx]
            gv, cv, dv, socv = r["gv"], r["cv"], r["dv"], r["socv"]
            fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
            ax[0].plot(h, r["price_raw"], color="tab:red")
            ax[0].set_ylabel("电价 (元/kWh)")
            ax[0].set_title(f"问题4(问题3,波动电价,{args.epochs}点调整)：{rd}")
            ax[1].plot(h, r["actual_load_kw"], label="实际负荷", color="k")
            ax[1].plot(h, r["actual_pv_kw"], label="实际光伏", color="tab:orange")
            ax[1].plot(h, r["f_load_kw"], label="预测负荷", color="k", ls="--", alpha=0.6)
            ax[1].plot(h, r["f_pv_kw"], label="预测光伏", color="tab:orange", ls="--", alpha=0.6)
            ax[1].plot(h, gv / DT, label="调整后购电功率", color="tab:blue")
            ax[1].plot(h, (dv - cv) / DT, label="储能净放电功率", color="tab:green")
            ax[1].legend(ncol=3, fontsize=7)
            ax[1].set_ylabel("功率 (kW)")
            ax[2].plot(np.arange(T + 1) * DT, socv, color="tab:purple")
            ax[2].axhline(dac.SOC_MIN, ls="--", c="gray")
            ax[2].axhline(dac.SOC_MAX, ls="--", c="gray")
            ax[2].set_ylabel("储电量 (kWh)")
            ax[2].set_xlabel("时刻 (h)")
            fig.tight_layout()
            out_png = outdir / f"problem4_3_plot_{rd.replace('-', '')}.png"
            fig.savefig(out_png, dpi=150)
            plt.close(fig)
            print(f"[作图] 已保存: {out_png}")
    except Exception as e:
        print(f"[提示] 未作图（{e}）")

    print("\n完成。")


if __name__ == "__main__":
    main()
