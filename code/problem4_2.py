# -*- coding: utf-8 -*-
"""
问题4（对应问题2部分）：波动电价下的日前购电策略——LightGBM负荷/光伏预测 + 报童安全边际。

重要修正：电价"实时波动"意味着 0:00 制定计划时，当天的电价和负荷、光伏一样是未知的，
必须先预测，不能直接拿附件4当天的真实电价去解日前MILP——那等于假设运营商在0点就已经
100%精确知道当天每一格的电价，属于"开天眼"，不成立。正确流程分两层：
  1. 决策层：用"电价预测"（不是附件4真实值）构建 DayAheadMILP 的目标函数，据此决定
     买多少电、怎么充放电——这是运营商在0点真正能做到的事。
  2. 结算层：电价的现实值（附件4当天真实曲线）才决定"最终到底要付多少钱"——计划购电费、
     紧急购电费都用附件4的真实电价去结算，因为不管你以为电价是多少，账单是按实际发生的
     电价算的。这和负荷/光伏的处理完全对称："预测值决定动作，真实值决定后果"。

电价预测和负荷/光伏一样用 LightGBM 逐日滚动重训（问题2、问题3全线都用LightGBM，这里
不例外）：把附件4长表化（date/time/t/price）后，直接复用 problem2_lightgbm.py 里
"时间周期项sin/cos + 该时刻自身滞后1/2/3天 + 7/14日滑动均值"这套特征工程和
train_predict_one_day 训练函数——这两个函数本来就是按 `target` 列名参数化的通用函数，
不用改一行就能套到"price"这个新目标上；唯一需要的是往 problem2_lightgbm.FEATURES 这个
字典里加一条"price"的特征列表，用运行时赋值(`FEATURES["price"] = [...]`)而不是编辑
problem2_lightgbm.py 源文件——避免改动这份负荷/光伏预测已经验证过的文件。

其余部分与之前版本一致：
- 负荷/光伏预测仍是 problem2_lightgbm.py 的LightGBM逐日滚动重训。
- day_ahead_common.py 不做任何改动；导出/汇总因为"每天电价不同"仍是本文件自己写的版本
  （充放电表、紧急购电表和电价无关，直接复用 day_ahead_common 里那两个价格无关的函数）。
- DayAheadMILP 现在用"电价预测"构建（而不是之前误用的"电价真实值"），电价预测天天更新，
  所以仍然是每天重新构建一次（cvxpy构建本身很便宜，365次构建+求解实测约20秒）。
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import day_ahead_common as dac
from problem2_lightgbm import read_attachment2, add_features, add_lags, train_predict_one_day, FEATURES

DT = dac.DT
T = dac.T

# 电价的特征列表和 load/pv 同款写法，只是列名前缀换成 price（运行时挂到 FEATURES 上，
# 不编辑 problem2_lightgbm.py 源文件）。
FEATURES["price"] = ["t", "hour", "dow", "month", "doy", "sin_doy", "cos_doy", "sin_t", "cos_t",
                      "price_lag1", "price_lag2", "price_lag3", "price_mean7", "price_mean14"]


def read_attachment4_price(path: Path):
    """附件4：365天，每天一条144点电价曲线，格式与附件2各sheet完全一致。"""
    df = pd.read_excel(path, header=0)
    dates = pd.to_datetime(df.iloc[:, 0]).dt.normalize().to_numpy()
    price_kw_all = df.iloc[:, 1:1 + T].to_numpy(dtype=float)
    dates_str = np.array([pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates])
    return dates_str, price_kw_all


def read_attachment4_long(path: Path):
    """把附件4长表化成 date/time/t/price，供 add_features/add_lags/train_predict_one_day 使用
    （这几个函数本来就是按列名参数化的通用函数，和 problem2_lightgbm.read_attachment2 读
    小区负载/光伏的方式完全一样，只是这里只有一个sheet、一个目标列，不用merge）。"""
    df = pd.read_excel(path, header=0)
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"])
    data = df.melt(id_vars="date", var_name="time", value_name="price")
    data["t"] = data.groupby("date").cumcount()
    data = data.sort_values(["date", "t"]).reset_index(drop=True)
    return data


def export_purchase_sheet_perday(ws, results, key, price_key, export_start_idx, export_end_idx, n_days):
    """同 day_ahead_common._fill_purchase_sheet，只是"全天购电费"用每天自己的*真实*电价算
    （price_key 一般传 'price_actual_avg'，不能传预测电价——账单按真实电价结算）。"""
    for i in range(export_start_idx, export_end_idx + 1):
        r = results[i]
        row = 2 + (i - export_start_idx)
        gv = r[key]
        for col in range(2, 145):
            ws.cell(row=row, column=col, value=round(float(gv[col - 1]), 6))
        next_gv0 = results[i + 1][key][0] if i + 1 < n_days else gv[0]
        ws.cell(row=row, column=145, value=round(float(next_gv0), 6))
        ws.cell(row=row, column=146, value=round(float(gv.sum()), 6))
        ws.cell(row=row, column=147, value=round(float(r[price_key] @ gv), 6))


def export_result4_2_template(template_file, out_file, results, export_start_idx, export_end_idx, n_days):
    from openpyxl import load_workbook
    wb = load_workbook(template_file)
    export_purchase_sheet_perday(wb["计划购电量"], results, "gv", "price_actual_avg",
                                  export_start_idx, export_end_idx, n_days)
    dac._fill_charge_sheet(wb["充放电量"], results, export_start_idx, export_end_idx)
    dac._fill_emergency_sheet(wb["紧急购电量"], results, export_start_idx, export_end_idx)
    wb.save(out_file)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default=dac.EXPORT_START)
    ap.add_argument("--end", default=dac.EXPORT_END)
    ap.add_argument("--debug-days", type=int, default=None, help="调试：仅仿真前N天")
    ap.add_argument("--min-train-rows", type=int, default=100)
    ap.add_argument("--no-safety-margin", action="store_true", help="关闭报童安全边际，得到点预测基线结果")
    ap.add_argument("--outdir", default=None, help="输出目录，默认 results/problem4/problem2_volatile")
    args = ap.parse_args()

    use_safety_margin = not args.no_safety_margin

    here = Path(__file__).resolve().parent
    root, data_dir, template_dir = dac.resolve_data_dirs(here)
    outdir = Path(args.outdir) if args.outdir else root / "results" / "problem4" / "problem2_volatile"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[数据目录] {data_dir}")
    print(f"[模板目录] {template_dir}")
    print(f"[输出目录] {outdir}")

    pack = dac.load_price_and_actuals(data_dir)   # 只用它的负荷/光伏种子曲线和全年实际值，电价不用（附件1的）
    load_kw_seed, pv_kw_seed = pack["load_kw_seed"], pack["pv_kw_seed"]
    dates_str = pack["dates_str"]
    load_kw_all, pv_kw_all = pack["load_kw_all"], pack["pv_kw_all"]
    n_days_total = pack["n_days"]
    all_dates = pd.to_datetime(dates_str)

    print("准备读取附件4波动电价 ...")
    price_dates_str, price_kw_all = read_attachment4_price(data_dir / "附件4.xlsx")
    assert np.array_equal(price_dates_str, dates_str), "附件4 与附件2 的日期顺序不一致"
    print(f"附件4读取完成: {len(price_dates_str)} 天")

    print("准备读取附件2构造 LightGBM 特征表 ...")
    data = read_attachment2(data_dir / "附件2.xlsx")
    data = add_features(data)
    data = add_lags(data, "load")
    data = add_lags(data, "pv")
    print(f"特征表构造完成: {len(data)} 行 ({n_days_total} 天 x {T} 时刻)")

    print("准备读取附件4构造电价 LightGBM 特征表 ...")
    price_data = read_attachment4_long(data_dir / "附件4.xlsx")
    price_data = add_features(price_data)
    price_data = add_lags(price_data, "price")
    print(f"电价特征表构造完成: {len(price_data)} 行")

    n_days = min(n_days_total, args.debug_days) if args.debug_days else n_days_total
    if args.debug_days:
        print(f"[调试模式] 仅仿真前 {n_days} 天")

    export_start_idx, export_end_idx, do_export = dac.compute_export_range(
        dates_str, n_days, export_start=args.start, export_end=args.end)
    if not do_export:
        print(f"[调试模式] n_days={n_days} 未覆盖导出起点，跳过导出")
    print(f"[导出区间] {args.start} (day_idx={export_start_idx}) ~ {args.end} (day_idx={export_end_idx})")

    net_error_tracker = dac.NetErrorTracker(T)
    soc_prev_end = dac.SOC0_INIT
    results = []
    n_fallback = {"load": 0, "pv": 0, "price": 0}
    t_start = time.time()

    for day_idx in range(n_days):
        day = all_dates[day_idx]

        # ---------- 电价预测：0点还不知道今天的真实电价，用LightGBM预测（和load/pv同款套路） ----------
        f_price_kw = train_predict_one_day(price_data, day, "price", args.min_train_rows)
        if f_price_kw is None:
            f_price_kw = (price_kw_all[day_idx - 1] if day_idx > 0 else price_kw_all[0]).copy()
            n_fallback["price"] += 1
        f_price_avg = dac.adjacent_average(f_price_kw)
        milp = dac.DayAheadMILP(f_price_avg)   # 决策只能基于预测电价，不能偷看附件4真实值

        f_load_kw = train_predict_one_day(data, day, "load", args.min_train_rows)
        if f_load_kw is None:
            f_load_kw = (load_kw_all[day_idx - 1] if day_idx > 0 else load_kw_seed).copy()
            n_fallback["load"] += 1

        f_pv_kw = train_predict_one_day(data, day, "pv", args.min_train_rows)
        if f_pv_kw is None:
            f_pv_kw = (pv_kw_all[day_idx - 1] if day_idx > 0 else pv_kw_seed).copy()
            n_fallback["pv"] += 1

        f_load_e = dac.adjacent_average(f_load_kw) * DT
        f_pv_e = dac.adjacent_average(f_pv_kw) * DT
        f_net_e = f_load_e - f_pv_e

        margin = net_error_tracker.get_margin() if use_safety_margin else np.zeros(T)
        f_load_e_target = np.clip(f_load_e + margin, 0.0, None)

        gv, cv, dv, socv = milp.solve(soc_prev_end, f_load_e_target, f_pv_e)

        # ---------- 结算：真实负荷/光伏/电价回代，账单按真实电价算 ----------
        actual_price_raw = price_kw_all[day_idx]
        actual_price_avg = dac.adjacent_average(actual_price_raw)
        actual_load_kw = load_kw_all[day_idx]
        actual_pv_kw = pv_kw_all[day_idx]
        actual_load_e = dac.adjacent_average(actual_load_kw) * DT
        actual_pv_e = dac.adjacent_average(actual_pv_kw) * DT
        actual_net_e = actual_load_e - actual_pv_e

        deficit_e, plan_cost, emerg_cost = dac.settle_day(gv, cv, dv, actual_load_e, actual_pv_e, actual_price_avg)

        results.append(dict(
            day_idx=day_idx, date=dates_str[day_idx],
            gv=gv, cv=cv, dv=dv, socv=socv,
            price_raw=actual_price_raw, price_actual_avg=actual_price_avg,
            f_price_kw=f_price_kw,
            f_load_kw=f_load_kw, f_pv_kw=f_pv_kw,
            actual_load_kw=actual_load_kw, actual_pv_kw=actual_pv_kw,
            deficit_e=deficit_e, plan_cost=plan_cost, emerg_cost=emerg_cost,
        ))

        net_error_tracker.update(f_net_e, actual_net_e)
        soc_prev_end = float(socv[-1])

        if day_idx % 30 == 0 or day_idx == n_days - 1:
            print(f"[{day_idx + 1}/{n_days}] {dates_str[day_idx]}  "
                  f"计划购电费={plan_cost:8.2f}  紧急购电费={emerg_cost:8.2f}  soc_end={soc_prev_end:8.1f}")

    print(f"\n仿真完成，用时 {time.time() - t_start:.1f}s")
    print(f"冷启动回退天数: load={n_fallback['load']}  pv={n_fallback['pv']}  price={n_fallback['price']}")
    print(f"安全边际: {'启用 (q=' + str(dac.SAFETY_QUANTILE) + ')' if use_safety_margin else '未启用（点预测基线）'}")

    # ---------- 导出 result4-2.xlsx ----------
    try:
        if not do_export:
            raise RuntimeError("调试模式下仿真天数未覆盖导出区间，跳过导出")
        template_file = template_dir / "result4-2.xlsx"
        out_file = outdir / "result4-2.xlsx"
        export_result4_2_template(template_file, out_file, results, export_start_idx, export_end_idx, n_days)
        print(f"\n[模板填充] 已写入: {out_file}")
    except Exception as e:
        print(f"[提示] 未写入 result4-2.xlsx: {e}")
        if do_export:
            raise

    # ---------- 汇总报告（每天用自己的电价，不能调用 dac.build_summary_report） ----------
    exp_slice = results[export_start_idx:export_end_idx + 1]
    f_load_mat = np.array([r["f_load_kw"] for r in exp_slice])
    a_load_mat = np.array([r["actual_load_kw"] for r in exp_slice])
    f_pv_mat = np.array([r["f_pv_kw"] for r in exp_slice])
    a_pv_mat = np.array([r["actual_pv_kw"] for r in exp_slice])
    f_price_mat = np.array([r["f_price_kw"] for r in exp_slice])
    a_price_mat = np.array([r["price_raw"] for r in exp_slice])
    load_mae, load_rmse, load_mape = dac.error_metrics(f_load_mat, a_load_mat)
    pv_mae, pv_rmse, pv_mape = dac.error_metrics(f_pv_mat, a_pv_mat, mape_thresh=50.0)
    price_mae, price_rmse, price_mape = dac.error_metrics(f_price_mat, a_price_mat)

    total_plan_cost = sum(r["plan_cost"] for r in exp_slice)
    total_emerg_cost = sum(r["emerg_cost"] for r in exp_slice)
    emerg_days = sum(1 for r in exp_slice if r["emerg_cost"] > 1e-6)
    emerg_blocks = sum(int(np.sum(r["deficit_e"] > 1e-6)) for r in exp_slice)

    lines = []
    lines.append("=" * 70)
    lines.append("问题4(对应问题2，波动电价)  结果汇总")
    lines.append("=" * 70)
    lines.append(f"求解器: {dac.SOLVER}  (ETA_C={dac.ETA_C:.4f}, ETA_D={dac.ETA_D:.4f})")
    lines.append("电价: 附件4波动电价，0点用LightGBM预测决策，结算用当天真实值")
    lines.append("负荷/光伏/电价预测: LightGBM 逐日滚动重训（特征=时间周期项sin/cos + 滞后1/2/3天 + 7/14日滑动均值）")
    lines.append(f"冷启动回退天数: load={n_fallback['load']}  pv={n_fallback['pv']}  price={n_fallback['price']}（前一天实际值兜底）")
    lines.append(f"安全边际: {'启用 q=' + str(dac.SAFETY_QUANTILE) if use_safety_margin else '未启用'}")
    lines.append("")
    lines.append(f"预测误差（{args.start}~{args.end}）")
    lines.append("-" * 70)
    lines.append(f"  负荷  MAE={load_mae:8.2f} kW   RMSE={load_rmse:8.2f} kW   MAPE={load_mape:6.2f}%")
    lines.append(f"  光伏  MAE={pv_mae:8.2f} kW   RMSE={pv_rmse:8.2f} kW   MAPE(出力>50kW)={pv_mape:6.2f}%")
    lines.append(f"  电价  MAE={price_mae:8.4f} 元/kWh RMSE={price_rmse:8.4f} 元/kWh MAPE={price_mape:6.2f}%")
    lines.append("")
    lines.append(f"全年费用（{args.start}~{args.end}，{len(exp_slice)}天，按真实电价结算）")
    lines.append("-" * 70)
    lines.append(f"  计划购电费合计 = {total_plan_cost:12.2f} 元")
    lines.append(f"  紧急购电费合计 = {total_emerg_cost:12.2f} 元")
    lines.append(f"  总费用         = {total_plan_cost + total_emerg_cost:12.2f} 元")
    lines.append(f"  发生紧急购电天数 = {emerg_days} / {len(exp_slice)}")
    lines.append(f"  发生紧急购电时段数 = {emerg_blocks}")
    lines.append("")

    for rd in dac.REP_DATES:
        idx = int(np.where(dates_str == rd)[0][0])
        if idx >= len(results):
            continue
        r = results[idx]
        gv, cv, dv, socv, deficit_e = r["gv"], r["cv"], r["dv"], r["socv"], r["deficit_e"]
        price_actual_avg = r["price_actual_avg"]
        lines.append("=" * 70)
        lines.append(f"代表日 {rd}")
        lines.append("=" * 70)
        lines.append("表1  微网在指定时间段的购电量 (kWh)")
        for name, t in dac.TBL1:
            lines.append(f"  {name:<14s} {gv[t]:12.4f}")
        lines.append(f"  {'全天购电量 (kWh)':<14s} {gv.sum():12.4f}")
        lines.append(f"  {'全天购电费 (元)':<14s} {float(price_actual_avg @ gv):12.2f}")
        lines.append("")
        lines.append("表2  储能设备在指定时间段的充放电量 (kWh)")
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

    summary_path = outdir / "problem4_2_summary.txt"
    summary_path.write_text(report, encoding="utf-8")
    print(f"\n[汇总] 已写入: {summary_path}")

    # ---------- 代表日画图（每天用自己的电价曲线，不能用 dac.plot_representative_days） ----------
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
            ax[0].plot(h, r["price_raw"], label="真实电价", color="tab:red")
            ax[0].plot(h, r["f_price_kw"], label="预测电价", color="tab:red", ls="--", alpha=0.6)
            ax[0].legend(fontsize=7)
            ax[0].set_ylabel("电价 (元/kWh)")
            ax[0].set_title(f"问题4(问题2,波动电价)：{rd} 电价 / 功率平衡 / 储能电量")
            ax[1].plot(h, r["actual_load_kw"], label="实际负荷", color="k")
            ax[1].plot(h, r["actual_pv_kw"], label="实际光伏", color="tab:orange")
            ax[1].plot(h, r["f_load_kw"], label="预测负荷", color="k", ls="--", alpha=0.6)
            ax[1].plot(h, r["f_pv_kw"], label="预测光伏", color="tab:orange", ls="--", alpha=0.6)
            ax[1].plot(h, gv / DT, label="计划购电功率", color="tab:blue")
            ax[1].plot(h, (dv - cv) / DT, label="储能净放电功率", color="tab:green")
            ax[1].legend(ncol=3, fontsize=7)
            ax[1].set_ylabel("功率 (kW)")
            ax[2].plot(np.arange(T + 1) * DT, socv, color="tab:purple")
            ax[2].axhline(dac.SOC_MIN, ls="--", c="gray")
            ax[2].axhline(dac.SOC_MAX, ls="--", c="gray")
            ax[2].set_ylabel("储电量 (kWh)")
            ax[2].set_xlabel("时刻 (h)")
            fig.tight_layout()
            out_png = outdir / f"problem4_2_plot_{rd.replace('-', '')}.png"
            fig.savefig(out_png, dpi=150)
            plt.close(fig)
            print(f"[作图] 已保存: {out_png}")
    except Exception as e:
        print(f"[提示] 未作图（{e}）")

    print("\n完成。")


if __name__ == "__main__":
    main()
