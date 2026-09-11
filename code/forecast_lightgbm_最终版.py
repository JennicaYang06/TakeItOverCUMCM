# -*- coding: utf-8 -*-
"""
问题2 第二版：LightGBM 负荷/光伏滚动预测 + 日前购电策略（全年滚动仿真）。

与 problem2.py（Holt-Winters + 晴空包络版）对比用：除预测方法外，日前 MILP、报童安全边际、
result2.xlsx 导出、汇总报告全部复用 day_ahead_common.py，确保两版可比。

预测思路（沿用原始版本的特征工程）：按144个时刻分别构造特征——时间周期项（星期/月份/年内日序的
sin/cos）+ 该时刻自身的滞后1/2/3天值 + 7/14天滑动均值，每天用"当天之前的全部历史"重新训练一个
LightGBM 回归模型（load、pv 各一个），预测当天。历史不足（开局约2周内，7/14日滑动均值还没数据）
时退化为"前一天实际值"兜底（day0 退化为附件1的示例曲线）。
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

import day_ahead_common as dac

DT = dac.DT
T = dac.T


# ======================= 数据读取 + 特征工程（原始版本，未改动核心逻辑） =======================
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


def add_features(df):
    df = df.copy()
    df["hour"] = df["t"] * DT
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["doy"] = df["date"].dt.dayofyear
    df["sin_doy"] = np.sin(2 * np.pi * df["doy"] / 365.0)
    df["cos_doy"] = np.cos(2 * np.pi * df["doy"] / 365.0)
    df["sin_t"] = np.sin(2 * np.pi * df["t"] / 144.0)
    df["cos_t"] = np.cos(2 * np.pi * df["t"] / 144.0)
    return df


def add_lags(df, col):
    df = df.copy()
    df = df.sort_values(["t", "date"]).reset_index(drop=True)
    grp = df.groupby("t")[col]
    df[f"{col}_lag1"] = grp.shift(1)
    df[f"{col}_lag2"] = grp.shift(2)
    df[f"{col}_lag3"] = grp.shift(3)
    df[f"{col}_mean7"] = grp.transform(lambda x: x.shift(1).rolling(7).mean())
    df[f"{col}_mean14"] = grp.transform(lambda x: x.shift(1).rolling(14).mean())
    df = df.sort_values(["date", "t"]).reset_index(drop=True)
    return df


FEATURES = {
    "load": ["t", "hour", "dow", "month", "doy", "sin_doy", "cos_doy", "sin_t", "cos_t",
              "load_lag1", "load_lag2", "load_lag3", "load_mean7", "load_mean14"],
    "pv": ["t", "hour", "dow", "month", "doy", "sin_doy", "cos_doy", "sin_t", "cos_t",
            "pv_lag1", "pv_lag2", "pv_lag3", "pv_mean7", "pv_mean14"],
}


def train_predict_one_day(data, day, target, min_train_rows):
    """用 day 之前的全部历史训练一个 LightGBM，预测 day 当天144个时刻。历史不足返回 None。"""
    feat_cols = FEATURES[target]
    tr = data[data["date"] < day].dropna(subset=feat_cols + [target])
    te = data[data["date"] == day].sort_values("t")
    if len(tr) < min_train_rows or len(te) == 0:
        return None
    Xtr = tr[feat_cols].to_numpy(float)
    ytr = tr[target].to_numpy(float)
    Xte = te[feat_cols].to_numpy(float)
    model = lgb.LGBMRegressor(
        n_estimators=120, learning_rate=0.06, num_leaves=31,
        subsample=0.9, colsample_bytree=0.9, random_state=42,
        n_jobs=-1, verbosity=-1)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    return np.clip(pred, 0.0, None)


# ======================= 主流程：全年滚动仿真（日前MILP + 报童安全边际 + 导出，逻辑与problem2.py一致） =======================
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default=dac.EXPORT_START, help="导出区间起始日期")
    ap.add_argument("--end", default=dac.EXPORT_END, help="导出区间结束日期")
    ap.add_argument("--debug-days", type=int, default=None, help="调试：仅仿真前N天")
    ap.add_argument("--min-train-rows", type=int, default=100, help="LightGBM最少训练样本数，不足则用前一天实际值兜底")
    ap.add_argument("--no-safety-margin", action="store_true", help="关闭报童安全边际，得到点预测基线结果")
    ap.add_argument("--outdir", default=None, help="输出目录，默认 results/problem2/lightgbm")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    root, data_dir, template_dir = dac.resolve_data_dirs(here)
    outdir = Path(args.outdir) if args.outdir else root / "results" / "problem2" / "lightgbm"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[数据目录] {data_dir}")
    print(f"[模板目录] {template_dir}")
    print(f"[输出目录] {outdir}")

    pack = dac.load_price_and_actuals(data_dir)
    price_raw, price_avg = pack["price_raw"], pack["price_avg"]
    load_kw_seed, pv_kw_seed = pack["load_kw_seed"], pack["pv_kw_seed"]
    dates_str = pack["dates_str"]
    load_kw_all, pv_kw_all = pack["load_kw_all"], pack["pv_kw_all"]
    n_days_total = pack["n_days"]
    all_dates = pd.to_datetime(dates_str)

    print("准备读取附件2构造 LightGBM 特征表 ...")
    data = read_attachment2(data_dir / "附件2.xlsx")
    data = add_features(data)
    data = add_lags(data, "load")
    data = add_lags(data, "pv")
    print(f"特征表构造完成: {len(data)} 行 ({n_days_total} 天 x {T} 时刻)")

    n_days = min(n_days_total, args.debug_days) if args.debug_days else n_days_total
    if args.debug_days:
        print(f"[调试模式] 仅仿真前 {n_days} 天")

    export_start_idx, export_end_idx, do_export = dac.compute_export_range(
        dates_str, n_days, export_start=args.start, export_end=args.end)
    if not do_export:
        print(f"[调试模式] n_days={n_days} 未覆盖导出起点，跳过 result2.xlsx 导出")
    print(f"[导出区间] {args.start} (day_idx={export_start_idx}) ~ {args.end} (day_idx={export_end_idx})")

    use_safety_margin = not args.no_safety_margin

    milp = dac.DayAheadMILP(price_avg)
    net_error_tracker = dac.NetErrorTracker(T)

    soc_prev_end = dac.SOC0_INIT
    results = []
    n_fallback = {"load": 0, "pv": 0}
    t_start = time.time()

    for day_idx in range(n_days):
        day = all_dates[day_idx]

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

        net_error_tracker.update(f_net_e, actual_net_e)
        soc_prev_end = float(socv[-1])

        if day_idx % 30 == 0 or day_idx == n_days - 1:
            print(f"[{day_idx + 1}/{n_days}] {dates_str[day_idx]}  "
                  f"计划购电费={plan_cost:8.2f}  紧急购电费={emerg_cost:8.2f}  soc_end={soc_prev_end:8.1f}")

    print(f"\n仿真完成，用时 {time.time() - t_start:.1f}s")
    print(f"冷启动回退天数: load={n_fallback['load']}  pv={n_fallback['pv']}"
          f"（历史有效训练样本 < {args.min_train_rows} 行时，用前一天实际值兜底）")
    print(f"安全边际: {'启用 (q=' + str(dac.SAFETY_QUANTILE) + ')' if use_safety_margin else '未启用（点预测基线）'}")

    # ---------- 导出 result2.xlsx ----------
    try:
        if not do_export:
            raise RuntimeError("调试模式下仿真天数未覆盖导出区间，跳过导出")
        template_file = template_dir / "result2.xlsx"
        out_file = outdir / "result2.xlsx"
        dac.export_result2_template(template_file, out_file, results, export_start_idx, export_end_idx,
                                     price_avg, n_days)
        print(f"\n[模板填充] 已写入: {out_file}")
    except Exception as e:
        print(f"[提示] 未写入 result2.xlsx: {e}")
        if do_export:
            raise

    # ---------- 汇总报告 ----------
    header_lines = [
        f"求解器: {dac.SOLVER}  (ETA_C={dac.ETA_C:.4f}, ETA_D={dac.ETA_D:.4f})",
        "负荷/光伏预测: LightGBM 逐日滚动重训（特征=时间周期项sin/cos + 滞后1/2/3天 + 7/14日滑动均值）",
        f"冷启动回退天数: load={n_fallback['load']}  pv={n_fallback['pv']}（前一天实际值兜底）",
        f"安全边际: {'启用 q=' + str(dac.SAFETY_QUANTILE) if use_safety_margin else '未启用'}",
    ]
    report, metrics_dict = dac.build_summary_report(header_lines, results, dates_str, export_start_idx,
                                                      export_end_idx, price_avg)
    print("\n" + report)

    summary_path = outdir / "problem2_summary.txt"
    summary_path.write_text(report, encoding="utf-8")
    print(f"\n[汇总] 已写入: {summary_path}")

    # ---------- 代表日画图 ----------
    try:
        saved = dac.plot_representative_days(results, dates_str, price_raw, outdir, title_prefix="问题2(LightGBM)")
        for p in saved:
            print(f"[作图] 已保存: {p}")
    except Exception as e:
        print(f"[提示] 未作图（{e}）")

    print("\n完成。")


if __name__ == "__main__":
    main()
