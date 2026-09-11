from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd
import cvxpy as cp
np.set_printoptions(suppress=True)


# ===================== 公共参数（附录 1）=====================
DT = 1.0 / 6.0                    # 每格时长（小时）
SOC_MIN, SOC_MAX = 1200.0, 10800.0
CAP_MAX = 12000.0                 # 最大容量（此处 SOC_MAX < CAP_MAX，实际以 10800 为准）
SOC0 = 6000.0                     # 2025-1-1 0:00 初始电量
P_MAX = 5000.0                    # 最大充放电功率 kW
E_STEP_MAX = P_MAX * DT           # 每格最大充/放电量 kWh ≈ 833.333
RT_EFF = 0.90                     # 充放电总效率（往返）

ETA_C = np.sqrt(RT_EFF)
ETA_D = np.sqrt(RT_EFF)


# ======================= 读取附件 1 =======================
data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '附件', '附件1.xlsx')
a = pd.read_excel(data_file, header=None)
# 列顺序：时间 | 电价(元/kWh) | 小区负载(kW) | 光伏发电预测功率(kW)
b = a.values
x = b[1:, 0]        # 时间列
price = b[1:, 1]    # 电价
load_kw = b[1:, 2]  # 负载
pv_kw = b[1:, 3]    # 光伏
T = len(price)
assert T == 144, f"附件1 应为 144 行数据，实际读到 {T} 行"

# ========== 首尾相接平均函数 ==========
def adjacent_average(arr):
    """
    首尾相接平均：
    第1个 = (最后一个 + 第一个) / 2
    第2个 = (第一个 + 第二个) / 2
    ...
    第n个 = (第n-1个 + 第n个) / 2
    数据点数不变
    """
    result = np.zeros(len(arr))
    result[0] = (arr[-1] + arr[0]) / 2  # 最后一个和第一个平均，作为第1个
    for i in range(1, len(arr)):
        result[i] = (arr[i - 1] + arr[i]) / 2  # 前一个和当前平均
    return result

# ========== 调用函数处理数据==========
price_avg = adjacent_average(price)
load_avg = adjacent_average(load_kw)
pv_avg = adjacent_average(pv_kw)
load_e = load_avg * DT            # 每格负载电量 kWh
pv_e = pv_avg * DT               # 每格光伏电量 kWh

# ======================= 建立 MILP 模型 =======================
g = cp.Variable(T, nonneg=True, name="plan_buy")
c = cp.Variable(T, nonneg=True, name="charge")
d = cp.Variable(T, nonneg=True, name="discharge")
z = cp.Variable(T, boolean=True, name="charge_discharge_flag")
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
    c <= E_STEP_MAX * z,            # 若 z[i]=0 则 c[i]=0（不充电）
    d <= E_STEP_MAX * (1 - z),
]

cost = price_avg @ g                                       # 全天购电费（元）
prob = cp.Problem(cp.Minimize(cost), cons)


prob.solve(solver= 'GLPK_MI')
print(f"求解状态: {prob.status} | 求解器: {'GLPK_MI'}")
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
print(f"最小(供电-负载) = {viol:.6e}  (满足到容差范围内)")
print(f"soc 范围 = [{socv.min():.3f}, {socv.max():.3f}]  应 ⊆ [1200, 10800]")
print(f"同时充放电的时段数 = {both}  (期望 0)")
print(f"0:00 储电量 = {socv[0]:.3f} | 24:00 储电量 = {socv[-1]:.3f}")

daily_buy = float(gv.sum())
daily_cost = float(price_avg @ gv)
print(f"\n全天购电量 = {daily_buy:.3f} kWh")
print(f"全天购电费 = {daily_cost:.2f} 元")
try:
    from openpyxl import load_workbook
    template_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),'附件','附件5', 'result1.xlsx')
    wb_tpl = load_workbook(template_file)

    # ---------- Sheet1: 计划购电量 ----------
    ws_tpl1 = wb_tpl['计划购电量']
    # 模板时间格式：row2="0:10-0:20" → gv[1], row3="0:20-0:30" → gv[2], ..., row144="23:50-0:00+1" → gv[143], row145="0:00+1-0:10+1" → gv[0]
    for r in range(2, 145):  # row 2 ~ row 144
        idx = r - 1  # row2→1, row3→2, ..., row144→143
        ws_tpl1.cell(row=r, column=2, value=round(float(gv[idx]), 6))
    # row 145: "0:00+1-0:10+1" 对应下一天第一个时段 → gv[0]
    ws_tpl1.cell(row=145, column=2, value=round(float(gv[0]), 6))

    # ---------- Sheet2: 充放电量 ----------
    ws_tpl2 = wb_tpl['充放电量']
    # 6个4小时时段: 0:00-4:00(0:24), 4:00-8:00(24:48), 8:00-12:00(48:72), 12:00-16:00(72:96), 16:00-20:00(96:120), 20:00-24:00(120:144)
    periods = [(2, 0, 24), (3, 24, 48), (4, 48, 72), (5, 72, 96), (6, 96, 120), (7, 120, 144)]
    for row_num, start_idx, end_idx in periods:
        ws_tpl2.cell(row=row_num, column=2, value=round(float(cv[start_idx:end_idx].sum()), 6))  # B列=充电量
        ws_tpl2.cell(row=row_num, column=3, value=round(float(dv[start_idx:end_idx].sum()), 6))  # C列=放电量
    # D2="0:00" 和 D3="24:00" 模板已有，无需填写
    # E2=0:00储电量, E3=24:00储电量
    ws_tpl2.cell(row=2, column=5, value=round(float(socv[0]), 6))
    ws_tpl2.cell(row=3, column=5, value=round(float(socv[-1]), 6))

    wb_tpl.save(template_file)
    print(f"\n[模板填充] 已将优化结果写入: {template_file}")
    print("  Sheet1 计划购电量: 144行购电量已填充")
    print("  Sheet2 充放电量: 6时段充放电量+储电量已填充")
except Exception as e:
    print(f"[提示] 填充模板失败: {e}")
# ======================= 时间格式化辅助函数 =======================
def time_range_str(i, T_total=144):
    """将第 i 个10分钟时段转为区间字符串，如 '00:00-00:10'、'23:50-24:00'"""
    start_min = i * 10
    end_min = (i + 1) * 10
    # 处理 24:00 的情况（end_min=1440 时显示为 24:00）
    if end_min >= 1440:
        end_h, end_m = 24, 0
    else:
        end_h, end_m = end_min // 60, end_min % 60
    start_h, start_m = start_min // 60, start_min % 60
    return f"{start_h:02d}:{start_m:02d}-{end_h:02d}:{end_m:02d}"
# ======================= 论文表 1 / 表 2 =======================
# 表 1：指定单个 10 分钟时段的购电量。t = 起始分钟 / 10
TBL1 = [("10:00-10:10", 60), ("12:00-12:10", 72), ("14:00-14:10", 84),
        ("16:00-16:10", 96), ("18:00-18:10", 108), ("20:00-20:10", 120)]
# 表 2：6 个 4 小时时段的充/放电量合计
TBL2 = [("0:00-4:00", 0, 24), ("4:00-8:00", 24, 48), ("8:00-12:00", 48, 72),
        ("12:00-16:00", 72, 96), ("16:00-20:00", 96, 120), ("20:00-24:00", 120, 144)]

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment, PatternFill, Border, Side

    wb = Workbook()

    # ---------- Sheet1: 平均后数据 ----------
    ws1 = wb.active
    ws1.title = "平均后数据"

    # 表头样式
    header_font = Font(bold=True, size=11)
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    header_font_white = Font(bold=True, size=11, color="FFFFFF")
    thin_border = Border(
        left=Side(style='thin'), right=Side(style='thin'),
        top=Side(style='thin'), bottom=Side(style='thin')
    )

    # 写表头
    headers1 = ["序号", "时间", "电价(元/kWh)", "负载kW(平均后)", "光伏kW(平均后)"]
    for col_idx, h in enumerate(headers1, 1):
        cell = ws1.cell(row=1, column=col_idx, value=h)
        cell.font = header_font_white
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal='center', vertical='center')
        cell.border = thin_border

    # 写数据
    for i in range(T):
        time_str = time_range_str(i)
        row = i + 2
        ws1.cell(row=row, column=1, value=i + 1).border = thin_border
        ws1.cell(row=row, column=2, value=time_str).border = thin_border
        ws1.cell(row=row, column=3, value=round(price_avg[i], 4)).border = thin_border
        ws1.cell(row=row, column=4, value=round(load_avg[i], 4)).border = thin_border
        ws1.cell(row=row, column=5, value=round(pv_avg[i], 4)).border = thin_border

    # 设置列宽
    ws1.column_dimensions['A'].width = 8
    ws1.column_dimensions['B'].width = 16
    ws1.column_dimensions['C'].width = 14
    ws1.column_dimensions['D'].width = 16
    ws1.column_dimensions['E'].width = 16
    for row in ws1.iter_rows():
        for cell in row:
            cell.alignment = Alignment(horizontal='center', vertical='center')
    # 保存文件
    output_path = "平均后数据与优化结果.xlsx"
    wb.save(output_path)
    print(f"\n[Excel导出] 文件已保存: {output_path}")
    print(f"  Sheet1: 平均后数据 ({T}行)")
    print(f"  Sheet2: 优化结果 ({T+1}行)")
except ImportError:
    print("[提示] openpyxl 未安装，跳过Excel导出。运行: pip install openpyxl")
except Exception as e:
    print(f"[提示] Excel导出失败: {e}")


lines = []
lines.append("=" * 60)
lines.append("问题 1  结果汇总")
lines.append("=" * 60)
lines.append(f"求解器: GLPK_MI (ETA_C={ETA_C:.4f}, ETA_D={ETA_D:.4f})")
lines.append("")
lines.append("表 1  微网在指定时间段的购电量 (kWh)")
lines.append("-" * 60)
for name, t in TBL1:
    lines.append(f"  {name:<14s} {gv[t]:12.4f}")
lines.append("-" * 60)
lines.append(f"  {'全天购电量 (kWh)':<14s} {daily_buy:12.4f}")
lines.append(f"  {'全天购电费 (元)':<14s} {daily_cost:12.2f}")
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
lines.append(f"  min(供电 - 负载) = {viol:.3e}  (满足到容差范围内)")
lines.append(f"  soc ∈ [{socv.min():.2f}, {socv.max():.2f}]")
lines.append(f"  同时充放电时段数 = {both}")
report = "\n".join(lines)
print("\n" + report)

# ======================= 弹窗展示结果 =======================
from tkinter import Tk, Text, Scrollbar, Button, Frame
def show_popup(title, text):
    """用 tkinter 弹窗显示文本内容，带滚动条"""
    root = Tk()
    root.title(title)
    root.geometry("700x500")

    # 文本区
    text_widget = Text(root, wrap="word", font=("Microsoft YaHei", 10))
    text_widget.insert("1.0", text)
    text_widget.config(state="disabled")  # 只读

    # 滚动条
    scrollbar = Scrollbar(root, command=text_widget.yview)
    text_widget.config(yscrollcommand=scrollbar.set)

    text_widget.pack(side="left", fill="both", expand=True, padx=5, pady=5)
    scrollbar.pack(side="right", fill="y", padx=0, pady=5)

    # 关闭按钮
    btn_frame = Frame(root)
    btn_frame.pack(side="bottom", fill="x", padx=5, pady=5)
    close_btn = Button(btn_frame, text="关闭", font=("Microsoft YaHei", 10),
                    command=root.destroy)
    close_btn.pack(side="right", padx=5)

    root.mainloop()

# 弹窗显示结果汇总
show_popup("问题1 结果汇总", report)

print("\n弹窗已显示，关闭窗口后作图。")

# ======================= 可选作图 =======================
try:
    
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
    plt.show()
except Exception as e:
    print(f"[提示] 未作图（{e}）")

print("\n图像已显示，关闭窗口后程序退出。")

