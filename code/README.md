# 代码说明

## 环境

`python` 指向的是 Microsoft Store 占位程序，装不了包。机器上已有 anaconda3，其中
`envs\robowriter` 这个环境装了 numpy/pandas/openpyxl/cvxpy/matplotlib，可以直接用：

```powershell
& "$env:USERPROFILE\anaconda3\envs\robowriter\python.exe" code\problem2.py
```

也可以新建一个专用环境：

1. `conda create -n cumcm python=3.11 -y && conda activate cumcm`
2. ```powershell
   python -m pip install -r code/requirements.txt
   ```

注意：这些环境都没装问题1用的 `GLPK_MI` 求解器，`problem2.py` 改用了 cvxpy 自带的
`HIGHS`（开源、速度更快）。如果要让 `problem1.py` 也能跑，把 `solver='GLPK_MI'`
改成 `solver='HIGHS'` 即可。

## 运行问题 1

```powershell
python code/problem1.py
```

输出写到项目根目录 `results/`：

| 文件 | 内容 |
| --- | --- |
| `result1.xlsx` | 在官方模板 `附件5/result1.xlsx` 上原地填数（提交用） |
| `problem1_summary.txt` | 论文表 1、表 2 的数值 + 校验信息 |
| `problem1_plot.png` | 电价 / 功率平衡 / 储能电量曲线（装了 matplotlib 才有） |

## 运行问题 2

问题2有两个可对比的版本，共用 `day_ahead_common.py` 里的日前 MILP、报童安全边际、
result2.xlsx 导出、汇总报告逻辑——**两版除"怎么预测负荷/光伏"外，其余建模假设完全一致**，
这样费用对比才有意义。

模型共同点：与问题1不同，负荷和光伏逐日变化且**不再给定预测值**，需要自己预测。每天0:00
只能用"过去数据"预测当天负荷、光伏曲线，代入问题1同款单日 MILP（`g,c,d,z,soc`，去掉了问题1
"0:00=24:00电量相同"的单日闭环，改成"次日初始电量=前一天计划末电量"的全年滚动衔接）求出
当天计划购电和充放电，储能按计划执行、不随实际负荷/光伏调整。白天过去后用附件2真实值回代：
供给不够的部分按5倍电价紧急购电，多余部分弃用。全年从 2025-1-1（SOC0=6000）跑起，1月作为
预测模型的历史预热，只导出 2025-2-1~12-31（334天）。

**安全边际（报童模型，两版都用）**：逐日独立MILP只对"点预测"取等号满足，没有理由为吸收预测
误差多买电，导致储能天天被放空、紧急购电占比极高。多买1单位电正常价p，用不完纯浪费；少买1
单位、缺口按5倍价紧急买单，比提前买多花 `5p-p=4p`。最优服务水位 `q*=4p/(4p+p)=0.8`，与p无关。
代码维护144个时刻各自的"净负荷(负荷-光伏)预测误差"滚动80分位数（`NetErrorTracker`，60天窗口），
叠加到点预测上再喂给LP。

求解器用 `HIGHS`（本机没装问题1用的 `GLPK_MI`），性能上365次MILP全部求解约15-20秒。

### 版本A：`problem2.py`（Holt-Winters + 晴空包络）

```powershell
python code/problem2.py          # 全年 365 天完整仿真
python code/problem2.py 45       # 调试：只跑前 45 天，快速验证
```

- **负荷预测**：144个时刻分别做加性 Holt-Winters（周期=7天，捕捉工作日/周末模式）。
- **光伏预测**：晴空包络（滚动90分位数，反映季节性最大出力）× 晴空指数（指数平滑，反映近期
  天气持续性）。
- 输出到 `results/problem2/holtwinters/`。

### 版本B：`forecast_lightgbm_最终版.py`（LightGBM 逐日滚动重训）

```powershell
python "code/forecast_lightgbm_最终版.py"                    # 全年
python "code/forecast_lightgbm_最终版.py" --debug-days 40    # 调试
python "code/forecast_lightgbm_最终版.py" --no-safety-margin # 关掉安全边际看基线
```

- **负荷/光伏预测**：每天用"当天之前的全部历史"重新训练一个 LightGBM（load、pv 各一个），
  特征=时间周期项（星期/月份/年内日序的sin/cos）+ 该时刻自身的滞后1/2/3天值 + 7/14天滑动均值。
  历史不足（开局约15天，7/14日滑动均值还没数据）时退化为"前一天实际值"兜底。
  全年365天×2个目标要重训约700次模型，耗时约2.5-3分钟（Holt-Winters版几秒钟就跑完，
  这是两版最大的实际差异——LightGBM 用更重的模型换取略高的精度）。
- 输出到 `results/problem2/lightgbm/`。

### 两版全年效果对比（2025.2.1-12.31，334天，均已启用安全边际）

| | Holt-Winters+晴空包络 | LightGBM |
| --- | --- | --- |
| 负荷 MAPE | 3.23% | 3.06% |
| 光伏 MAPE（出力>50kW） | 7.13% | 7.09% |
| 计划购电费 | 1301.4万 | 1289.0万 |
| 紧急购电费 | 97.3万 | 104.5万 |
| **总费用** | **1398.7万** | **1393.4万** |

两版预测精度、总费用都很接近，LightGBM 略胜一筹但优势有限，说明这个问题里预测方法本身
不是瓶颈——安全边际（报童分位数）才是把紧急购电费用压下来的关键（见上文，不加安全边际时
两版总费用都在1510-1520万左右）。

## `day_ahead_common.py`：两版共用的基础设施

- `DayAheadMILP`：单日 MILP（cvxpy Parameter 化，编译一次、365天复用求解）。
- `NetErrorTracker`：报童安全边际用的滚动分位数状态。
- `load_price_and_actuals` / `resolve_data_dirs`：附件读取与数据路径探测（优先找
  `code/附件/`，找不到则退回仓库根目录的 `题目和附件/C题/附件/`）。
- `export_result2_template` / `build_summary_report` / `plot_representative_days`：
  写官方模板、生成论文素材汇总、画四个代表日的曲线图。

## 模型（问题 1）

单日线性规划，决策为每 10 分钟的计划购电量 `g[t]`、储能充电量 `c[t]`、放电量 `d[t]`、
储能电量轨迹 `soc[t]`；目标最小化 `Σ price[t]·g[t]`；约束见 `problem1.py` 顶部注释。

关键假设：充放电总效率 90% 在充、放两侧平均分配（各 `sqrt(0.9)`）。
改 `problem1.py` 里的 `SPLIT_MODE = "charge"` 可把损耗全放到充电侧，用于灵敏度分析。

## 已知的模板小问题

`附件5/result1.xlsx` 的「计划购电量」工作表 A 列时间标签疑似整体错位一格
（从 `0:10-0:20` 开始，末行是 `0:00+1-0:10+1`，缺 `0:00-0:10`）。
脚本按「行序 = 当天第 t 个 10 分钟时段」填 B 列（第 2 行 = `0:00-0:10`……第 145 行 = `23:50-24:00`），
不改动 A 列。若最终要求按标签对齐，把 B 列整体上/下移一格即可。
