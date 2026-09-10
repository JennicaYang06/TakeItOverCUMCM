# 代码说明

## 环境

这台机器上没有安装可用的 Python（`python` 指向的是 Microsoft Store 占位程序）。
先装一个真正的 Python，再装依赖：

1. 从 https://www.python.org/downloads/windows/ 下载 Python 3.11/3.12，
   安装时勾选 **Add python.exe to PATH**。
   （或用 conda：`conda create -n cumcm python=3.11 -y && conda activate cumcm`）
2. 安装依赖：
   ```powershell
   python -m pip install -r code/requirements.txt
   ```

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
