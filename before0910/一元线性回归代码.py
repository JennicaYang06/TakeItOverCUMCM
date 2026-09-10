#一元线性回归
import numpy as np
import pandas as pd #可以读取excel等表格

#可视化库：直接用里面的库函数
import matplotlib.pyplot as plt
import seaborn as sns

#数据集测试，训练，验证（更好的测试模型怎么样）
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error#(所有的回归模型都可以用)
#均方误差，r方决定系数，平均绝对误差

#统计分析库
import statsmodels.api as sm
from scipy import stats

#设置可视化风格
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif']=['SimHei']#用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False

#加载一下外部文件，如csv

#数据模拟
#1.生成随机数，保证运行程序时假数据是一样的
np.random.seed(25)
#2.制造学习时间
study_hours = np.random.uniform(1,10,100).reshape(-1,1)
#niform(1, 10, 100)：意思是随机生成 100个数字，这些数字均匀分布在 1到10之间。
# 代表100个学生，每人每天学习1到10个小时
# .reshape(-1, 1)：这是个格式调整操作，把它变成机器学习算法喜欢的“竖长条”表格格式。

#3.制造考试成绩
exam_scores = 30 + 7 * study_hours + np.random.normal(0, 5, 100).reshape(-1, 1)
#最后一项是随机噪声

#4.打包进excel表格
data = pd.DataFrame({'Study_Hours':study_hours.flatten(),'Exam_Scores':exam_scores.flatten()})


#检查一下
#print(data.head())

#print(study_hours.flatten())



#缺失值检查
#均值填充，删除缺失值，异常值分析
#均值填充的话，可以用
print('数据基本统计信息')
print(data.describe())
#用mean：后面的平均值干就行

#数据预处理
import pandas as pd

# 读取数据（换成你的文件名）
#data = pd.read_excel("数据.xlsx")   # 如果是CSV就写 pd.read_csv("数据.csv")

# 看前5行
print(data.head())

# 看基本信息：有多少行、有没有缺失值、数据类型
print(data.info())

# 看统计摘要：均值、标准差、最大最小值、四分位数
print(data.describe())

#一。数据缺失值
# 看看每列有多少缺失值
print(data.isnull().sum())

# 方法A：删掉有缺失的行（简单粗暴，数据多的时候用）
data_clean = data.dropna()

# 方法B：用均值/中位数填充（数据少的时候用，保留样本量）
#data['某列名'] = data['某列名'].fillna(data['某列名'].mean())   # 均值填充
# data['某列名'] = data['某列名'].fillna(data['某列名'].median())  # 中位数填充

#二。处理异常值
import matplotlib.pyplot as plt

# 方法A：箱线图可视化（直观看一下数据的样子）
plt.figure(figsize=(6, 4))#新建一个画布的代码，figsize是尺寸）
plt.boxplot(data['Exam_Scores'])#画箱线图
plt.title('考试成绩箱线图')
plt.ylabel('分数')
plt.show()

# 方法B：3σ 法则（超过均值±3倍标准差的视为异常）
mean = data['Exam_Scores'].mean()#算出来平均值
std = data['Exam_Scores'].std()#标准差
lower = mean - 3 * std
upper = mean + 3 * std

# 筛选出正常数据
data_clean = data[(data['Exam_Scores'] >= lower) & (data['Exam_Scores'] <= upper)]

# 方法C：IQR 法则（箱线图原理，不知道为啥要这么干但是是专家的方法）
Q1 = data['Exam_Scores'].quantile(0.25)
Q3 = data['Exam_Scores'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

data_clean = data[(data['Exam_Scores'] >= lower) & (data['Exam_Scores'] <= upper)]

#三。建模
# 第1步：确认线性关系（散点图+相关系数）
import seaborn as sns
from scipy import stats

# 散点图：直观看X和Y是不是大致呈直线关系
plt.figure(figsize=(8, 5))
sns.scatterplot(x=data_clean['X列名'], y=data_clean['Y列名'])
plt.xlabel('X（自变量）')
plt.ylabel('Y（因变量）')
plt.title('X-Y 散点图')
plt.show()

# 相关系数 + 显著性检验
corr, p_value = stats.pearsonr(data_clean['X列名'], data_clean['Y列名'])
print(f"Pearson相关系数 r = {corr:.4f}")
print(f"显著性 p值 = {p_value:.4f}")
# r越接近1或-1越好，p值<0.05说明相关性显著

# 第2步：拆分训练集和测试集 

# 取出X和Y（注意reshape成竖列）
X = data_clean['X列名'].values.reshape(-1, 1)
y = data_clean['Y列名'].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print(f"训练集: {len(X_train)} 条, 测试集: {len(X_test)} 条")

# ==================== 第3步：建模 ====================
model = LinearRegression()
model.fit(X_train, y_train)

a = model.coef_[0][0]   # 斜率
b = model.intercept_[0]  # 截距

print(f"\n回归方程: y = {a:.4f}x + {b:.4f}")

# ==================== 第4步：预测 + 评估 ====================
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"测试集 MSE: {mse:.4f}")
print(f"测试集 MAE: {mae:.4f}")
print(f"测试集 R²:  {r2:.4f}")

# ==================== 第5步：画回归直线图（论文必备配图）====================
plt.figure(figsize=(8, 5))
# 画原始数据点（散点）
plt.scatter(X_test, y_test, color='blue', alpha=0.6, label='实际值')
# 画回归直线
plt.plot(X_test, y_pred, color='red', linewidth=2, label='回归直线')
plt.xlabel('X（自变量）')
plt.ylabel('Y（因变量）')
plt.title(f'一元线性回归拟合图 (R² = {r2:.4f})')
plt.legend()
plt.show()

# ==================== 第6步：统计检验表（statsmodels，论文加分项）====================
import statsmodels.api as sm

# 加一列常数项（对应截距b）
X_sm = sm.add_constant(X)
model_sm = sm.OLS(y, X_sm).fit()

# 打印完整统计表
print(model_sm.summary())
