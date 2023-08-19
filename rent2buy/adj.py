from imblearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from factor_analyzer.factor_analyzer import FactorAnalyzer
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler

plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 1 加载数据
df = pd.read_csv(r"feat1.csv",index_col=0)
df.dropna(inplace=True)
df = df.sample(frac=0.125, random_state=42)

# 欠采样
under = RandomUnderSampler(sampling_strategy=1)
steps = [('u', under)]
pipeline = Pipeline(steps=steps)
X=df
X_resampled, y_resampled = pipeline.fit_resample(X, X['Label'])
X_resampled.to_csv(r'data.csv', index=False)
label = X_resampled.iloc[:, 10]
df = X_resampled.iloc[:,[0,1,2,3,4,5,6,7,8,9]]
df.columns=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10']

# 充分性检验
faa = FactorAnalyzer(25, rotation=None)
faa.fit(df)
# 建立因子分析模型
faa_six = FactorAnalyzer(7, rotation='varimax')  # 旋转方式：varimax，固定因子： 5
faa_six.fit(df)
# （1）因子载荷矩阵
df1 = pd.DataFrame(np.abs(faa_six.loadings_), index=df.columns)
df11 = np.array(df1)
# （2）因子得分
df2 = pd.DataFrame(faa_six.transform(df))  # 用于将原始数据转换为因子得分
df22 = np.array(df2)

# 因子得分矩阵计算邻接矩阵
A = df22
A = np.dot(A,A.T)
A = (A - np.min(A)) / (np.max(A) - np.min(A))
matrix = np.where(A >= np.percentile(A, 80),1, 0)
np.save("A.npy",A)
np.save("matrix.npy",matrix)