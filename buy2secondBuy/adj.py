import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from factor_analyzer.factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus']=False

## 1 加载数据
df = pd.read_csv(r"feat2.csv")
df.drop(['Unnamed: 0'],axis=1,inplace=True)
df.drop(['XH'],axis=1,inplace=True)
df.dropna(inplace=True)

# 欠采样
under=RandomUnderSampler(sampling_strategy=1)
from imblearn.pipeline import Pipeline
steps=[('u',under)]
pipeline=Pipeline(steps=steps)
X=df.iloc[:,[0,1,2,3,5,6,7,9,10,12,13,15,16,14]]
X_resampled, y_resampled = pipeline.fit_resample(X, X['Label'])
X_resampled.to_csv(r'data.csv', index=False)
label=X_resampled.iloc[:,12]
df = X_resampled.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12]]
df.columns=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13']

# 充分性检验
faa = FactorAnalyzer(25,rotation=None)
faa.fit(df)
# 建立因子分析模型
faa_seven = FactorAnalyzer(9,rotation='varimax')
faa_seven.fit(df)

# （1）因子载荷矩阵
df1 = pd.DataFrame(np.abs(faa_seven.loadings_),index=df.columns)
df11 = np.array(df1)
# （2）因子得分
df2 = pd.DataFrame(faa_seven.transform(df))  # 用于将原始数据转换为因子得分
df22 = np.array(df2)

# 因子得分矩阵计算邻接矩阵
A = df22
A = np.array(A)
A = np.dot(A,A.T)
A = (A - np.min(A)) / (np.max(A) - np.min(A))
np.save("A.npy",A)
matrix = np.where(A >= np.percentile(A, 70), 1, 0)  # Threshold
np.save("matrix.npy",matrix)