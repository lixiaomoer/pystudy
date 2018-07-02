import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(47)
s = pd.Series(np.random.randint(1, 10, 5))
print("生成的Series序列：")
print(s)
print('\n\n')

# DataFrame是Pandas中另一个重要的数据结构
df = pd.DataFrame(data=np.random.randn(6, 4), columns=['A', 'B', 'C', 'D'])  # DataFrame的生成
print('生成的DataFrame对象：')
print(df)
print('DataFrame维度为：', df.shape[0], '行', df.shape[1], '列')

# 下面是对DataFrame对象的各种操作

# DataFrame中的每一行和每一列都是一个Series对象
print('\n\n')
print('df中的第1行为：')
print(df.iloc[0])
print('\n\n')

print(df.loc[0])  # loc和iloc方法的效果似乎一样
print('\n\n')

print('DataFrame中的第A列：')
print(df.A)  # 引用df中的数据列，直接使用列名就可以
print('\n\n')

print('DataFrame对象的基本描述：')
print(df.describe())
print('\n\n')

print('查看DataFrame对象的列名：')
print(df.columns)
print('\n\n')

### DataFrame对象的排序 ###
print(pd.__version__)  # 查看Pandas的版本
print('\n\n')

print('排序前：')
print(df)

print(df.sort_index())
print('\n')

print(df.sort_values(by='A'))  # 按照值进行排序
print('\n\n')

### DataFrame数据访问 ###
print('df的第1、2行：')
print(df[0:2])
print('\n')

print('df的倒数第1行：')
print(df.tail(1))
print('\n')

print('df的倒数第1、2行：')
print(df.tail(2))
print('\n')

print('访问df中的单个列：')
print(df['B'])
print('\n')

print('访问df中的多个列：')
print(df[['A', 'C']])
print('\n')

print('第1行，第1列的元素：')
print(df.iloc[0, 0])
print('\n')

print(df.iloc[0, 0:2])  # 其他的索引方式
print(df.iloc[0:4, 0:3])
print('\n')
print('\n')

print('使用loc：')
print(df.loc[0:4, 'A'])
print('在loc上加个函数也是没问题的：', df.loc[0:4, 'A'].sum())
print('使用loc时选择多列：')
print(df.loc[0:4:2, ['A', 'B']])
print('\n')

print('使用布尔值对df进行过滤：')
print(df[df.C > 0])
print('\n')
print('使用布尔值对df进行过滤，然后进行排序：')
print(df[df.C > 0].sort_values(by='C', ascending=False))
print('\n')

print('对df添加一列：')
df['Tag'] = list(['cat', 'dog', 'cat', 'cat', 'dog', 'cat'])
print(df)
print('\n')
print('统计Tag为dog的A列的平均值：')
print(df[df.Tag == 'dog'])
print(df[df.Tag == 'dog'].mean())  # 这个mean算出的是每一列的平均值
print('\n\n')

### Pandas时间序列操作 ###
# Pandas被提出的时候，就是以一种时间序列库的形式被提出的，时间序列处理是它的强项
print('Pandas时间序列操作：')
n_items = 366
ts = pd.Series(np.random.randn(n_items), index=pd.date_range('20010101', periods=n_items))
print(type(ts))
print('\n')
print(ts)
print('\n')
print('head of time series:')
print(ts.head(7))
print('\n')

# 说起Pandas的时间API那可就多了
print(ts.resample('1m').sum())
print('\n\n')

# 时间序列的可视化
# plt.figure(figsize=(10, 6), dpi=144)
# cs = ts.cumsum()
# print(cs)
# cs.plot()
# plt.show()

### Pandas的文件读写 ###
# 其实，Pandas用的最多的还是读取CSV文件
iris_csv = pd.read_csv('~/Desktop/iris.csv', index_col=0)
print('读取到的数据类型：')
print(type(iris_csv))
print('\n')

print('读取到的数据：')
print(iris_csv)
print('\n')



