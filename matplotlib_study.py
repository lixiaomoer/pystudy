from matplotlib import pyplot as plt
import numpy as np

# matplotlib的基本使用
x = np.linspace(-np.pi, np.pi, num=50)
C, S = np.sin(x), np.cos(x)
plt.plot(x, C, color='blue', linewidth=2.0, linestyle='-')  # Cos
plt.plot(x, S, color='red', linewidth=2.0)  # Sin
# 设置坐标轴的长度
plt.xlim(x.min() * 1.1, x.max() * 1.1)
plt.ylim(C.min() * 1.1, C.max() * 1.1)
# 设置坐标轴的刻度和标签
plt.xticks((-np.pi, -np.pi / 2, np.pi / 2, np.pi), (r'$-\pi$', r'$-\pi/2$', r'$\pi/2$', r'$\pi$'))
plt.yticks([-1, -0.5, 0, 0.5, 1])
# 更改坐标轴位置
ax = plt.gca()  # get current axis
ax.spines['right'].set_color('none')  # 隐藏坐标轴
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')  # 设置刻度显示位置
ax.spines['bottom'].set_position(('data', 0))  # 设置下方坐标轴位置
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))
# 添加图例
# plt.legend(loc='upper left')
# 添加标记点
t = 2 * np.pi / 3
plt.plot([t, t], [0, np.cos(t)], color='blue', linewidth=1.5, linestyle='--')
# 绘制蓝色标记点
plt.scatter([t, ], [np.cos(t), ], 50, color='blue')
# 画出标记点的值
plt.annotate(r'$cos(\frac{2\pi}{3})=-\frac{1}{2}$',
             xy=(t, np.cos(t)), xycoords='data',
             xytext=(-90, -50), textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=.2'))
# 修改坐标轴字体大小
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(16)
    label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.65))
plt.show()  # 显示画图窗口

# Tips: beautiful plot is not important, but plot is important.


# plot.figure()函数的使用
X = np.linspace(-np.pi, np.pi, 200, endpoint=True)
C, S = np.sin(X), np.cos(X)
plt.figure(num='sin', figsize=(16, 4))
plt.plot(X, S)
plt.figure(num='cos', figsize=(16, 4))  # 再创建一个figure对象
plt.plot(X, C)
plt.figure(num='sin')  # 注意！再切换到sin图形上
plt.plot(X, C)
print(plt.figure(num='sin').number)
print(plt.figure(num='cos').number)
plt.show()

# 绘制子图
plt.figure(figsize=(18, 4), dpi=256)
plt.subplot(2, 2, 1)  # 绘图区域为2X2，并在第1个子图上进行绘制
plt.xticks(())
plt.yticks(())
plt.text(0.5, 0.5, 'subplot(2,2,1)', ha='center', va='center', size=20, alpha=0.5)

plt.subplot(2, 2, 2)  # 在第2个子图上进行绘制
plt.xticks(())
plt.yticks(())
plt.text(0.5, 0.5, 'subplot(2,2,2)', ha='center', va='center', size=20, alpha=0.5)

plt.subplot(2, 2, 3)  # 在第3个子图上进行绘制
plt.xticks(())
plt.yticks(())
plt.text(0.5, 0.5, 'subplot(2,2,3)', ha='center', va='center', size=20, alpha=0.5)

plt.subplot(2, 2, 4)  # 在第4个子图上进行绘制
plt.xticks(())
plt.yticks(())
plt.text(0.5, 0.5, 'subplot(2,2,4)', ha='center', va='center', size=20, alpha=0.5)

plt.tight_layout()
plt.show()

# 比较复杂的绘制子图需要使用gridspec来实现
import matplotlib.gridspec as gridspec

plt.figure(figsize=(18, 4))
G = gridspec.GridSpec(3, 3)  # 一个3X3的画布

axes_1 = plt.subplot(G[0, :])  # 第1行，所有的列
plt.xticks()
plt.yticks()
plt.text(.5, .5, 'Axes 1', ha='center', va='center', size=24, alpha=.5)

axes_2 = plt.subplot(G[1:, 0])  # 第2、3行，第1列
plt.xticks()
plt.yticks()
plt.text(.5, .5, 'Axes 2', ha='center', va='center', size=24, alpha=.5)

axes_3 = plt.subplot(G[1:, -1])  # 第2、3行，倒数第1列
plt.xticks()
plt.yticks()
plt.text(.5, .5, 'Axes 3', ha='center', va='center', size=24, alpha=.5)

axes_4 = plt.subplot(G[1, 1])  # 第2行，第2列
plt.xticks()
plt.yticks()
plt.text(.5, .5, 'Axes 4', ha='center', va='center', size=24, alpha=.5)

axes_5 = plt.subplot(G[2, 1])  # 第3行，第2列
plt.xticks()
plt.yticks()
plt.text(.5, .5, 'Axes 5', ha='center', va='center', size=24, alpha=.5)

plt.tight_layout()
plt.show()

# 图中图的制作需要使用axes实现
plt.figure(figsize=(18, 4))
plt.axes([.1, .1, .8, .8])  # 坐标轴定位矩形，[.left,.bottom,width,height]
plt.xticks()
plt.yticks()
plt.text(.2, .5, 'axes([.1, .1, .8, .8])', va='center', ha='center', size=20, alpha=0.5)

plt.axes([.5, .5, .3, .3])  # 坐标轴定位矩形，[.left,.bottom,width,height]
plt.xticks()
plt.yticks()
plt.text(.5, .5, 'axes([.5, .5, .3, .3])', va='center', ha='center', size=20, alpha=0.5)
plt.show()


# 显示刻度标签
def tickline():
    plt.xlim(0, 10)
    plt.ylim(-1, 1)
    plt.yticks([])
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_minor_locator(plt.MultipleLocator(.1))
    for label in ax.get_xticklabels():
        label.set_fontsize(16)
    ax.plot(np.arange(11), np.zeros(11))
    return ax


locators = [
    'plt.NullLocator()',
    'plt.MultipleLocator(base=1.0)',
    'plt.FixedLocator(locs=[0,2,8,9,10])',
    'plt.IndexLocator(base=3, offset=1)',
    'plt.LinearLocator(numticks=5)',
    'plt.LogLocator(base=2, subs=[1.0])',  # 对数
    'plt.MaxNLocator(nbins=3, steps=[1, 3, 5, 7, 9, 10])',
    'plt.AutoLocator()'
]

n_locators = len(locators)
print(len(locators))

# 计算图形对象的大小
size = 1024, 60 * n_locators
dpi = 72.0
fig_size = size[0] / float(dpi), size[1] / float(dpi)
fig = plt.figure(figsize=fig_size, dpi=dpi)
fig.patch.set_alpha(0)

for i, locator in enumerate(locators):
    plt.subplot(n_locators, 1, i + 1)
    ax = tickline()
    ax.xaxis.set_major_locator(eval(locator))  # eval()直接运行字符串形式的python语句
    plt.text(5, .3, locator[3:], ha='center', size=16)

plt.subplots_adjust(bottom=.1, top=.99, left=.01, right=.99)
plt.show()

### matplotlib画图 ###

## 散点图 ##
n = 1024
X = np.random.normal(0, 1, n)
Y = np.random.normal(0, 1, n)
T = np.arctan2(Y, X)
plt.subplot(1, 2, 1)  # 1行2列，并激活第1图
plt.scatter(X, Y, s=75, c=T, alpha=.5)
plt.xlim(-1.5, 1.5)
plt.xticks(())
plt.ylim(-1.5, 1.5)
plt.yticks(())

## 填充图 ##
n = 256
X = np.linspace(-np.pi, np.pi, n, endpoint=True)
Y = np.sin(2 * X)
plt.subplot(1, 2, 2)  # 激活第2图
plt.plot(X, Y + 1, color='blue', alpha=1.0)
plt.fill_between(X, y1=1, y2=Y + 1, color='blue', alpha=.25)
plt.plot(X, Y - 1, color='blue', alpha=1.0)
plt.fill_between(X, y1=-1, y2=Y - 1, where=(Y - 1) > -1, color='blue', alpha=.25)
plt.fill_between(X, -1, Y - 1, (Y - 1) < -1, color='red', alpha=.25)
plt.xlim(-np.pi, np.pi)
plt.xticks(())  # 不设置坐标轴标签
plt.ylim(-2.5, 2.5)
plt.yticks(())

plt.show()

## 柱状图和等高线图 ##
n = 12  # 柱状图
X = np.arange(n)
Y1 = (1 - X / float(n)) * np.random.uniform(.5, 1, n)
Y2 = (1 - X / float(n)) * np.random.uniform(.5, 1.0, n)
plt.subplot(1, 2, 1)
plt.bar(X, Y1, facecolor='#9999ff', edgecolor='white')
plt.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')
# 柱状图添加标签
for x, y in zip(X, Y1):
    plt.text(x + 0.4, y + 0.05, '%.2f' % y, ha='center', va='bottom')
for x, y in zip(X, Y2):
    plt.text(x + 0.4, -y - 0.05, '%.2f' % y, ha='center', va='top')
plt.xlim(-0.5, n)
plt.xticks(())
plt.ylim(-1.25, 1.25)
plt.yticks(())


def f(x, y):
    return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)


n = 256  # 等高线图
x = np.linspace(-3, 3, n)
y = np.linspace(-3, 3, n)
X, Y = np.meshgrid(x, y)
plt.subplot(1, 2, 2)
plt.contourf(X, Y, f(X, Y), 8, alpha=.75, cmap=plt.cm.hot)
C = plt.contour(X, Y, f(X, Y), 8, colors='black')
plt.clabel(C, inline=1, fontsize=10)
plt.xticks(())
plt.yticks(())

plt.show()

### 热成像图和饼图 ###
plt.subplot(1, 2, 1)  # 热成像图
n = 10
x = np.linspace(-3, 3, 4 * n)
y = np.linspace(-3, 3, 3 * n)
X, Y = np.meshgrid(x, y)
plt.imshow(f(X, Y), cmap='hot', origin='low')
plt.colorbar(shrink=.85)
plt.xticks(())
plt.yticks(())
plt.subplot(1, 2, 2)  # 饼图
n = 20
Z = np.ones(n)
Z[-1] *= 2
plt.pie(Z, explode=Z * .05, colors=['%f' % (i / float(n)) for i in range(n)])
plt.axis('equal')
plt.xticks(())
plt.yticks(())
plt.show()

### 网格和极坐标图 ###
ax = plt.subplot(1, 2, 1)
ax.set_xlim(0, 4)
ax.set_ylim(0, 3)
ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))  # 主次刻度
ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
ax.grid(which='major', axis='x', linewidth=.75, linestyle='-', color='grey')
ax.grid(which='minor', axis='x', linewidth=.25, linestyle='-', color='grey')
ax.grid(which='major', axis='y', linewidth=.75, linestyle='-', color='grey')
ax.grid(which='minor', axis='y', linewidth=.25, linestyle='-', color='grey')
ax.set_xticklabels([])  # 标签什么也不设
ax.set_yticklabels([])

ax = plt.subplot(1, 2, 2, polar=True)  # 极坐标图
N = 20
theta = np.arange(0.0, 2 * np.pi, 2 * np.pi)
radii = 10 * np.random.rand(N)
width = np.pi / 4 * np.random.rand(N)
bars = plt.bar(theta, radii, width=width)
for r, bar in zip(radii, bars):
    bar.set_facecolor(plt.cm.jet(r / 10.))
    bar.set_alpha(.5)

ax.set_xticklabels([])
ax.set_yticklabels([])
plt.show()
