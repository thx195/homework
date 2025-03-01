import numpy as np
import matplotlib.pyplot as plt

# 设置高斯分布的参数
mu, sigma = 0, 0.1  # 均值和标准差

# 生成高斯分布的数据
s = np.random.normal(mu, sigma, 1000)

# 绘制直方图
count, bins, ignored = plt.hist(s, 30, density=True)

# 绘制高斯分布曲线
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
         np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
         linewidth=2, color='r')
plt.title('Gaussian Distribution')
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()