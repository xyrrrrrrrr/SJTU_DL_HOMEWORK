from cgi import test
from ctypes import sizeof
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import jittor as jt
from jittor import Module
from jittor import nn

# 定义正态分布函数

def normfun(x, mu, sigma):
  pdf = 5 * np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
  return pdf

# 定义多项式函数

def myfun(x):
    return x*x*x+2*x*x+3*x+4


# 预设均值方差等参数

mu = 3
sigma = 1
range_low = 0
range_high = 6
T = 0.001

# 生成采样点

range_scale = np.arange(range_low, range_high, T)
# print(type(range_scale))
range_scale = list(range_scale)
x_set = random.sample(range_scale,1000)
train_xset = random.sample(x_set, 800)
test_xset = []
for i in x_set:
    if i in train_xset:
        continue
    test_xset.append(i)
train_xset.sort()
test_xset.sort()
x_set.sort()
train_yset = []
test_yset = []
y_set = []

for x in x_set:
    y = normfun(x, mu, sigma)
    # y = myfun(x)
    y_set.append(y)

# 先画出预设好的曲线的样子
plt.axis = [-0.5,6.5,0,0.5]
plt.title("task 1.1")
plt.plot(x_set, y_set,'bo-')
plt.savefig("task_1.1.png")

# 定义网络架构

class FCM(Module):
    def __init__(self):
        self.layer1 = nn.Linear(1, 100)
        self.Relu1 = nn.Relu()
        self.Relu2 = nn.Relu()
        self.sigmoid = nn.Sigmoid()
        self.layer2 = nn.Linear(100, 50)
        self.layer3 = nn.Linear(50, 1)
    
    def execute(self, x):
        x = self.layer1(x)
        x = self.Relu1(x)
        x = self.layer2(x)
        x = self.Relu2(x)
        x = self.layer3(x)
        return x


# 模型参数

np.random.seed(0)
jt.set_seed(3)
batch_size = 10
lr = 0.01
Epoch = 1000
model = FCM()
optim = nn.SGD(model.parameters(), lr=lr, weight_decay=0.01)
loss = nn.CrossEntropyLoss()

# 定义获取数据函数

def get_data(xset):
    for i in range(800):
        x = xset[i]
        y = normfun(x, mu, sigma) 
        # y = myfun(x)     
        yield jt.float32(x),jt.float32(y)
    

# 模型训练
for epoch in range(Epoch):
    print("Epoch:%d" % epoch)
    # time.sleep(0.1)
    # 对于每一个epoch，都要将数据集打乱
    random.shuffle(train_xset)
    for i,(x,y) in enumerate(get_data(train_xset)):
        pred_y = model(x)
        loss = ((pred_y - y)**2)
        loss_mean = loss.mean()
        optim.step(loss_mean)
        # optim.step(loss)
        print(f"step {i}, loss = {loss_mean.data.sum()}") 
    
# assert loss_mean.data < 0.005

print("train over")

# 测试

test_xset.sort()

for x in test_xset:
    y = model(jt.float32(x))
    test_yset.append(y)

# for x in test_xset:
#     print(normfun(x,mu,sigma))



plt.axis = [-0.5, 6.5, 0, 1]
plt.title("task 1.2")
plt.plot(x_set, y_set, 'bo-', markersize=0.5)
plt.plot(test_xset, test_yset, 'ro-', markersize=0.5)
# plt.savefig('task_1.2.png')
plt.savefig('task_1.3.png')




