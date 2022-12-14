from pickletools import optimize
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import jittor as jt
from jittor import nn
from jittor import transform
from jittor.optim import Adam, SGD
from Mymodel import mynet, DeepPermNet, get_acc

data_dir = './deep_learning/TASK_2/dataset'

#### 模型参数 ####
mode = 'train'
# mode = 'test'
batch_size = 128
num_classes = 10
epochs = 100
lr = 0.001

#### 定义模型 ####
model_1 = mynet()

#### 定义损失函数和优化器 ####
loss_func = nn.CrossEntropyLoss()
optimizer_1 = Adam(model_1.parameters(), lr=lr)

#### 用pt_model初始化model_1 ####
model_1.load_state_dict(jt.load('pt_model1.pkl'))
model_1.classifier.requires_grad = True

#### 定义数据集 ####
train_transformer = transform.Compose([
    transform.RandomHorizontalFlip(),
    # transform.RandomGray(),
    transform.ToTensor(),
    transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
test_transformer = transform.Compose([
    # transform.RandomHorizontalFlip(),
    # transform.RandomGray(),
    transform.ToTensor(),
    transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

train_loader = jt.dataset.CIFAR10(root=data_dir,download=False,train=True,transform=train_transformer) 
test_loader = jt.dataset.CIFAR10(root=data_dir,download=False,train=False,transform=test_transformer)
train_loader.set_attrs(batch_size=batch_size)
test_loader.set_attrs(batch_size=batch_size)


#### 定义训练函数 ####
def train_model(model, x, y):
    pred = model(x)
    loss = loss_func(pred, y)
    return pred, loss

#### 定义测试函数 ####
def test_model(model, x, y):
    outputs = model(x)
    pred, _ = jt.argmax(outputs, 1)
    return pred, y

#### 训练 ####
if mode == 'train':
    losses = []
    for epoch in range(epochs):
        if (epoch + 1) % 20 == 0:
                lr /= 100
                for params in optimizer_1.param_groups:
                        params['lr'] = lr
        for i, (images, labels) in enumerate(train_loader):
            _, loss = train_model(model_1, images, labels)
            optimizer_1.step(loss)
            # print(i)
            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch + 1, epochs, i + 1, len(train_loader)/batch_size, loss.item()))
                losses.append(loss.item())
    #### save model ####
    jt.save(model_1.state_dict(), 'model_2.pkl')

    #### 画出loss曲线 ####
    plt.plot(losses)
    plt.savefig('loss3.2.png')

#### 测试 ####
if mode == 'test':
    model_1.load_state_dict(jt.load('model_2.pkl'))
    correct = 0
    total = 0
    for images, labels in test_loader:
        pred, y = test_model(model_1, images, labels)
        # print(pred,y)
        correct += (pred==y).sum().item()
        total += len(y)
    print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

