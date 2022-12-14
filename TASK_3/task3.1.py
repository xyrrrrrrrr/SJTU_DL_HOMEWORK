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

# 模型参数
# mode = 'train'
mode = 'test'
batch_size = 128
num_classes = 10
epochs = 100
lr = 0.001
model = mynet(num_classes)
pt_model = DeepPermNet()
# 高斯随机初始化模型
def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        jt.init.trunc_normal_(m.weight, mean=0.05, std=0.01)
        jt.init.constant_(m.bias, 0.1)
        
pt_model.apply(init_weights)


# train_dataset = mydataset('train')
# test_dataset = mydataset('test')
# 定义数据加载器
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
 
transformer = transform.Compose([
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
  
train_loader = jt.dataset.CIFAR10(root=data_dir,download=False,train=True,transform=transformer) 
test_loader = jt.dataset.CIFAR10(root=data_dir,download=False,train=False,transform=test_transformer)
train_loader.set_attrs(batch_size=batch_size)
test_loader.set_attrs(batch_size=batch_size)
#### 训练DeepPermNet ####
if mode == 'train':
    # optimizer1 = SGD(pt_model.parameters(), lr=lr, momentum=0.9)
    optimizer1 = Adam(pt_model.parameters(), lr=lr)
    loss1 = nn.CrossEntropyLoss()# or MSE
    loss1 = loss1
    losses = []
    for epoch in range(epochs):
        if (epoch + 1) % 20 == 0:
            lr /= 100
            for params in optimizer1.param_groups:
                    params['lr'] = lr
        for i, (images, labels) in enumerate(train_loader):
            # print('stage:',i)
            outputs = pt_model(images)
            true_label = pt_model.get_label()
            true_label = jt.array(true_label)
            loss = 0
            for j in range(4):
                loss += loss1(outputs[:,j,:], true_label[:,j])
            # print(loss)
            # loss.backward()
            optimizer1.step(loss)
            # print(i)
            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, epochs, i + 1, len(train_loader)/batch_size, loss.item()))
                losses.append(loss.item())
            

                

    #### save model ####
    jt.save(pt_model.state_dict(), 'pt_model1.pkl')

    #### 画出loss曲线 ####
    plt.plot(losses)
    plt.savefig('loss1.png')
    
#### test ####
if mode == 'test':
    pt_model.load_state_dict(jt.load('pt_model.pkl'))
    correct = 0.
    total = 0.
    for images, _ in test_loader:
        images = images
        outputs = pt_model(images)
        label = pt_model.get_label()
        # print(outputs[0],label[0])
        for i in range(len(outputs)):
            pred, _ = jt.argmax(outputs[i], dim=1)
            for j in range(4):
                if pred[j] == label[i][j]:
                    correct += 1     
            total += 4             
        print(correct, total)
    print('Accuracy of the network on the 40000 test images: %d %%' % (100 * correct / total))