from re import T
import jittor as jt
from jittor import nn
from Mymodel import mynet
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

data_dir = './TASK_2/dataset/cifar-10-batches-py/'
train_batch_list = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# 定义随机翻转函数
def random_flip(img):
    img0 = img
    img1 = img[:, :, :, ::-1]
    img2 = img[:, :, ::-1, :]
    img3 = img[:, :, ::-1, ::-1]
    return img0, img1, img2, img3



# 转换数据格式
def data_transform(data):
    data = np.array(data, dtype='float32')
    data = data.reshape(-1, 3, 32, 32)
    # normalize each channel with mean= [0.5,0.5,0.5] and std = [0.5,0.5,0.5]
    for i in range(len(data)):
        for j in range(3):
            data[i,j,:,:] = (data[i,j,:,:]-0.5) / (0.5) 
    return data


def load_train_data(batch_size):
    train_data = []
    train_labels = []
    for batch in train_batch_list:
        batch_data = unpickle(os.path.join(data_dir, batch))
        train_data.append(batch_data[b'data'])
        train_labels.append(batch_data[b'labels'])
    train_data = np.concatenate(train_data)
    train_labels = np.concatenate(train_labels)
    count = 0
    new_data = []
    new_label = []
    for i in range(train_data.shape[0]):
        if train_labels[i] < 5:
            count += 1
            if count % 10 == 0:
                new_data.append(data_transform(train_data[i]))
                new_label.append(train_labels[i])
        else:
            new_data.append(data_transform (train_data[i]))
            new_label.append(train_labels[i])
    train_data = []
    train_labels = []
    for i in range(len(new_data)):
        # 按照batch_size拼接数据
        if i % batch_size == 0:
            train_data.append(jt.concat(new_data[i:i + batch_size]))
            train_labels.append(jt.array(new_label[i:i + batch_size]))
    return train_data, train_labels

def load_test_data(batch_size):
    test_data = []
    test_labels = []
    batch_data = unpickle(os.path.join(data_dir, 'test_batch'))
    test_data.append(batch_data[b'data'])
    test_labels.append(batch_data[b'labels'])
    for i in range(len(test_data)):
        x = data_transform(test_data[i])
        test_data[i] = jt.array(x)
    new_data = []
    new_labels = []
    for i in range(len(test_data)):
        if i % batch_size == 0:
            new_data.append (jt.concat(test_data[i:i + batch_size]))
            new_labels.append(jt.array(test_labels[i:i + batch_size]))
    return new_data,new_labels


# 模型参数
lr = 0.001
epochs = 100
batch_size = 32
model = mynet()
loss = nn.CrossEntropyLoss()
optimizer = jt.optim.Adam(model.parameters(), lr=lr)
penalty_weight = 0.1
mode = 'train'

train_data,train_labels = load_train_data(batch_size)
test_data,test_labels = load_test_data(batch_size)

# 训练模型
if mode == 'train':
    loss_all = []
    total_step = len(train_data)

    for epoch in range(epochs):
        if (epoch+1)%50 == 0:
            lr = lr/100
            for params in optimizer.param_groups:
                        params['lr'] = lr
        for i in range(total_step):
            # print(train_data[i].shape)
            outputs = model(train_data[i])
            l = loss(outputs, jt.array(train_labels[i]))
            for j in range(len(train_labels[i])):
                if train_labels[i][j] < 5:
                    x = outputs[j].reshape(1, -1)
                    l += penalty_weight * loss(x, train_labels[i][j])
            optimizer.step(l)
            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch + 1, epochs, i + 1, total_step, l.item()))
                loss_all.append(l.item())
    # 画loss曲线
    plt.title('Loss')
    plt.plot(range(0, len(loss_all)),loss_all)
    plt.savefig('loss2.png')

    # 保存模型
    jt.save(model.state_dict(), 'mynet3.ckpt')

if mode == 'test':
    mode.load_state_dict(jt.load('mynet3.ckpt'))
    model.eval()
    with jt.no_grad():
        correct = 0
        total = 0
        for i in range(len(test_data)):
            outputs = model(test_data[i])
            predicted, _ = jt.argmax(outputs.data, 1)
            total += test_labels[i].size(1)
            correct += (predicted == jt.array(test_labels[i])).sum().item()

        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

