import jittor as jt
from jittor import nn
from Mymodel import mynet
import numpy as np
import os
import matplotlib.pyplot as plt

data_dir = './dataset/cifar-10-batches-py/'
train_batch_list = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']

# 模型参数
lr = 0.001
epochs = 1
batch_size = 16
model = mynet()
loss = nn.CrossEntropyLoss()
optimizer = jt.optim.Adam(model.parameters(), lr=lr)
mode = 'train'


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

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
    test_data1 = []
    for i in range(len(test_data)):
        test_data1.append(data_transform(test_data[i]))
    new_data = []
    new_labels = []
    for i in range(len(test_data1)):
        if i % batch_size == 0:
            new_data.append (jt.concat(test_data1[i:i + batch_size]))
            new_labels.append(jt.array(test_labels[i:i + batch_size]))
    return new_data,new_labels

train_data,train_labels = load_train_data(batch_size)
test_data,test_labels = load_test_data(batch_size)


# 训练模型
if mode =='train':
    loss_all = []
    total_step = len(train_data)
    for epoch in range(epochs):
        for i in range(total_step):
            outputs = model(train_data[i])
            l = loss(outputs, jt.array(train_labels[i]))
            optimizer.step(l)
            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch + 1, epochs, i + 1, total_step, l.item()))
                loss_all.append(l.item())

    plt.title('Loss')
    plt.plot(range(0, len(loss_all)),loss_all)
    plt.savefig('loss11.png')

    # 保存模型
    jt.save(model.state_dict(), 'mynet2.ckpt')

# 测试模型
if mode == 'test':
    model.load_state_dict(jt.load('mynet2.ckpt'))
    model.eval()
    with jt.no_grad():
        correct = 0
        total = 0
        for i in range(len(test_data)):
            outputs = model(test_data[i])
            predicted, _ = jt.argmax(outputs.data, 1)
            total += test_labels[i].size(1)
            correct += (predicted == test_labels[i]).sum().item()
        print(total, correct)
        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
