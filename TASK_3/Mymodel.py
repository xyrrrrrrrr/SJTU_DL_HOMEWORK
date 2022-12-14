from doctest import TestResults
import jittor as jt
from jittor import nn
from jittor import transform
import pygmtools as pygm
import cvxpy as cp
# import torch
# from cvxpylayers.numpy import CvxpyLayer
import numpy as np
import os

pygm.BACKEND = 'jittor'
# use torch to construct a LeNet model
class mynet(nn.Module):
    def __init__(self):
        super(mynet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                            out_channels=16,
                            kernel_size=3,
                            stride=2,
                            padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,3,2,1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32,64,3,2,1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64,64,2,2,0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.mlp1 = nn.Linear(4 * 64,128)
        self.mlp2 = nn.Linear(128,16)
        self.classifier = nn.Sequential(nn.Linear(16,16),nn.ReLU(),nn.Linear(16,12),nn.ReLU(),nn.Linear(12,10))

    def execute(self, x):
        x_list = self.cut_data(x)
        change_list = []
        for i in range(4):
            cur = x_list[i]
            cur = jt.array(cur)
            cur = self.conv1(cur)
            cur = self.conv2(cur)
            cur = self.conv3(cur)
            cur = self.conv4(cur)
            change_list.append(cur)
        cur = jt.concat(change_list, dim=1)
        cur = self.mlp1(cur.view(cur.size(0),-1))
        cur = self.mlp2(cur)
        # cur = jt.reshape(cur, (cur.shape[0], 4, 4))
        cur = self.classifier(cur)
        
        return cur

    def cut_data(self,x):
        x1 = x[:, :, 0:16, 0:16]
        x2 = x[:, :, 0:16, 16:32]
        x3 = x[:, :, 16:32, 0:16]
        x4 = x[:, :, 16:32, 16:32]
   
        x_list = [x1,x2,x3,x4] 
        return x_list

# define DeepPermNet for 16*16 image
class DeepPermNet(nn.Module):
    def __init__(self):
        super(DeepPermNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                            out_channels=16,
                            kernel_size=3,
                            stride=2,
                            padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,3,2,1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32,64,3,2,1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64,64,2,2,0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.mlp1 = nn.Linear(4 * 64,128)
        self.mlp2 = nn.Linear(128,16)

        
        

    def execute(self, x):
        x_list,label = self.cut_data(x)
        self.label = label
        change_list = []
        for i in range(4):
            cur = x_list[i]
            cur = jt.array(cur)
            cur = self.conv1(cur)
            cur = self.conv2(cur)
            cur = self.conv3(cur)
            cur = self.conv4(cur)
            change_list.append(cur)
        cur = jt.concat(change_list, dim=1)
        cur = self.mlp1(cur.view(cur.size(0),-1))
        cur = self.mlp2(cur)
        cur = jt.reshape(cur, (cur.shape[0], 4, 4))
        cur = pygm.sinkhorn(cur)
  

        return cur


        
    def cut_data(self,x):
        x1 = x[:, :, 0:16, 0:16]
        x2 = x[:, :, 0:16, 16:32]
        x3 = x[:, :, 16:32, 0:16]
        x4 = x[:, :, 16:32, 16:32]
        # 打乱顺序
        label = [0,1,2,3]
        labels = []
        x1_ = np.zeros_like(x1)
        x2_ = np.zeros_like(x2)
        x3_ = np.zeros_like(x3)
        x4_ = np.zeros_like(x4)
        for i in range(x.shape[0]):
            x_list = [x1[i,:,:,:],x2[i,:,:,:],x3[i,:,:,:],x4[i,:,:,:]]
            np.random.shuffle(label)
            x_list1 = [x_list[i] for i in label]
            x1_[i,:,:,:] = x_list1[0].numpy()
            x2_[i,:,:,:] = x_list1[1].numpy()
            x3_[i,:,:,:] = x_list1[2].numpy()
            x4_[i,:,:,:] = x_list1[3].numpy()       
            labels.extend(label)
        labels = np.reshape(labels, (x.shape[0],4))   
        x_list = [x1_,x2_,x3_,x4_] 
        return x_list, labels
    
    def get_label(self):
        return self.label


#### 数据集 ####

data_dir = './deep_learning/TASK_2/dataset/cifar-10-batches-py'
train_batch_list = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def data_transform(data):
        data = np.array(data)
        data = data.reshape(3, 32, 32)
        data = data.transpose(1, 2, 0)
        return data
def load_train_data():
    train_data = []
    train_labels = []
    batch_data = []
    batch_label = []
    for batch in train_batch_list:
        cur_batch_data = unpickle(data_dir + '/' + batch)
        batch_data.append(cur_batch_data[b'data'])
        batch_label.append(cur_batch_data[b'labels'])
    batch_data = np.concatenate(batch_data)
    batch_label = np.concatenate(batch_label)
    for i in range(len(batch_data)):
        train_data.append(data_transform(batch_data[i]))
        train_labels.append(batch_label[i])   
    return train_data, train_labels

def load_test_data():
    test_data = []
    test_labels = []
    batch_data = unpickle(os.path.join(data_dir, 'test_batch'))
    for i in range(len(batch_data[b'data'])):
        test_data.append(data_transform(batch_data[b'data'][i]))
        test_labels.append(batch_data[b'labels'][i])
    return test_data, test_labels


 
def get_acc(y_hat, y):
    n, m = y_hat.shape[-2:]
    
    assert n == m

    e = np.ones((n, 1))
    Q = cp.Parameter((n, n))
    P_hat = cp.Variable((n, n), boolean=True)

    objective = cp.Minimize(
        cp.norm(P_hat - Q, p='fro')
    )
    constraints = [
        P_hat @ e == e
        , e.T @ P_hat == e.T
        , P_hat >= 0
        , P_hat <= 1
    ]
    problem = cp.Problem(objective, constraints)

    # Iterate through batch
    accuracy_batch = jt.zeros(y_hat.shape[0])
    for i, y_hat_i in enumerate(y_hat):
        # print(y_hat_i.numpy())
        Q.value = y_hat_i.numpy()
        problem.solve(solver='ECOS_BB')       
        acc = 0
        # print(out, y[i])
        if problem.status == 'optimal':
            a = P_hat.value
            idx = np.argpartition(a,-1,axis=1)[:,-1:]
            out = np.zeros(a.shape, dtype=int)
            np.put_along_axis(out,idx,1,axis=1)
            y_hat_i_bin = jt.array(out, dtype=jt.int)
            acc = (y_hat_i_bin * y[i]).sum() / n
        accuracy_batch[i] = acc
    return accuracy_batch.mean()

if __name__ == '__main__':
    x = jt.rand([1, 3, 32, 32])
    model = DeepPermNet()
    print('model')
    y = model(x)
    print(y.shape)
