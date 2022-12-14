from cgi import test
import jittor as jt
import jittor.dataset as dataset
from jittor import nn
from jittor import transform
import numpy as np
import os
import matplotlib.pyplot as plt
from Mymodel import mynet, ResNet18
from PIL import Image

root = './deep_learning/TASK_2/dataset'

# 模型参数

batch_size = 128
num_classes = 10
epochs = 100
learning_rate = 0.001
# device = jt.device('cuda' if jt.has_cuda else 'cpu')
model = mynet(num_classes)
model = ResNet18()

# 制作数据集

class Cutout(object):
     def __init__(self, hole_size):
         self.hole_size = hole_size
 
     def __call__(self, img):
         return cutout(img, self.hole_size)
 
 
def cutout(img, hole_size):
    y = np.random.randint(32)
    x = np.random.randint(32)
    half_size = hole_size // 2
    x1 = np.clip(x - half_size, 0, 32)
    x2 = np.clip(x + half_size, 0, 32)
    y1 = np.clip(y - half_size, 0, 32)
    y2 = np.clip(y + half_size, 0, 32)
    imgnp = np.array(img)
    imgnp[y1:y2, x1:x2] = 0
    img = Image.fromarray(imgnp.astype('uint8')).convert('RGB')
    return img

transformer = transform.Compose([
    transform.RandomCrop(32), 
    Cutout(6),
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

train_dataset = dataset.CIFAR10(root=root, train=True, download=False, transform=transformer)
test_dataset = dataset.CIFAR10(root=root, train=False, download=False, transform=test_transformer)
# 定义数据加载器
train_loader = train_dataset.set_attrs(batch_size=batch_size, shuffle=True)
test_loader = test_dataset.set_attrs(batch_size=batch_size, shuffle=False)

# 损失函数和优化器
loss = nn.CrossEntropyLoss()
optimizer = jt.optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
loss_all = []
total_step = int(len(train_loader) / batch_size)
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        outputs = model(images)
        l = loss(outputs, labels)
        optimizer.step(l)
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, epochs, i + 1, total_step, l.item()))
            loss_all.append(l.item())

plt.title('Loss')
plt.plot(range(0, len(loss_all)),loss_all)
plt.savefig('loss.png')

# 测试模型
model.eval()
with jt.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        predicted, _ = jt.argmax(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# 保存模型
jt.save(model.state_dict(), 'mynet.ckpt')