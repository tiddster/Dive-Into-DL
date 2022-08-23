# 加载数据及预处理
import torch as t
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from torch.autograd import Variable

show = ToPILImage()  # 可以把tensor转为image
# 第一次运行程序torchvision会自动下载CIFAR-10数据集
# 若已下载，可通过root参数指定
# 定义对数据的预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 转为Tensor，把灰度范围从0-255变换到0-1，归一化
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 把0-1变为-1到1，标准化
])

# 训练集
trainset = tv.datasets.CIFAR10(
    root='F:\Dataset\CIFAR',
    train=True,
    download=True,
    transform=transform)

trainloader = t.utils.data.DataLoader(  # 主要用来将自定义的数据读取接口的输出或者PyTorch已有的数据读取接口的输入按照batch size封装成Tensor，
    # 后续只需要再包装成Variable即可作为模型的输入
    trainset,
    batch_size=4,
    shuffle=True,
    num_workers=0  # 多线程
)

# 测试集
testset = tv.datasets.CIFAR10(
    root='F:\Dataset\CIFAR',
    train=False,
    download=True,
    transform=transform)

testloader = t.utils.data.DataLoader(
    testset,
    batch_size=4,
    shuffle=False,
    num_workers=0)

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

# 定义网络
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    # 把网络中具有可学习参数的层放在构造函数__inif__中
    def __init__(self):
        # 下式等价于nn.Module.__init__.(self)
        super(Net, self).__init__()  # RGB 3*32*32
        self.conv1 = nn.Conv2d(3, 15, 3)  # 输入3通道，输出15通道，卷积核为3*3
        self.conv2 = nn.Conv2d(15, 75, 4)  # 输入15通道，输出75通道，卷积核为4*4
        self.conv3 = nn.Conv2d(75, 375, 3)  # 输入75通道，输出375通道，卷积核为3*3
        self.fc1 = nn.Linear(1500, 400)  # 输入2000，输出400
        self.fc2 = nn.Linear(400, 120)  # 输入400，输出120
        self.fc3 = nn.Linear(120, 84)  # 输入120，输出84
        self.fc4 = nn.Linear(84, 10)  # 输入 84，输出 10（分10类）

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # 3*32*32  -> 150*30*30  -> 15*15*15
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # 15*15*15 -> 75*12*12  -> 75*6*6
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)  # 75*6*6   -> 375*4*4   -> 375*2*2
        x = x.view(x.size()[0], -1)  # 将375*2*2的tensor打平成1维，1500
        x = F.relu(self.fc1(x))  # 全连接层 1500 -> 400
        x = F.relu(self.fc2(x))  # 全连接层 400 -> 120
        x = F.relu(self.fc3(x))  # 全连接层 120 -> 84
        x = self.fc4(x)  # 全连接层 84  -> 10
        return x


net = Net()
print(net)

# 定义损失函数和优化器

from torch import optim

criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # 优化器：随机梯度下降

# 训练网络

for epoch in range(8):  # 训练8次
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # enumerate() 函数：用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中
        # enumerate(sequence, [start=0])
        # sequence -- 一个序列、迭代器或其他支持迭代对象。
        # start -- 下标起始位置。

        # 输入数据
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)

        # 梯度清零
        optimizer.zero_grad()

        # forward + backward
        outputs = net(inputs)
        loss = criterion(outputs, labels)  # 计算单个batch误差
        loss.backward()  # 反向传播

        # 更新参数
        optimizer.step()

        # 打印log信息
        running_loss += loss.item()  # 2000个batch的误差和
        if i % 2000 == 1999:  # 每2000个batch打印一次训练状态
            print("[%d,%5d] loss: %.3f" \
                  % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
print("Finished Training")

# 测试网络

dataiter = iter(testloader)  # 采用iter(dataloader)返回的是一个迭代器，然后可以使用next()访问。iter(dataloader)访问时，
# imgs在前，labels在后，分别表示：图像转换0~1之间的值，labels为标签值。并且imgs和labels是按批次进行输入的。
images, labels = dataiter.next()
print("实际的label:", " ".join( \
    "%08s" % classes[labels[j]] for j in range(4)))
show(tv.utils.make_grid(images / 2 - 0.5)).resize((400, 100))

# 计算图片在每个类别上的分数
outputs = net(Variable(images))

# 得分最高的那个类
_, predicted = t.max(outputs.data, 1)  # torch.max()返回两个值，第一个值是具体的value，，也就是输出的最大值（我们用下划线_表示 ，指概率），
# 第二个值是value所在的index（也就是predicted ， 指类别）
# 选用下划线代表不需要用到的变量
# 数字1：其实可以写为dim=1，表示输出所在行的最大值，若改写成dim=0则输出所在列的最大值
print("预测结果:", " ".join("%5s" \
                        % classes[predicted[j]] for j in range(4)))

correct = 0  # 预测正确图片数
total = 0  # 总图片数
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = t.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
print("10000张测试集中的准确率为:%d %%" % (100 * correct / total))

# gpu加速
if t.cuda.is_available():
    net.cuda()
    images = images.cuda()
    labels = labels.cuda()
    outputs = net(Variable(images))
    loss = criterion(outputs, Variable(labels))