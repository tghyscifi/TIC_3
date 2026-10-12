import torch
import torch.nn as nn
import torch.nn.functional as F
class LeNet5(nn.Module):
    def __init__(self):
        '''构造函数，定义网络的结构'''
        super().__init__()
        #定义卷积层，1个输入通道，6个输出通道，5*5的卷积filter，外层补上了两圈0,因为输入的是32*32
        self.conv1 = nn.Conv2d(1, 6, 5)
        #第二个卷积层，6个输入，16个输出，5*5的卷积filter 
        self.conv2 = nn.Conv2d(6, 16, 5)

        #最后是三个全连接层
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        '''前向传播函数'''
        #先卷积，然后调用relu激活函数，再最大值池化操作
        print(x.shape)
        x = self.conv1(x)
        print(x.shape)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))
        print(x.shape)
        #第二次卷积+池化操作
        x = self.conv2(x)
        print(x.shape)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))
        print(x.shape)
        #重新塑形,将多维数据重新塑造为二维数据，256*400
        x = x.permute(0, 2, 3, 1)
        x = x.view(-1, self.num_flat_features(x))
        print('size', x.size())
        #第一个全连接
        x = F.relu(self.fc1(x))
        print(x.shape)
        x = F.relu(self.fc2(x))
        print(x.shape)
        x = self.fc3(x)
        print(x.shape)
        return x

    def num_flat_features(self, x):
        #x.size()返回值为(256, 16, 5, 5)，size的值为(16, 5, 5)，256是batch_size
        size = x.size()[1:]        #x.size返回的是一个元组，size表示截取元组中第二个开始的数字
        num_features = 1
        for s in size:
            num_features *= s
        return num_features 

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.uniform_(0, 1)
        m.weight.data = m.weight.data * 255
        m.weight.data=m.weight.data.float()
        # print(m.weight.data)
    elif isinstance(m, nn.Linear):
        m.weight.data.uniform_(0, 1)
        m.weight.data = m.weight.data * 255
        m.weight.data=m.weight.data.float()
        # print(m.weight.data)

x = torch.rand((1, 1, 32, 32))*255
x = x.float()
model = LeNet5()
model.apply(weight_init)
model.eval()

y0= model(x)
torch.save(model.state_dict(), 'parameter_lenet.pkl')