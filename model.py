import sys
import torch
import torch.nn as nn
from torch import optim
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dropout = 0.5
batch_size = 4
learning_rate = 0.001
num_epoch = 10



transform = transforms.Compose(
    [transforms.RandomResizedCrop(224),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))] #RGB三个通道都进行张量归一化
)

# 请于别处下载
# train_dataset = CIFAR10('./data', train=True, transform=transform, download=True)
# test_dataset = CIFAR10('./data', train=False, transform=transform)


# 直接使用绝对路径
data_root = r'D:\桌面\CIFAR10\data'

# 确保数据集路径正确
if not os.path.exists(data_root):
    raise RuntimeError(f"Dataset not found at {data_root}. Please ensure the path is correct.")

# 加载训练数据集（不下载）
train_dataset = CIFAR10(root=data_root, train=True, transform=transform, download=False)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# 加载测试数据集（不下载）
test_dataset = CIFAR10(root=data_root, train=False, transform=transform, download=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)




class  VGG(nn.Module):
    def __init__(self,features,init_weights = False,num_classes = 10):
        super(VGG,self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()


    def forward(self,x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)



def make_features(cfg: list):
    layers = []
    in_channels = 3
    for v in cfg :
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size = 2, stride = 2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    # 非关键字参数输入，转化成元组方式
    return nn.Sequential(*layers)

cfgs = {
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}

#实例化vgg网络，看是cfgs中哪个网络被使用
def vgg(model_name = "vgg16", **kwargs):
    try:
        cfg = cfgs[model_name]
    except:
        print("warning")
        exit(-1)
    model = VGG(make_features(cfg), **kwargs)
    return model

model_name = "vgg16"
model = vgg(model_name=model_name, num_classes=10, init_weights=True)
model.to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(num_epoch):
    # 训练模式，可以改变参数
    model.train()
    running_loss = 0
    train_bar = tqdm(train_loader, file=sys.stdout)
    for step, data in enumerate(train_bar):
        images, labels = data
        # 让梯度为0
        optimizer.zero_grad()
        # forward+ backward + loss
        outputs = model(images.to(device))
        loss = loss_function(outputs,labels.to(device))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        train_bar.set_description(f"train epoch[{epoch + 1}/{num_epoch}] loss:{loss.item():.3f}")


    # 验证模式，参数不得改变
    model.eval()
    acc = 0
    with torch.no_grad():
        test_bar = tqdm(test_loader, file=sys.stdout)
        for test_data in test_bar:
            test_image, test_labels = test_data
            outputs = model(test_image.to(device))
            predict = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict, test_labels.to(device)).sum().item()

        test_acc =  acc / len(test_loader.dataset)
        print(f'[epoch {epoch + 1}] train_loss: {running_loss / len(train_loader):.3f} test_acc: {test_acc:.3f}')

